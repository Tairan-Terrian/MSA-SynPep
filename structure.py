"""GPU fixed-sequence pose sampling on five conformers and independent geometry evidence."""

import argparse
import json
import random
import time
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import openmm
from openmm import app, unit
from pdbfixer import PDBFixer

from design import ROOT, TARGETS, read_jsonl, write_jsonl

VENDOR = ROOT / "vendor/PepGLAD"
sys.path.insert(1, str(VENDOR))
from data.codesign import calculate_covariance_matrix
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.format import Atom, Block, VOCAB
from utils.const import sidechain_atoms

FORCEFIELD = None


def model_input(pocket, receptor, length, sequence):
    lookup = {(chain, number, insert) for chain, (number, insert) in pocket}
    selected, positions = [], []
    for chain_id, blocks in receptor.items():
        for index, block in enumerate(blocks):
            if (chain_id, block.id[0], block.id[1]) in lookup:
                selected.append(block)
                positions.append(index + 1)
    peptide = [Block(VOCAB.symbol_to_abrv(letter),
                     [Atom(name, [0, 0, 0], name[0]) for name in VOCAB.backbone_atoms + sidechain_atoms[letter]])
               for letter in sequence]
    mask = torch.tensor([False] * len(selected) + [True] * length)
    positions.extend(range(1, length + 1))
    coordinates, symbols, atom_masks = [], [], []
    for block in selected + peptide:
        symbol = VOCAB.abrv_to_symbol(block.abrv)
        atoms = {atom.name: atom.get_coord() for atom in block.units}
        centroid = np.mean(list(atoms.values()), axis=0).tolist()
        names = VOCAB.backbone_atoms + sidechain_atoms.get(symbol, [])
        coordinates.append([atoms.get(name, centroid) for name in names] + [centroid] * (14 - len(names)))
        atom_masks.append([name in atoms for name in names] + [False] * (14 - len(names)))
        symbols.append(VOCAB.symbol_to_idx(symbol))
    x = torch.tensor(coordinates, dtype=torch.float)
    atom_mask = torch.tensor(atom_masks, dtype=torch.bool)
    covariance = calculate_covariance_matrix(x[~mask][:, 1][atom_mask[~mask][:, 1]].numpy()) + 1e-4 * np.eye(3)
    return {"X": x, "S": torch.tensor(symbols, dtype=torch.long), "position_ids": torch.tensor(positions),
            "mask": mask, "atom_mask": atom_mask, "lengths": len(symbols),
            "L": torch.from_numpy(np.linalg.cholesky(covariance)).float().unsqueeze(0)}


def batch_inputs(samples):
    batch = {name: torch.cat([s[name] for s in samples]) for name in ("X", "S", "position_ids", "mask", "atom_mask", "L")}
    batch["lengths"] = torch.tensor([s["lengths"] for s in samples], dtype=torch.long)
    return batch


def max_hydrophobic_run(sequence):
    longest = current = 0
    for letter in sequence:
        current = current + 1 if letter in "IVLFWYM" else 0
        longest = max(longest, current)
    return longest


def initialize_relaxation(gpu):
    global FORCEFIELD
    platform = openmm.Platform.getPlatformByName("OpenCL")
    platform.setPropertyDefaultValue("DeviceIndex", str(gpu))
    platform.setPropertyDefaultValue("Precision", "mixed")
    FORCEFIELD = app.ForceField("charmm36.xml")


def atom_data(chains, device):
    coords, residues, polar, positive, negative = [], [], [], [], []
    residue_index = 0
    for chain in chains:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                if atom.element == "H":
                    continue
                coords.append(atom.coord)
                residues.append(residue_index)
                polar.append(atom.element in ("N", "O"))
                positive.append(residue.resname in ("LYS", "ARG") and atom.name in ("NZ", "NH1", "NH2", "NE"))
                negative.append(residue.resname in ("ASP", "GLU") and atom.name in ("OD1", "OD2", "OE1", "OE2"))
            residue_index += 1
    return {"xyz": torch.tensor(np.asarray(coords), dtype=torch.float32, device=device),
            "residue": torch.tensor(residues, device=device),
            "polar": torch.tensor(polar, device=device),
            "positive": torch.tensor(positive, device=device),
            "negative": torch.tensor(negative, device=device)}


def metrics(peptide, central, assembly, length):
    distance = torch.cdist(peptide["xyz"], central["xyz"])
    pairs = peptide["residue"][:, None] * (int(central["residue"].max()) + 1) + central["residue"][None, :]
    contacts = torch.unique(pairs[distance < 4.5]).numel()
    polar = ((distance < 3.5) & peptide["polar"][:, None] & central["polar"][None, :]).sum().item()
    opposite = (((peptide["positive"][:, None] & central["negative"][None, :]) |
                 (peptide["negative"][:, None] & central["positive"][None, :])) & (distance < 4.5)).sum().item()
    assembly_distance = torch.cdist(peptide["xyz"], assembly["xyz"])
    return {"contacts_per_residue": round(contacts / length, 3),
            "clashes_per_residue": round((distance < 2).sum().item() / length, 3),
            "assembly_clashes_per_residue": round((assembly_distance < 2).sum().item() / length, 3),
            "polar_pairs_per_residue": round(polar / length, 3),
            "opposite_charge_pairs_per_residue": round(opposite / length, 3)}


def relax(raw, output, gpu):
    fixer = PDBFixer(filename=str(raw))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    system = FORCEFIELD.createSystem(fixer.topology, nonbondedMethod=app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.2 * unit.nanometer, constraints=app.HBonds)
    force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    for name in ("k", "x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    for i, atom in enumerate(fixer.topology.atoms()):
        if atom.element.symbol != "H":
            force.addParticle(i, [20.0 if atom.residue.chain.id == "K" else 1000.0,
                                  *fixer.positions[i].value_in_unit(unit.nanometer)])
    system.addForce(force)
    integrator = openmm.LangevinIntegrator(0 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(fixer.topology, system, integrator,
                                openmm.Platform.getPlatformByName("OpenCL"),
                                {"DeviceIndex": str(gpu), "Precision": "mixed"})
    simulation.context.setPositions(fixer.positions)
    simulation.minimizeEnergy(tolerance=10 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=0)
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    with output.open("w") as handle:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), handle, keepIds=True)
    return round(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), 2)


def worker(args):
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    random.seed(args.seed + args.gpu)
    np.random.seed(args.seed + args.gpu)
    torch.manual_seed(args.seed + args.gpu)
    parser = PDBParser(QUIET=True)
    model = torch.load(VENDOR / "checkpoints/fixseq.ckpt", map_location="cpu").to(device).eval()
    contexts = {x["site"]: x for x in json.loads((args.run / "epitope_context.json").read_text())}
    assembly_context = json.loads((args.run / "assembly_context.json").read_text())
    candidates = read_jsonl(args.candidates)[args.gpu::2]
    output = args.run / "poses" / args.phase
    output.mkdir(parents=True, exist_ok=True)
    log = args.run / f"{args.phase}_gpu{args.gpu}_poses.jsonl"
    completed = {(row["id"], row["target"]) for row in read_jsonl(log)} if args.resume_poses and log.exists() else set()
    start_time = time.monotonic()
    pending = []
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_relaxation, initargs=(args.gpu,)) as pool, log.open("a" if args.resume_poses else "w") as handle:
        for key in TARGETS:
            central_id = assembly_context[key]["central_chain"]
            model_chains = assembly_context[key]["model_chains"]
            original = parser.get_structure(key, args.run / "targets" / f"{key}.pdb")[0]
            receptor = pdb_to_list_blocks(str(args.run / "targets" / f"{key}.pdb"), model_chains, dict_form=True)
            target_candidates = [c for c in candidates if (c["id"], key) not in completed]
            for start in range(0, len(target_candidates), args.batch_size):
                batch_candidates = target_candidates[start:start + args.batch_size]
                samples = [model_input([[chain, [n, " "]] for chain in model_chains
                                        for n in contexts[c["site"]]["common_anchor_residues"]],
                                       receptor, len(c["sequence"]), c["sequence"]) for c in batch_candidates]
                inputs = {k: v.to(device) for k, v in batch_inputs(samples).items()}
                with torch.no_grad():
                    xyzs, _, _ = model.sample(inputs["X"], inputs["S"], inputs["mask"],
                                              inputs["position_ids"], inputs["lengths"], inputs["atom_mask"],
                                              L=inputs["L"], sample_opt={"energy_func": "default", "energy_lambda": 0.5})
                for candidate, xyz in zip(batch_candidates, xyzs):
                    peptide = [Block(VOCAB.symbol_to_abrv(letter),
                                     [Atom(name, coord, name[0]) for name, coord in
                                      zip(VOCAB.backbone_atoms + sidechain_atoms[letter], coords)])
                               for letter, coords in zip(candidate["sequence"], xyz)]
                    name = f"{candidate['id']}_{key}_rep{args.repeat}"
                    raw, relaxed = output / f"{name}_raw.pdb", output / f"{name}.pdb"
                    list_blocks_to_pdb([receptor[chain] for chain in model_chains] + [peptide], model_chains + ["K"], str(raw))
                    future = pool.submit(relax, raw, relaxed, args.gpu)
                    pending.append((future, candidate, key, central_id, relaxed, name))
                print(f"{args.phase} GPU{args.gpu} sampled {key} {start + len(batch_candidates)}/{len(candidates)} "
                      f"elapsed={time.monotonic() - start_time:.1f}s", flush=True)
        for index, (future, candidate, key, central_id, relaxed, name) in enumerate(pending, 1):
            energy = future.result()
            handle.write(json.dumps({"id": candidate["id"], "target": key, "repeat": args.repeat,
                                     "pdb": str(relaxed), "energy_kj_mol": energy}) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(pending):
                print(f"{args.phase} GPU{args.gpu} relaxed {index}/{len(pending)} "
                      f"elapsed={time.monotonic() - start_time:.1f}s", flush=True)


def collect(args):
    candidates = read_jsonl(args.candidates)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    parser = PDBParser(QUIET=True)
    models = {key: parser.get_structure(key, args.run / "targets" / f"{key}.pdb")[0] for key in TARGETS}
    assembly_context = json.loads((args.run / "assembly_context.json").read_text())
    candidate_map = {c["id"]: c for c in candidates}
    records = []
    for path in sorted(args.run.glob(f"{args.phase}_gpu*_poses.jsonl")):
        updated = []
        for row in read_jsonl(path):
            key = row["target"]
            candidate = candidate_map[row["id"]]
            structure = parser.get_structure(row["id"], ROOT / row["pdb"])[0]
            chains = assembly_context[key]["model_chains"]
            # Assemble a single consistent coordinate state: relaxed modeled layers plus untouched other layers.
            assembly = atom_data([structure[c] for c in chains] + [c for c in models[key] if c.id not in chains], device)
            values = metrics(atom_data([structure["K"]], device), atom_data([structure["C"]], device),
                             assembly, len(candidate["sequence"]))
            reference_ca = [r["CA"].coord for c in chains for r in models[key][c] if r.id[0] == " " and "CA" in r]
            relaxed_ca = [r["CA"].coord for c in chains for r in structure[c] if r.id[0] == " " and "CA" in r]
            delta = torch.tensor(np.asarray(reference_ca), device=device) - torch.tensor(np.asarray(relaxed_ca), device=device)
            values["receptor_backbone_rmsd_a"] = round(torch.sqrt((delta ** 2).sum(-1).mean()).item(), 3)
            peptide_residues = list(structure["K"].get_residues())
            peptide_c = torch.tensor(np.array([r["C"].coord for r in peptide_residues[:-1]]), device=device)
            peptide_n = torch.tensor(np.array([r["N"].coord for r in peptide_residues[1:]]), device=device)
            values["peptide_cn_max_deviation_a"] = round(
                (torch.linalg.norm(peptide_c - peptide_n, dim=-1) - 1.33).abs().max().item(), 3)
            updated.append({**row, **values})
        write_jsonl(path, updated)
        records.extend(updated)
    write_jsonl(args.run / f"{args.phase}_geometry.jsonl", records)
    print(f"Measured {len(records)} independent poses on CUDA", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("worker", "collect"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--resume-poses", action="store_true", help="Reuse completed poses from this named production stage")
    args = parser.parse_args()
    {"worker": worker, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
