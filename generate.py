"""Generate peptide sequence and coordinates with the published PepGLAD weights."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "PepGLAD"
sys.path.insert(0, str(VENDOR))

from data.codesign import calculate_covariance_matrix  # noqa: E402
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb  # noqa: E402
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks  # noqa: E402
from data.format import Atom, Block, VOCAB  # noqa: E402
from utils.const import sidechain_atoms  # noqa: E402


def model_input(pocket, full_chains, length):
    selected, positions = [], []
    pocket_lookup = {(chain, number, insert) for chain, (number, insert) in pocket}
    for chain_id, blocks in full_chains.items():
        for index, block in enumerate(blocks):
            if (chain_id, block.id[0], block.id[1]) in pocket_lookup:
                selected.append(block)
                positions.append(index + 1)
    peptide = [Block(VOCAB.symbol_to_abrv(VOCAB.UNK), [Atom("CA", [0, 0, 0], "C")]) for _ in range(length)]
    mask = torch.tensor([False] * len(selected) + [True] * length)
    positions.extend(range(1, length + 1))
    x, s, atom_mask = [], [], []
    for block in selected + peptide:
        symbol = VOCAB.abrv_to_symbol(block.abrv)
        atom2coord = {atom.name: atom.get_coord() for atom in block.units}
        centroid = np.mean(list(atom2coord.values()), axis=0).tolist()
        coords, present = [], []
        for atom_name in VOCAB.backbone_atoms + sidechain_atoms.get(symbol, []):
            coords.append(atom2coord.get(atom_name, centroid))
            present.append(atom_name in atom2coord)
        coords.extend([centroid] * (14 - len(coords)))
        present.extend([False] * (14 - len(present)))
        x.append(coords)
        s.append(VOCAB.symbol_to_idx(symbol))
        atom_mask.append(present)
    x = torch.tensor(x, dtype=torch.float)
    atom_mask = torch.tensor(atom_mask, dtype=torch.bool)
    cov = calculate_covariance_matrix(x[~mask][:, 1][atom_mask[~mask][:, 1]].numpy())
    cov = cov + 1e-4 * np.identity(cov.shape[0])
    return {"X": x, "S": torch.tensor(s, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
            "mask": mask, "atom_mask": atom_mask,
            "lengths": len(s), "L": torch.from_numpy(np.linalg.cholesky(cov)).float().unsqueeze(0)}


def batch_inputs(samples):
    batch = {}
    for name in ("X", "S", "position_ids", "mask", "atom_mask", "L"):
        batch[name] = torch.cat([sample[name] for sample in samples])
    batch["lengths"] = torch.tensor([sample["lengths"] for sample in samples], dtype=torch.long)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    pdb_path = ROOT / "data" / "pdb" / "6XYO.pdb"
    pocket = json.loads((ROOT / "results" / f"{args.site}_pocket.json").read_text())
    chains = list(dict.fromkeys(item[0] for item in pocket))
    selected_chains = pdb_to_list_blocks(str(pdb_path), chains, dict_form=True)
    all_chains = pdb_to_list_blocks(str(pdb_path), dict_form=True)
    model = torch.load(VENDOR / "checkpoints" / "codesign.ckpt", map_location="cpu")
    model.to(device)
    model.eval()

    output = ROOT / "results" / "generated" / args.site
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "candidates.jsonl"
    with summary.open("w") as handle, torch.no_grad():
        for start in range(0, args.samples, args.batch_size):
            sample_count = min(args.batch_size, args.samples - start)
            lengths = np.random.randint(10, 26, size=sample_count)
            inputs = [model_input(pocket, selected_chains, int(length)) for length in lengths]
            batch = {key: value.to(device) for key, value in batch_inputs(inputs).items()}
            x_out, s_out, _ = model.sample(
                batch["X"], batch["S"], batch["mask"],
                batch["position_ids"], batch["lengths"], batch["atom_mask"],
                L=batch["L"], sample_opt={"energy_func": "default", "energy_lambda": 0.8})
            for offset, (x, s) in enumerate(zip(x_out, s_out)):
                peptide = []
                for coord, symbol in zip(x, s):
                    abrv = VOCAB.symbol_to_abrv(symbol)
                    atoms = VOCAB.backbone_atoms + sidechain_atoms[VOCAB.abrv_to_symbol(abrv)]
                    peptide.append(Block(abrv, [Atom(name, value, name[0]) for name, value in zip(atoms, coord)]))
                name = f"{args.site}_{start + offset:03d}"
                peptide_chain = "K"
                pdb_out = output / f"{name}.pdb"
                list_blocks_to_pdb(list(all_chains.values()) + [peptide],
                                   list(all_chains.keys()) + [peptide_chain], str(pdb_out))
                sequence = "".join(VOCAB.abrv_to_symbol(block.abrv) for block in peptide)
                handle.write(json.dumps({"id": name, "site": args.site, "sequence": sequence,
                                         "length": len(sequence), "pdb": str(pdb_out),
                                         "target_pdb": "6XYO", "peptide_chain": peptide_chain}) + "\n")
                handle.flush()
            print(f"{args.site}: {start + sample_count}/{args.samples}", flush=True)


if __name__ == "__main__":
    main()
