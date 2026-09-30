"""Relax peptide complexes on GPU while retaining the experimental fibril scaffold."""

import argparse
import json
from pathlib import Path

import openmm
from openmm import app, unit
from pdbfixer import PDBFixer


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def minimize(candidate, gpu):
    fixer = PDBFixer(filename=candidate["pdb"])
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    forcefield = app.ForceField("charmm36.xml")
    system = forcefield.createSystem(fixer.topology,
                                     nonbondedMethod=app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.2 * unit.nanometer,
                                     constraints=app.HBonds)
    restraints = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    for name in ("k", "x0", "y0", "z0"):
        restraints.addPerParticleParameter(name)
    for index, atom in enumerate(fixer.topology.atoms()):
        if atom.element.symbol == "H":
            continue
        stiffness = 20.0 if atom.residue.chain.id == candidate["peptide_chain"] else 1000.0
        x, y, z = fixer.positions[index].value_in_unit(unit.nanometer)
        restraints.addParticle(index, [stiffness, x, y, z])
    system.addForce(restraints)
    integrator = openmm.LangevinIntegrator(0 * unit.kelvin, 1 / unit.picosecond,
                                          0.002 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("OpenCL")
    simulation = app.Simulation(fixer.topology, system, integrator, platform,
                                {"DeviceIndex": str(gpu), "Precision": "mixed"})
    simulation.context.setPositions(fixer.positions)
    initial = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    simulation.minimizeEnergy(tolerance=10 * unit.kilojoule_per_mole / unit.nanometer,
                              maxIterations=300)
    final_state = simulation.context.getState(getPositions=True, getEnergy=True)
    final = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    path = RESULTS / "relaxed" / candidate["site"] / f"{candidate['id']}.pdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), handle, keepIds=True)
    return {"id": candidate["id"], "site": candidate["site"], "pdb": str(path),
            "energy_initial_kj_mol": round(initial, 2),
            "energy_final_kj_mol": round(final, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    log = args.log or RESULTS / f"minimization_gpu{args.gpu}.jsonl"
    with log.open("w") as handle:
        for site in args.sites:
            candidates = [json.loads(line) for line in (RESULTS / "generated" / site / "candidates.jsonl").read_text().splitlines()]
            for index, candidate in enumerate(candidates, 1):
                result = minimize(candidate, args.gpu)
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                print(f"{site}: {index}/{len(candidates)} energy {result['energy_final_kj_mol']}", flush=True)


if __name__ == "__main__":
    main()
