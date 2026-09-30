"""Build the structure inventory and rank exposed, fold-distinct MSA pockets."""

import csv
import json
from pathlib import Path

import freesasa
import numpy as np
from Bio.PDB import PDBParser


ROOT = Path(__file__).resolve().parent
PDB_DIR = ROOT / "data" / "pdb"
RESULTS = ROOT / "results"
STRUCTURES = {
    "6XYO": ("MSA", "Type I", "A,C,E,G,I / B,D,F,H,J"),
    "6XYP": ("MSA", "Type II-1", "A,C,E,G,I / B,D,F,H,J"),
    "6XYQ": ("MSA", "Type II-2", "A,C,E,G,I / B,D,F,H,J"),
    "8A9L": ("PD/PDD/DLB", "Lewy fold", "A"),
    "6CU7": ("in-vitro alpha-synuclein", "rod polymorph", "ten chains"),
    "8Q2L": ("tau control", "AD tau filament", "six chains"),
    "6SHS": ("amyloid-beta control", "Abeta morphology I", "twelve chains"),
}


def atom_rows(chain):
    return {
        residue.id[1]: residue["CA"].coord
        for residue in chain
        if residue.id[0] == " " and "CA" in residue
    }


def residue_areas(pdb_path, chain):
    structure = freesasa.Structure(str(pdb_path))
    areas = freesasa.calc(structure).residueAreas()[chain]
    return {int(k.strip()): v.total for k, v in areas.items()}


def main():
    RESULTS.mkdir(exist_ok=True)
    parser = PDBParser(QUIET=True)
    models = {key: parser.get_structure(key, PDB_DIR / f"{key}.pdb")[0] for key in STRUCTURES}

    inventory = []
    for key, (disease, fold, chains) in STRUCTURES.items():
        for chain in models[key]:
            ids = [r.id[1] for r in chain if r.id[0] == " " and "CA" in r]
            if ids:
                inventory.append({"pdb": key, "disease": disease, "fold": fold,
                                  "chain": chain.id, "residue_start": min(ids),
                                  "residue_end": max(ids), "residue_count": len(ids),
                                  "assembly_note": chains,
                                  "source": f"https://www.rcsb.org/structure/{key}"})
    with (RESULTS / "structure_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory[0])
        writer.writeheader()
        writer.writerows(inventory)

    msa_ca = atom_rows(models["6XYO"]["C"])
    lewy_ca = atom_rows(models["8A9L"]["A"])
    msa_areas = {key: residue_areas(PDB_DIR / f"{key}.pdb", "C")
                 for key in ("6XYO", "6XYP", "6XYQ")}
    area = msa_areas["6XYO"]
    common = sorted(set(msa_ca) & set(lewy_ca))
    residues = []
    for number in common:
        if number < 35 or number > 91:
            continue
        peers = [other for other in common if abs(number - other) >= 4]
        msa_dist = np.array([np.linalg.norm(msa_ca[number] - msa_ca[other]) for other in peers])
        lewy_dist = np.array([np.linalg.norm(lewy_ca[number] - lewy_ca[other]) for other in peers])
        local = (msa_dist < 12) | (lewy_dist < 12)
        novelty = float(np.mean(np.abs(msa_dist[local] - lewy_dist[local]))) if local.any() else 0.0
        exposed = area.get(number, 0.0)
        common_exposed = min(msa_areas[key].get(number, 0.0) for key in msa_areas)
        score = exposed * min(novelty, 10.0) / 10.0
        residues.append({"residue": number, "msa_sasa_a2": round(exposed, 2),
                         "msa_common_sasa_a2": round(common_exposed, 2),
                         "fold_distance_a": round(novelty, 2), "priority": round(score, 2)})
    with (RESULTS / "residue_surface_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=residues[0])
        writer.writeheader()
        writer.writerows(residues)

    selected = []
    for row in sorted(residues, key=lambda item: item["priority"], reverse=True):
        center = row["residue"]
        if row["msa_common_sasa_a2"] < 50:
            continue
        if any(np.linalg.norm(msa_ca[center] - msa_ca[prior["center"]]) < 13 for prior in selected):
            continue
        pocket = []
        for chain_id in ("A", "C", "E"):
            for residue in models["6XYO"][chain_id]:
                if residue.id[0] != " " or "CA" not in residue:
                    continue
                residue_id = residue.id[1]
                if residue_id not in area or area[residue_id] < 15:
                    continue
                distance = np.linalg.norm(residue["CA"].coord - msa_ca[center])
                if distance < 10:
                    pocket.append([chain_id, [residue_id, " "]])
        if len(pocket) < 8:
            continue
        site = f"site_{len(selected) + 1}"
        (RESULTS / f"{site}_pocket.json").write_text(json.dumps(pocket, indent=2) + "\n")
        selected.append({"site": site, "center": center, "pocket_residues": len(pocket),
                         "msa_sasa_a2": row["msa_sasa_a2"],
                         "msa_common_sasa_a2": row["msa_common_sasa_a2"],
                         "fold_distance_a": row["fold_distance_a"],
                         "priority": row["priority"],
                         "pocket_file": str(RESULTS / f"{site}_pocket.json")})
        if len(selected) == 4:
            break
    with (RESULTS / "selected_epitopes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0])
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
