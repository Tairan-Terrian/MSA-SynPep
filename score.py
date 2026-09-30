"""Rank designed peptides by target contacts and transferred-pose counter-screens."""

import csv
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PDB_DIR = ROOT / "data" / "pdb"
PARSER = PDBParser(QUIET=True)


def chain_ca(chain):
    return {res.id[1]: res["CA"].coord for res in chain if res.id[0] == " " and "CA" in res}


def atom_table(chains):
    coords, labels = [], []
    for chain in chains:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                if atom.element != "H":
                    coords.append(atom.coord)
                    labels.append((chain.id, residue.id[1]))
    return np.asarray(coords), labels


def interface(peptide_xyz, peptide_labels, receptor_xyz, receptor_labels):
    tree = cKDTree(receptor_xyz)
    nearby = tree.query_ball_point(peptide_xyz, 4.5)
    contact_pairs = set()
    clash_pairs = 0
    for atom_index, partners in enumerate(nearby):
        for partner in partners:
            distance = np.linalg.norm(peptide_xyz[atom_index] - receptor_xyz[partner])
            contact_pairs.add((peptide_labels[atom_index], receptor_labels[partner]))
            clash_pairs += distance < 2.0
    return len(contact_pairs), int(clash_pairs)


def align_transfer(peptide_xyz, reference_ca, target_ca, anchor_numbers):
    shared = [number for number in anchor_numbers if number in reference_ca and number in target_ca]
    reference = np.asarray([reference_ca[number] for number in shared])
    target = np.asarray([target_ca[number] for number in shared])
    x0, y0 = reference.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((reference - x0).T @ (target - y0))
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    transformed = (peptide_xyz - x0) @ rotation + y0
    rmsd = np.sqrt(np.mean(np.sum(((reference - x0) @ rotation + y0 - target) ** 2, axis=1)))
    return transformed, len(shared), float(rmsd)


def max_hydrophobic_run(sequence):
    longest = current = 0
    for letter in sequence:
        current = current + 1 if letter in "IVLFWYM" else 0
        longest = max(longest, current)
    return longest


def main():
    minimization = {}
    for log in RESULTS.glob("minimization_gpu*.jsonl"):
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            minimization[entry["id"]] = entry
    structures = {key: PARSER.get_structure(key, PDB_DIR / f"{key}.pdb")[0]
                  for key in ("6XYO", "6XYP", "6XYQ", "8A9L", "6CU7")}
    other_msa = {
        "msa_ii1": (chain_ca(structures["6XYP"]["C"]), atom_table(structures["6XYP"]),
                    atom_table([structures["6XYP"]["C"]])),
        "msa_ii2": (chain_ca(structures["6XYQ"]["C"]), atom_table(structures["6XYQ"]),
                    atom_table([structures["6XYQ"]["C"]])),
    }
    negatives = {
        "lewy": (chain_ca(structures["8A9L"]["A"]), atom_table([structures["8A9L"]["A"]])),
        "pff": (chain_ca(structures["6CU7"]["C"]), atom_table(structures["6CU7"])),
    }
    site_anchors = {}
    for site_num in range(1, 5):
        site = f"site_{site_num}"
        pocket = json.loads((RESULTS / f"{site}_pocket.json").read_text())
        site_anchors[site] = sorted({number for chain, (number, _) in pocket if chain == "C"})

    rows = []
    summaries = sorted((RESULTS / "generated").glob("site_*/candidates.jsonl"))
    for summary in summaries:
        for line in summary.read_text().splitlines():
            candidate = json.loads(line)
            generated_pdb = candidate["pdb"]
            candidate["pdb"] = str(RESULTS / "relaxed" / candidate["site"] / f"{candidate['id']}.pdb")
            model = PARSER.get_structure(candidate["id"], candidate["pdb"])[0]
            peptide_xyz, peptide_atom_labels = atom_table([model[candidate["peptide_chain"]]])
            peptide_labels = [number for _, number in peptide_atom_labels]
            msa_full = atom_table([chain for chain in model if chain.id != candidate["peptide_chain"]])
            msa_c = atom_table([model["C"]])
            original_numbers = [res.id[1] for res in structures["6XYO"]["C"]
                                if res.id[0] == " " and "CA" in res]
            relaxed_residues = [res for res in model["C"] if res.id[0] == " " and "CA" in res]
            msa_ca = {number: residue["CA"].coord
                      for number, residue in zip(original_numbers, relaxed_residues)}
            target_contacts, target_clashes = interface(peptide_xyz, peptide_labels, *msa_full)
            chain_contacts, chain_clashes = interface(peptide_xyz, peptide_labels, *msa_c)
            sequence = candidate["sequence"]
            length = len(sequence)
            record = {"id": candidate["id"], "site": candidate["site"],
                      "sequence": sequence, "length": length, "pdb": candidate["pdb"],
                      "generated_pdb": generated_pdb,
                      "energy_final_kj_mol": minimization[candidate["id"]]["energy_final_kj_mol"],
                      "msa_contacts": target_contacts, "msa_clashes": target_clashes,
                      "msa_chain_contacts": chain_contacts, "msa_chain_clashes": chain_clashes}
            for label, (ca, full_atoms, central_atoms) in other_msa.items():
                transferred, anchors, rmsd = align_transfer(
                    peptide_xyz, msa_ca, ca, site_anchors[candidate["site"]])
                contacts, clashes = interface(transferred, peptide_labels, *full_atoms)
                central_contacts, central_clashes = interface(transferred, peptide_labels, *central_atoms)
                record[f"{label}_contacts"] = contacts
                record[f"{label}_clashes"] = clashes
                record[f"{label}_chain_contacts"] = central_contacts
                record[f"{label}_chain_clashes"] = central_clashes
                record[f"{label}_anchor_count"] = anchors
                record[f"{label}_anchor_rmsd_a"] = round(rmsd, 3)
            for label, (ca, atom_data) in negatives.items():
                transferred, anchors, rmsd = align_transfer(
                    peptide_xyz, msa_ca, ca, site_anchors[candidate["site"]])
                contacts, clashes = interface(transferred, peptide_labels, *atom_data)
                record[f"{label}_contacts"] = contacts
                record[f"{label}_clashes"] = clashes
                record[f"{label}_anchor_count"] = anchors
                record[f"{label}_anchor_rmsd_a"] = round(rmsd, 3)
            analysis = ProteinAnalysis(sequence)
            record["gravy"] = round(analysis.gravy(), 3)
            record["hydrophobic_run"] = max_hydrophobic_run(sequence)
            full_target_score = min(
                (record[f"{label}_contacts"] - 3 * record[f"{label}_clashes"]) / length
                for label in ("msa", "msa_ii1", "msa_ii2"))
            chain_target_score = min(
                (record[f"{label}_chain_contacts"] - 3 * record[f"{label}_chain_clashes"]) / length
                for label in ("msa", "msa_ii1", "msa_ii2"))
            lewy_score = (record["lewy_contacts"] - 3 * record["lewy_clashes"]) / length
            pff_score = (record["pff_contacts"] - 3 * record["pff_clashes"]) / length
            record["lewy_margin"] = round(chain_target_score - lewy_score, 3)
            record["pff_margin"] = round(full_target_score - pff_score, 3)
            record["selectivity_proxy"] = min(record["lewy_margin"], record["pff_margin"])
            quality_penalty = max(record["gravy"] - 0.8, 0) + max(record["hydrophobic_run"] - 4, 0)
            record["rank_score"] = round(full_target_score
                                         + min(record["selectivity_proxy"], 8.0) - quality_penalty, 3)
            record["geometry_valid"] = (record["energy_final_kj_mol"] < 1e8 and
                                        target_clashes == 0 and target_contacts >= length and
                                        chain_contacts >= length / 4)
            record["broad_msa_contact"] = all(
                record[f"{label}_contacts"] >= length / 2 and
                record[f"{label}_chain_contacts"] >= length / 4 and
                record[f"{label}_clashes"] <= 2
                for label in ("msa", "msa_ii1", "msa_ii2"))
            record["msa_all_compatible"] = full_target_score > 0
            record["comparison_reliable"] = (record["lewy_anchor_rmsd_a"] <= 5 and
                                             record["pff_anchor_rmsd_a"] <= 5)
            record["preference_proxy_positive"] = (record["lewy_margin"] > 0 and
                                                   record["pff_margin"] > 0)
            record["chemistry_pass"] = (record["gravy"] <= 0.8 and
                                        record["hydrophobic_run"] <= 4)
            record["tier"] = ("A" if record["geometry_valid"] and record["broad_msa_contact"]
                              and record["msa_all_compatible"] and
                              record["comparison_reliable"] and
                              record["preference_proxy_positive"] and record["chemistry_pass"]
                              else "B" if record["geometry_valid"] and record["chemistry_pass"]
                              else "C")
            rows.append(record)
    with (RESULTS / "candidate_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["geometry_valid"],
                                                       row["tier"] == "A",
                                                       row["msa_all_compatible"],
                                                       row["comparison_reliable"],
                                                       row["rank_score"]), reverse=True))

    chosen, seen = [], set()
    for site_num in range(1, 5):
        site = f"site_{site_num}"
        ranked = sorted((row for row in rows if row["site"] == site and row["geometry_valid"]
                         and row["chemistry_pass"]),
                        key=lambda row: (row["tier"] == "A", row["msa_all_compatible"],
                                         row["comparison_reliable"], row["rank_score"]), reverse=True)
        for row in ranked:
            if row["sequence"] in seen:
                continue
            chosen.append(row)
            seen.add(row["sequence"])
            if sum(item["site"] == site for item in chosen) == 4:
                break
    ranked = sorted((row for row in rows if row["geometry_valid"] and row["chemistry_pass"]),
                    key=lambda row: (row["tier"] == "A", row["msa_all_compatible"],
                                     row["comparison_reliable"], row["rank_score"]), reverse=True)
    for row in ranked:
        if len(chosen) == 24:
            break
        if row["sequence"] in seen or sum(item["site"] == row["site"] for item in chosen) == 8:
            continue
        chosen.append(row)
        seen.add(row["sequence"])
    chosen.sort(key=lambda row: (row["tier"] == "A", row["msa_all_compatible"],
                                 row["comparison_reliable"], row["rank_score"]), reverse=True)
    with (RESULTS / "shortlist_24.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=chosen[0])
        writer.writeheader()
        writer.writerows(chosen)
    with (RESULTS / "shortlist_24.fasta").open("w") as handle:
        for row in chosen:
            handle.write(f">{row['id']} score={row['rank_score']}\n{row['sequence']}\n")
    priority = [row for row in chosen if row["msa_all_compatible"]]
    with (RESULTS / "priority_candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=priority[0])
        writer.writeheader()
        writer.writerows(priority)
    with (RESULTS / "priority_candidates.fasta").open("w") as handle:
        for row in priority:
            handle.write(f">{row['id']} tier={row['tier']}\n{row['sequence']}\n")
    matrix_columns = ["candidate_id", "sequence", "msa_type_i_signal", "msa_type_ii1_signal",
                      "msa_type_ii2_signal", "pd_dlb_signal", "pff_signal", "monomer_signal",
                      "tau_signal", "abeta_signal", "assay", "batch", "replicate"]
    with (RESULTS / "experimental_binding_matrix_template.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix_columns)
        writer.writeheader()
        for row in chosen:
            writer.writerow({"candidate_id": row["id"], "sequence": row["sequence"]})
    summary = {"generated": len(rows),
               "geometry_valid": sum(row["geometry_valid"] for row in rows),
               "msa_all_compatible": sum(row["geometry_valid"] and row["msa_all_compatible"] for row in rows),
               "comparison_reliable": sum(row["geometry_valid"] and row["comparison_reliable"] for row in rows),
               "tier_a": sum(row["tier"] == "A" for row in rows),
               "shortlisted": len(chosen),
               "priority_candidates": len(priority),
               "shortlist_tier_a": sum(row["tier"] == "A" for row in chosen),
               "sites": {site: sum(row["site"] == site for row in chosen) for site in site_anchors},
               "top": [{"id": row["id"], "sequence": row["sequence"],
                        "rank_score": row["rank_score"]} for row in chosen[:5]]}
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
