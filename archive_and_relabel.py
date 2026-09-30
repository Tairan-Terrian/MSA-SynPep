"""Archive the initial Type-I-only epitope run and reuse its valid site calculations."""

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PILOT = RESULTS / "pilot_typeI_only"
PILOT.mkdir(exist_ok=True)

for name in ("selected_epitopes.csv", "residue_surface_scores.csv", "epitope_selection.json",
             "candidate_scores_pre_relax.csv", "shortlist_24_pre_relax.csv",
             "shortlist_24_pre_relax.fasta", "candidate_scores.csv", "shortlist_24.csv",
             "shortlist_24.fasta", "run_summary.json", "experimental_binding_matrix_template.csv",
             "minimization_gpu0.jsonl", "minimization_gpu1.jsonl",
             "site_1_pocket.json", "site_2_pocket.json", "site_3_pocket.json",
             "site_4_pocket.json"):
    path = RESULTS / name
    if path.exists():
        shutil.copy2(path, PILOT / name)

shutil.copytree(RESULTS / "generated", PILOT / "generated")
shutil.copytree(RESULTS / "relaxed", PILOT / "relaxed")
shutil.move(RESULTS / "generated" / "site_3", PILOT / "typeI_only_generated_site_3")
shutil.move(RESULTS / "relaxed" / "site_3", PILOT / "typeI_only_relaxed_site_3")

for folder in ("generated", "relaxed"):
    old = RESULTS / folder / "site_4"
    new = RESULTS / folder / "site_3"
    old.rename(new)
    for path in new.glob("site_4_*.pdb"):
        path.rename(new / path.name.replace("site_4_", "site_3_"))

candidate_file = RESULTS / "generated" / "site_3" / "candidates.jsonl"
changed = []
for line in candidate_file.read_text().splitlines():
    row = json.loads(line)
    row["id"] = row["id"].replace("site_4_", "site_3_")
    row["site"] = "site_3"
    row["pdb"] = str(RESULTS / "generated" / "site_3" / f"{row['id']}.pdb")
    changed.append(json.dumps(row))
candidate_file.write_text("\n".join(changed) + "\n")

for gpu in (0, 1):
    log = RESULTS / f"minimization_gpu{gpu}.jsonl"
    changed = []
    for line in log.read_text().splitlines():
        row = json.loads(line)
        if row["site"] == "site_3":
            continue
        if row["site"] == "site_4":
            row["id"] = row["id"].replace("site_4_", "site_3_")
            row["site"] = "site_3"
            row["pdb"] = str(RESULTS / "relaxed" / "site_3" / f"{row['id']}.pdb")
        changed.append(json.dumps(row))
    log.write_text("\n".join(changed) + "\n")

(PILOT / "README.txt").write_text(
    "The first run selected site_3 at residue 74 using MSA Type I accessibility only. "
    "That surface is poorly exposed in MSA Type II. The final run replaces it with a "
    "surface that is exposed across all three MSA structures. The complete original "
    "generated and relaxed structures, scores, and logs are archived here.\n")
print("Archived initial run; reused sites 1, 2, and former site 4 as final sites 1, 2, and 3.")
