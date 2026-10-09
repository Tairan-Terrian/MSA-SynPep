"""Compose raw multi-pose evidence and descriptive inverse-folding summaries."""

import argparse
import json
import statistics
from pathlib import Path

from design import ROOT, TARGETS, read_jsonl, write_jsonl


def cards(candidates, poses):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    grouped = {}
    for pose in poses:
        grouped.setdefault((pose["id"], pose["target"]), []).append(pose)
    result = []
    for candidate in candidates:
        by_target = {key: sorted(grouped[(candidate["id"], key)], key=lambda p:p["repeat"])
                     for key in TARGETS}
        compatible = {key: [p for p in rows if p["geometry_eligible"]] for key, rows in by_target.items()}
        all_contexts = all(compatible.values())
        nll_gap = (min(p["complex_nll"] for key in TARGETS[3:] for p in compatible[key])
                   - max(statistics.mean(p["complex_nll"] for p in compatible[key]) for key in TARGETS[:3])
                   if all_contexts else None)
        gain_gap = (min(statistics.mean(p["context_gain"] for p in compatible[key]) for key in TARGETS[:3])
                    - max(p["context_gain"] for key in TARGETS[3:] for p in compatible[key])
                    if all_contexts else None)
        residue_feedback = {key: [{"repeat":p["repeat"], "sequence_position_nll":p["complex_residue_nll"],
                                  "sequence_position_context_gain":p["residue_context_gain"]}
                                 for p in compatible[key]] for key in TARGETS}
        run = longest = 0
        for aa in candidate["sequence"]:
            run = run + 1 if aa in "IVLFWYM" else 0
            longest = max(longest, run)
        result.append({**candidate, "length":len(candidate["sequence"]),
                       "gravy":round(ProteinAnalysis(candidate["sequence"]).gravy(),3),
                       "hydrophobic_run":longest, "poses":by_target,
                       "residue_feedback":residue_feedback,
                       "inverse_summary":{"eligible_pose_counts":{k:len(v) for k,v in compatible.items()},
                                          "nll_contrast":round(nll_gap,5) if nll_gap is not None else None,
                                          "context_gain_contrast":round(gain_gap,5) if gain_gap is not None else None}})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    poses = [p for path in sorted(args.run.glob(f"{args.phase}_inverse_gpu*.jsonl")) for p in read_jsonl(path)]
    write_jsonl(args.run / f"{args.phase}_inverse.jsonl", poses)
    write_jsonl(args.run / f"{args.phase}_evidence.jsonl", cards(read_jsonl(args.candidates), poses))
    print(f"Combined {len(poses)} poses into current evidence", flush=True)


if __name__ == "__main__":
    main()
