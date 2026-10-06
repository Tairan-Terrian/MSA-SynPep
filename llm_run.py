"""Execute the production experiment, preserving each named stage."""

import argparse
import subprocess
import sys
from pathlib import Path

from llm_design import ROOT, read_jsonl, write_jsonl


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], check=True, cwd=ROOT)


def evaluate(folder, phase, candidates, repeat=1, operation="worker"):
    processes = [subprocess.Popen([sys.executable, str(ROOT / "llm_evaluate.py"), operation,
                                   "--run", str(folder), "--candidates", str(candidates),
                                   "--phase", phase, "--gpu", str(gpu), "--repeat", str(repeat),
                                   "--resume-poses",
                                   "--seed", str(20261006 + 1000 * (repeat - 1))], cwd=ROOT)
                 for gpu in (0, 1)]
    codes = [process.wait() for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"GPU production stage {phase} exited with {codes}")
    run("llm_evaluate.py", "collect", "--run", folder, "--candidates", candidates, "--phase", phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--start", choices=("round1_eval", "round2_generate", "round2_eval", "reoptimize", "rank", "audit", "report"),
                        default="round1_eval")
    args = parser.parse_args()
    folder = args.run.resolve()
    if args.start == "reoptimize":
        for phase in ("round1", "round2"):
            evaluate(folder, phase, folder / f"{phase}_candidates.jsonl", operation="relax-existing")
        args.start = "rank"
    stages = ("round1_eval", "round2_generate", "round2_eval", "rank", "audit", "report")
    for stage in stages[stages.index(args.start):]:
        print(f"STAGE {stage}", flush=True)
        if stage == "round1_eval":
            evaluate(folder, "round1", folder / "round1_candidates.jsonl")
        elif stage == "round2_generate":
            run("llm_design.py", "generate", "--run", folder, "--round", "2", "--resume-api")
        elif stage == "round2_eval":
            evaluate(folder, "round2", folder / "round2_candidates.jsonl")
        elif stage == "rank":
            evidence = [c for phase in ("round1", "round2") for c in read_jsonl(folder / f"{phase}_evidence.jsonl")]
            write_jsonl(folder / "all_evidence.jsonl", evidence)
            for view, order in (("contrastive", 1), ("contrastive", 2), ("positive_only", 1)):
                run("llm_design.py", "rank", "--run", folder, "--view", view, "--order", order, "--resume-api")
            run("llm_report.py", "select", "--run", folder)
        elif stage == "audit":
            evaluate(folder, "audit", folder / "finalist_candidates.jsonl", repeat=2)
        else:
            run("llm_report.py", "report", "--run", folder)


if __name__ == "__main__":
    main()
