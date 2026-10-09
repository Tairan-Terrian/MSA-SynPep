"""Current sequence proposal, complex sampling, inverse assessment and LLM ranking."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from design import ROOT, read_jsonl, write_jsonl


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], cwd=ROOT, check=True)


def gpu_pair(script, folder, phase, candidates, *extra):
    processes = [subprocess.Popen([sys.executable, str(ROOT / script), *map(str, extra),
                                   "--run",str(folder), "--phase",phase,
                                   "--candidates",str(candidates), "--gpu",str(gpu)], cwd=ROOT)
                 for gpu in (0,1)]
    for process in processes:
        process.wait()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args)


def inverse(folder, phase, candidates):
    gpu_pair("inverse.py", folder, phase, candidates)
    run("evidence.py", "--run",folder,"--phase",phase,"--candidates",candidates)


def sample(folder, phase, candidates, repeat):
    gpu_pair("structure.py", folder, phase, candidates, "worker", "--repeat",repeat,
             "--seed",20261009+1000*(repeat-1), "--resume-poses")
    run("structure.py","collect","--run",folder,"--phase",phase,"--candidates",candidates)


def reuse_inputs(folder):
    inputs = ROOT / "data/inputs"
    folder.mkdir(parents=True,exist_ok=True)
    for name in ("epitope_context.json","assembly_context.json","round1_candidates.jsonl","round2_candidates.jsonl"):
        shutil.copy2(inputs/name,folder/name)
    shutil.copytree(inputs/"targets",folder/"targets",dirs_exist_ok=True)
    candidates = read_jsonl(inputs/"round1_candidates.jsonl")+read_jsonl(inputs/"round2_candidates.jsonl")
    write_jsonl(folder/"candidates.jsonl",candidates)
    source_poses = [p for p in read_jsonl(inputs/"poses.jsonl") if p["repeat"]==1]
    for gpu in (0,1):
        selected={c["id"] for c in candidates[gpu::2]}
        write_jsonl(folder/f"initial_gpu{gpu}_poses.jsonl",[p for p in source_poses if p["id"] in selected])
    metadata = {"design":"LLM proposals + PepGLAD structures + ProteinMPNN original-sequence assessment + LLM ranking",
                "mode":"reuse", "input_provenance":json.loads((inputs/"PROVENANCE.json").read_text()),
                "ranking_inputs":"current geometry and inverse-folding evidence; current API responses",
                "evaluation_date":"2026-10-09"}
    (folder/"RUN.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))


def generate(folder):
    run("design.py","prepare","--run",folder)
    for round_number in (1,2):
        run("design.py","generate","--run",folder,"--round",round_number,"--resume-api")
        phase=f"round{round_number}"
        candidates=folder/f"{phase}_candidates.jsonl"
        sample(folder,phase,candidates,1)
        inverse(folder,phase,candidates)
    candidates=[c for number in (1,2) for c in read_jsonl(folder/f"round{number}_candidates.jsonl")]
    evidence=[c for number in (1,2) for c in read_jsonl(folder/f"round{number}_evidence.jsonl")]
    poses=[p for number in (1,2) for p in read_jsonl(folder/f"round{number}_inverse.jsonl")]
    write_jsonl(folder/"candidates.jsonl",candidates)
    write_jsonl(folder/"all_evidence.jsonl",evidence)
    write_jsonl(folder/"initial_inverse.jsonl",poses)
    (folder/"RUN.json").write_text(json.dumps({"mode":"generate","evaluation_date":"2026-10-09",
                                               "generation_feedback":"geometry and ProteinMPNN residue-level evidence"},indent=2))


def rank(folder):
    for view,order in (("full",1),("full",2),("without_inverse",1)):
        run("design.py","rank","--run",folder,"--view",view,"--order",order,"--resume-api")
    run("report.py","select","--run",folder)


def replicate(folder):
    candidates=read_jsonl(folder/"finalist_candidates.jsonl")
    mode=json.loads((folder/"RUN.json").read_text())["mode"]
    known=read_jsonl(ROOT/"data/inputs/poses.jsonl") if mode=="reuse" else []
    completed=[p for path in sorted(folder.glob("replicate_gpu*_poses.jsonl")) for p in read_jsonl(path)]
    reusable={(p["id"],p["target"],p["repeat"]):p for p in known+completed if p["repeat"]==2}
    for gpu in (0,1):
        ids={c["id"] for c in candidates[gpu::2]}
        write_jsonl(folder/f"replicate_gpu{gpu}_poses.jsonl",
                    [p for p in reusable.values() if p["id"] in ids])
    path=folder/"finalist_candidates.jsonl"
    sample(folder,"replicate",path,2)
    inverse(folder,"replicate",path)
    from evidence import cards
    first=[p for p in read_jsonl(folder/"initial_inverse.jsonl") if p["id"] in {c["id"] for c in candidates}]
    second=read_jsonl(folder/"replicate_inverse.jsonl")
    write_jsonl(folder/"final_evidence.jsonl",cards(candidates,first+second))
    run("design.py","rank","--run",folder,"--view","final","--evidence","final_evidence.jsonl","--resume-api")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=ROOT/"results/current")
    parser.add_argument("--mode",choices=("reuse","generate"),default="reuse")
    parser.add_argument("--start",choices=("evaluate","inverse","rank","replicate","report"),default="evaluate")
    args=parser.parse_args()
    folder=args.run.resolve()
    stages=("evaluate","inverse","rank","replicate","report")
    for stage in stages[stages.index(args.start):]:
        print(f"CURRENT STAGE {stage}",flush=True)
        if stage=="evaluate":
            if args.mode=="generate":
                generate(folder)
                continue
            reuse_inputs(folder)
            run("structure.py","collect","--run",folder,"--phase","initial","--candidates",folder/"candidates.jsonl")
        elif stage=="inverse":
            if args.mode=="reuse":
                inverse(folder,"initial",folder/"candidates.jsonl")
                shutil.copy2(folder/"initial_evidence.jsonl",folder/"all_evidence.jsonl")
        elif stage=="rank":
            rank(folder)
        elif stage=="replicate":
            replicate(folder)
        else:
            run("report.py","report","--run",folder)


if __name__=="__main__":
    main()
