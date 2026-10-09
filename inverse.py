"""GPU ProteinMPNN likelihood of the supplied peptide sequence in each pose."""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from design import ROOT, read_jsonl, write_jsonl

VENDOR = ROOT / "vendor/ProteinMPNN"
sys.path.insert(0, str(VENDOR))
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize


def features(path, orders, device, chains):
    structure = parse_PDB(str(path), input_chain_list=chains)[0]
    fixed = [c for c in chains if c != "K"]
    tensors = tied_featurize([copy.deepcopy(structure) for _ in range(orders)], device,
                            {structure["name"]: (["K"], fixed)})
    x, s, mask, _, chain_mask, chain_encoding = tensors[:6]
    residue_idx = tensors[12]
    return x, s, mask, chain_mask, residue_idx, chain_encoding


def score(model, inputs, peptide_order):
    x, s, mask, chain_mask, residue_idx, chains = inputs
    length = peptide_order.shape[1]
    # Fixed receptor identities precede every peptide position in the decoder.
    fixed_order = torch.arange(length, s.shape[1], device=s.device)[None].expand(s.shape[0], -1)
    order = torch.cat((fixed_order, peptide_order), dim=1)
    log_probs = model(x, s, mask, chain_mask, residue_idx, chains,
                      torch.zeros_like(chain_mask), use_input_decoding_order=True,
                      decoding_order=order)
    original_nll = -torch.gather(log_probs, 2, s[:, :, None]).squeeze(-1)
    # ProteinMPNN places the sole designed chain K first; target chains remain fixed.
    per_residue = original_nll[:, :length]
    return per_residue.mean(1), per_residue.mean(0)


def worker(args):
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    checkpoint = torch.load(VENDOR / "weights.pt", map_location=device, weights_only=True)
    model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3,
                        augment_eps=0, k_neighbors=checkpoint["num_edges"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    candidates = {c["id"]: c for c in read_jsonl(args.candidates)}
    rows = read_jsonl(args.run / f"{args.phase}_geometry.jsonl")[args.gpu::2]
    assembly = json.loads((args.run / "assembly_context.json").read_text())
    started = time.monotonic()
    results = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            candidate = candidates[row["id"]]
            length = len(candidate["sequence"])
            # Match peptide decoding permutations in the complex and peptide-only contexts.
            generator = torch.Generator(device=device).manual_seed(args.seed + index * 2 + args.gpu)
            order = torch.argsort(torch.rand(args.orders, length, device=device, generator=generator), dim=1)
            chains = ["K"] + assembly[row["target"]]["model_chains"]
            complex_inputs = features(ROOT / row["pdb"], args.orders, device, chains)
            solo_inputs = features(ROOT / row["pdb"], args.orders, device, ["K"])
            alphabet = "ACDEFGHIKLMNPQRSTVWYX"
            original = torch.tensor([alphabet.index(aa) for aa in candidate["sequence"]], device=device)
            complex_inputs[1][:, :length] = original
            solo_inputs[1][:, :length] = original
            full, residue_full = score(model, complex_inputs, order)
            solo, residue_solo = score(model, solo_inputs, order)
            gain = solo - full
            eligible = (row["peptide_cn_max_deviation_a"] <= .3
                        and row["clashes_per_residue"] == 0
                        and row["assembly_clashes_per_residue"] == 0)
            results.append({**row, "geometry_eligible": eligible, "inverse_model": "ProteinMPNN v_48_020",
                            "complex_nll": round(full.mean().item(), 5),
                            "complex_nll_sd": round(full.std(unbiased=False).item(), 5),
                            "peptide_only_nll": round(solo.mean().item(), 5),
                            "context_gain": round(gain.mean().item(), 5),
                            "context_gain_sd": round(gain.std(unbiased=False).item(), 5),
                            "complex_residue_nll": [round(v, 5) for v in residue_full.tolist()],
                            "residue_context_gain": [round(v, 5) for v in (residue_solo-residue_full).tolist()],
                            "decoding_orders": args.orders})
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                write_jsonl(args.run / f"{args.phase}_inverse_gpu{args.gpu}.jsonl", results)
                print(f"ProteinMPNN GPU{args.gpu} {index+1}/{len(rows)} poses "
                      f"elapsed={time.monotonic()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--orders", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    worker(parser.parse_args())


if __name__ == "__main__":
    main()
