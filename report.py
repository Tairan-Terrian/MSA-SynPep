"""Current-design shortlist, ablation comparison and evidence-grounded report."""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from design import TARGETS, read_jsonl, write_jsonl


def ranking(folder, view, order=1):
    return json.loads((folder/f"ranking_{view}_order{order}.json").read_text())


def select(args):
    cards=read_jsonl(args.run/"all_evidence.jsonl")
    orders=[ranking(args.run,"full",i)["ordered_ids"] for i in (1,2)]
    positions=[{key:i+1 for i,key in enumerate(order)} for order in orders]
    ordered=sorted(cards,key=lambda c:(positions[0][c["id"]]+positions[1][c["id"]],c["id"]))
    result=[{**c,"consensus_rank":i+1,"mean_llm_position":(positions[0][c["id"]]+positions[1][c["id"]])/2,
             "order1_position":positions[0][c["id"]],"order2_position":positions[1][c["id"]]}
            for i,c in enumerate(ordered)]
    write_jsonl(args.run/"consensus_ranking.jsonl",result)
    write_jsonl(args.run/"finalist_candidates.jsonl",result[:24])
    print("Selected current Top24 from two full-evidence orders",flush=True)


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        columns=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def describe(cards):
    complete=[c for c in cards if c["inverse_summary"]["nll_contrast"] is not None]
    return {"candidates":len(cards),
            "all_msa_have_eligible_pose":sum(all(c["inverse_summary"]["eligible_pose_counts"][k]>0 for k in TARGETS[:3]) for c in cards),
            "all_msa_sampled_poses_eligible":sum(all(p["geometry_eligible"] for k in TARGETS[:3] for p in c["poses"][k]) for c in cards),
            "all_five_have_eligible_pose":len(complete),
            "positive_nll_contrast":sum(c["inverse_summary"]["nll_contrast"]>0 for c in complete),
            "positive_context_gain_contrast":sum(c["inverse_summary"]["context_gain_contrast"]>0 for c in complete),
            "eligible_poses":sum(p["geometry_eligible"] for c in cards for poses in c["poses"].values() for p in poses)}


def report(args):
    folder=args.run
    initial=read_jsonl(folder/"all_evidence.jsonl")
    consensus=read_jsonl(folder/"consensus_ranking.jsonl")
    final_cards={c["id"]:c for c in read_jsonl(folder/"final_evidence.jsonl")}
    final_rank=ranking(folder,"final")
    final=[final_cards[key] for key in final_rank["ordered_ids"]]
    ablation=ranking(folder,"without_inverse")["ordered_ids"]
    by_id={c["id"]:c for c in initial}
    selected_ids={c["id"] for c in final}
    orders=[ranking(folder,"full",i)["ordered_ids"] for i in (1,2)]
    groups={"全部120条，初次姿势":describe(initial),
            "去逆折叠信息 Top24，初次姿势":describe([by_id[key] for key in ablation[:24]]),
            "完整证据 Top24，初次姿势":describe([by_id[c["id"]] for c in final]),
            "完整证据 Top24，两次姿势":describe(final)}
    usage=read_jsonl(folder/"api/usage.jsonl")
    pose_rows=read_jsonl(folder/"initial_inverse.jsonl")+read_jsonl(folder/"replicate_inverse.jsonl")
    stats={"groups":groups,"unique_candidates":len({c["sequence"] for c in initial}),
           "pose_count":len(pose_rows),"decoding_orders_per_context":8,
           "model_evaluations":len(pose_rows)*8*2,
           "inverse_ablation_top24_overlap":len(selected_ids & set(ablation[:24])),
           "input_order_top24_overlap":len(set(orders[0][:24]) & set(orders[1][:24])),
           "mean_absolute_order_change":round(statistics.mean(abs(c["order1_position"]-c["order2_position"]) for c in consensus),2),
           "api_calls":len(usage),
           "tokens":{k:sum(u["usage"][k] for u in usage) for k in ("prompt_tokens","completion_tokens","total_tokens")},
           "returned_models":sorted({u["returned_model"] for u in usage}),
           "finalist_sites":dict(Counter(c["site"] for c in final)),
           "geometry_ineligible_poses":sum(not p["geometry_eligible"] for p in pose_rows)}
    stats["new_second_poses"]=sum(Path(p["pdb"]).parts[-3]=="poses" and "replicate" in Path(p["pdb"]).parts
                                  for p in read_jsonl(folder/"replicate_inverse.jsonl"))
    (folder/"experiment_summary.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    shortlist=[]
    for i,c in enumerate(final,1):
        explanation=final_rank["evaluations"][c["id"]]
        shortlist.append({"rank":i,"id":c["id"],"sequence":c["sequence"],"site":c["site"],
                          "priority":explanation["priority"],**c["inverse_summary"],
                          "support":explanation["support"],"limitation":explanation["limitation"]})
    write_csv(folder/"shortlist_24.csv",shortlist)
    (folder/"shortlist_24.fasta").write_text("".join(f">{c['id']} rank={c['rank']}\n{c['sequence']}\n" for c in shortlist))
    write_csv(folder/"candidate_scores.csv",[{"id":c["id"],"sequence":c["sequence"],"site":c["site"],
               "consensus_rank":c["consensus_rank"],"mean_llm_position":c["mean_llm_position"],
               **c["inverse_summary"]} for c in consensus])
    write_csv(folder/"pose_scores.csv",pose_rows)
    residue_rows=[{"id":p["id"],"target":p["target"],"repeat":p["repeat"],"position":i+1,
                   "aa":by_id[p["id"]]["sequence"][i],"geometry_eligible":p["geometry_eligible"],
                   "complex_nll":nll,"context_gain":p["residue_context_gain"][i]}
                  for p in pose_rows for i,nll in enumerate(p["complex_residue_nll"])]
    write_csv(folder/"residue_scores.csv",residue_rows)
    write_csv(folder/"experimental_binding_matrix_template.csv",[
        {"candidate_id":c["id"],"sequence":c["sequence"],**{f"{t}_binding_signal":"" for t in TARGETS},
         "assay":"","batch":"","replicate":""} for c in final])
    table="| 实验组 | 候选数 | 三种MSA均有合格姿势 | 五构象均有合格姿势 | NLL对照差为正 | 环境增益对照差为正 |\n|---|---:|---:|---:|---:|---:|\n"
    table+="".join(f"| {name} | {g['candidates']} | {g['all_msa_have_eligible_pose']} | {g['all_five_have_eligible_pose']} | {g['positive_nll_contrast']} | {g['positive_context_gain_contrast']} |\n" for name,g in groups.items())
    candidate_table="| 排名 | ID | 序列 | 位点 | LLM优先级 | NLL对照差 | 环境增益对照差 |\n|---:|---|---|---|---:|---:|---:|\n"
    def display(value):
        return str(value) if value is not None else "证据不足"
    candidate_table+="".join(f"| {c['rank']} | {c['id']} | {c['sequence']} | {c['site']} | {c['priority']} | {display(c['nll_contrast'])} | {display(c['context_gain_contrast'])} |\n" for c in shortlist)
    meta=json.loads((folder/"RUN.json").read_text())
    mode_description=("本次以已有120条闭源LLM序列及600个初次姿势为固定输入，重新计算GPU几何和ProteinMPNN证据，并重新调用LLM排序。已有第二次姿势仅在候选进入本次Top24后复用；其余候选新增第二次采样。序列提出时的反馈来自先前几何计算。当前generate模式已接入逆折叠反馈；本次输入序列保持原样，因此本次结果评估的是新增验证与排序环节。"
                      if meta["mode"]=="reuse" else "本次两轮序列均按当前generate流程提出；第二轮读取第一轮几何及ProteinMPNN逐位置证据。")
    text=f"""# MSA-SynPep：正向结构生成与逆折叠评价实验报告

日期：2026-10-09。服务器工程：`/media/yjz/htr/synuclein`。当前结果：`results/current/`。

## 1. 研究目的与当前设计

MSA指多系统萎缩。研究目标是提出偏好MSA相关α-syn纤维构象的短肽，用于后续构象识别实验。构象面板包括MSA结构6XYO、6XYP、6XYQ，以及同蛋白对照8A9L和6CU7。positive/negative表达希望识别和需要区分的目标结构；短肽的真实结合标签由实验产生。

当前流程：闭源LLM提出序列 → PepGLAD固定序列生成复合物姿势 → GPU局部最小化及几何评价 → ProteinMPNN逆折叠评价原序列 → 闭源LLM跨构象排序 → Top24第二姿势 → 两姿势精排。

{mode_description}

## 2. 逆折叠评价怎样实现

使用[ProteinMPNN原论文，Science 2022](https://doi.org/10.1126/science.add2187)及[官方实现](https://github.com/dauparas/ProteinMPNN)的v_48_020权重。参数保持固定，输入骨架坐标扰动设为0。完整模型输入包括三层受体链及短肽链K。受体身份先进入解码条件，短肽链为唯一评价对象，原序列在每个位置的负对数概率按肽长平均。

每个姿势运行8种肽位置解码顺序，保存均值、标准差和逐位置分数。原序列适配程度以complex_nll表达，数值越低，模型越认为该序列适合这个结构。再移除受体，保留完全相同的肽骨架和肽解码顺序，得到peptide_only_nll。context_gain=peptide_only_nll−complex_nll，正值表示靶标环境增加了模型对原序列的支持。分数及其差值是模型推理量，真实亲和力与选择性由结合实验确认。

这项匹配比较部分控制骨架及序列组成的偏好；受体移除也会改变模型邻接图，环境增益仍有模型偏差。ProteinMPNN主要基于PDB蛋白结构训练，10–25 aa短肽与淀粉样纤维环境的迁移表现需要验证。ProteinMPNN与PepGLAD使用不同的模型与任务，训练来源可能重叠。模型一致性提供额外依据，准确性增益需要带实验标签的比较。

官方权重通过临时PyTorch2.4环境以weights_only=True读取，将原模型张量保存为GPU运行环境可读取的格式；未使用的浮点训练元数据移至SOURCE.json。当前GPU环境仍使用PyTorch1.13，ProteinMPNN权重加载使用weights_only=True。

## 3. 结构质量与跨构象规则

几何资格规则为：中央链和组装体2Å重原子冲突均为0，短肽相邻C–N距离相对近似理想1.33Å的最大偏离≤0.3Å。该规则延续现有宽松骨架完整性检查，完整立体化学仍需进一步分析。原始逆折叠分数全部保留；几何资格不足的姿势在LLM证据卡中使用null表示逆折叠证据未知。受体位移、接触、极性/电荷接近及解码顺序变化均随卡片提供。

NLL对照差=两种对照所有合格姿势中的最低NLL−三种MSA各自合格姿势平均NLL中的最高值。环境增益对照差=三种MSA各自平均增益中的最低值−两种对照所有合格姿势中的最高增益。五构象都存在合格姿势时才计算这两个描述量。正值代表该计算量偏向MSA，阈值与增益尚待实验校准。

LLM读取各姿势连续原始值，给出0–100的相对实验优先级，程序按优先级排序。两次完整排序反转输入顺序，按平均名次选择Top24。相同候选和几何证据再做一次去逆折叠信息排序，观察决策变化。Top24各构象补足第二姿势后，LLM读取两个观察完成内部精排。异常对照姿势留下未知证据，后续应继续采样与测量。

## 4. 运行结果

唯一候选数：**{stats['unique_candidates']}**。已评价复合物姿势：**{stats['pose_count']}**。每个姿势包含完整复合物与肽单独两种条件，每种8个解码顺序，总计**{stats['model_evaluations']}**次模型条件评价。几何资格不足姿势：**{stats['geometry_ineligible_poses']}**。

Top24第二姿势中，本次新采样**{stats['new_second_poses']}**个，复用固定输入对应坐标**{120-stats['new_second_poses']}**个；所有姿势均由当前代码重新评价。最终Top24每种MSA构象都至少存在一个合格姿势，其中**{groups['完整证据 Top24，两次姿势']['all_msa_sampled_poses_eligible']}/24**条在三种MSA的两次采样中全部合格。至少一次合格与两次稳定合格代表不同的证据强度。

{table}

完整证据和去逆折叠信息Top24重叠：**{stats['inverse_ablation_top24_overlap']}/24**。两次完整排序Top24重叠：**{stats['input_order_top24_overlap']}/24**，全池平均名次变化**{stats['mean_absolute_order_change']}**。这些量描述当前计算流程的决策与稳定性。去信息对照只有一次LLM调用，差异中包含LLM随机变化；该实验支持方法行为分析，真实命中率由湿实验评估。

## 5. 当前24条实验优先候选

LLM给出的当前判断：{final_rank['decision_logic']}

{candidate_table}

详细支持与限制见shortlist_24.csv；所有姿势均保留原始分数及资格标记。证据不足表示相应对照差无法可靠汇总。

## 6. GPU、费用与复现

PepGLAD、OpenMM及几何计算沿用CUDA/NVIDIA OpenCL，两块RTX3090并行。ProteinMPNN在CUDA上分担姿势，各姿势8种解码顺序合并成批。文件整理和PDB解析在CPU进行。当前服务器其他任务占用显存，结构采样批量设为2，各GPU安排1个最小化进程。

本次新增API调用**{stats['api_calls']}**次，输入**{stats['tokens']['prompt_tokens']:,}** tokens，输出**{stats['tokens']['completion_tokens']:,}** tokens，总计**{stats['tokens']['total_tokens']:,}** tokens。模型返回版本：`{', '.join(stats['returned_models'])}`。序列来源的历史请求作为data/inputs/generation_provenance保存，当前用量统计覆盖本次调用。实际货币费用以供应商账单为准。

```bash
cd /media/yjz/htr/synuclein
export SYNPep_API_KEY='<运行时密钥>'
.venv/bin/python pipeline.py --run results/current --mode reuse
```

从新序列开始的当前设计：`.venv/bin/python pipeline.py --run results/new_experiment --mode generate`。第二轮提案读取几何及逆折叠逐位置反馈。生成模式的第二姿势来自本次序列，复用模式仅复用与固定输入对应的姿势。已完成API命名阶段在同一实验目录继续运行时复用；输入或模型改变时使用新的实验目录。

## 7. 输出与贡献验证

all_evidence.jsonl保存全部候选的初筛卡片；initial_inverse.jsonl及replicate_inverse.jsonl保存逐姿势/逐位置证据；pose_scores.csv与residue_scores.csv便于分析；consensus_ranking.jsonl、shortlist_24.csv及FASTA保存当前排名。experimental_binding_matrix_template.csv提供五构象的实验信号录入字段。

可检验的贡献是：利用正向姿势与逆向序列适配的一致性，结合靶标环境对原序列的支持及同蛋白构象对照，改善MSA选择性候选的选择。后续应让完整证据与去逆折叠信息推荐的候选在同一条件下测量三种MSA和两种对照，比较相同预算下的选择性命中率和Top-K富集。当前报告给出计算证据、排序变化和实验候选，真实结合数据是确认增益的依据。
"""
    (folder/"REPORT.md").write_text(text)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("select","report"))
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    {"select":select,"report":report}[args.command](args)


if __name__=="__main__":
    main()
