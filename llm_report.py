"""Analyze generation, ranking sensitivity and finalist pose replication."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

from llm_design import ROOT, TARGETS, METRIC_COLUMNS, read_jsonl, write_jsonl


def ranking(folder, view, order):
    return json.loads((folder / f"ranking_{view}_order{order}.json").read_text())


def select(args):
    cards = read_jsonl(args.run / "all_evidence.jsonl")
    ranks = [ranking(args.run, "contrastive", i)["ordered_ids"] for i in (1, 2)]
    positions = [{key: i + 1 for i, key in enumerate(order)} for order in ranks]
    consensus = sorted(cards, key=lambda c: (positions[0][c["id"]] + positions[1][c["id"]], c["id"]))
    result = [{**c, "consensus_rank": i + 1,
               "mean_llm_position": (positions[0][c["id"]] + positions[1][c["id"]]) / 2,
               "order1_position": positions[0][c["id"]], "order2_position": positions[1][c["id"]]}
              for i, c in enumerate(consensus)]
    write_jsonl(args.run / "consensus_ranking.jsonl", result)
    write_jsonl(args.run / "finalist_candidates.jsonl", result[:24])
    print("Selected 24 finalists using mean LLM rank across two input orders", flush=True)


def summary(cards):
    def positive_compatible(c):
        return all(c["conformers"][k]["clashes_per_residue"] == 0
                   and c["conformers"][k]["assembly_clashes_per_residue"] == 0
                   and c["conformers"][k]["peptide_cn_max_deviation_a"] <= .3
                   and c["conformers"][k]["contacts_per_residue"] >= .5 for k in TARGETS[:3])
    return {"count": len(cards), "unique_sequences": len({c["sequence"] for c in cards}),
            "msa_geometry_compatible": sum(positive_compatible(c) for c in cards),
            "positive_geometric_contrast": sum(c["contrast"] > 0 for c in cards),
            "both_geometry_and_contrast": sum(positive_compatible(c) and c["contrast"] > 0 for c in cards),
            "peptide_geometry_suspect_poses": sum(v["peptide_cn_max_deviation_a"] > .3
                                                  for c in cards for v in c["conformers"].values()),
            "median_contrast": round(statistics.median(c["contrast"] for c in cards), 3),
            "median_weakest_msa_contacts": round(statistics.median(
                min(c["conformers"][k]["contacts_per_residue"] for k in TARGETS[:3]) for c in cards), 3)}


def replicates(args):
    first = read_jsonl(args.run / "finalist_candidates.jsonl")
    second = {c["id"]: c for c in read_jsonl(args.run / "audit_evidence.jsonl")}
    cards = []
    for c in first:
        repeats = {key: [[pose["conformers"][key][name] for name in METRIC_COLUMNS]
                         for pose in (c, second[c["id"]])] for key in TARGETS}
        cards.append({**c, "replicate_poses": repeats})
    write_jsonl(args.run / "replicated_evidence.jsonl", cards)


def report(args):
    cards = read_jsonl(args.run / "all_evidence.jsonl")
    consensus = read_jsonl(args.run / "consensus_ranking.jsonl")
    selected = {c["id"]: c for c in consensus[:24]}
    final_answer = ranking(args.run, "replicated", 1)
    finalists = [selected[key] for key in final_answer["ordered_ids"]]
    repeat = {c["id"]: c for c in read_jsonl(args.run / "audit_evidence.jsonl")}
    pos = ranking(args.run, "positive_only", 1)["ordered_ids"]
    original_orders = [ranking(args.run, "contrastive", i) for i in (1, 2)]
    first, second = [x["ordered_ids"] for x in original_orders]
    by_id = {c["id"]: c for c in cards}
    groups = {"LLM 第一轮 60 条": summary([c for c in cards if c["round"] == 1]),
              "LLM 第二轮 60 条": summary([c for c in cards if c["round"] == 2]),
              "全部 LLM 120 条": summary(cards),
              "MSA 信息 LLM Top24": summary([by_id[key] for key in pos[:24]]),
              "跨构象 LLM 共识 Top24": summary(finalists),
              "Top24 第二次结构采样": summary([repeat[c["id"]] for c in finalists])}
    usage = read_jsonl(args.run / "api/usage.jsonl")
    tokens = {field: sum(call["usage"][field] for call in usage)
              for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
    sensitivity = {"order_top24_overlap": len(set(first[:24]) & set(second[:24])),
                   "mean_absolute_position_change": round(statistics.mean(abs(c["order1_position"] - c["order2_position"]) for c in consensus), 2),
                   "positive_only_top24_overlap": len(set(c["id"] for c in finalists) & set(pos[:24]))}
    stats = {"groups": groups, "ranking_sensitivity": sensitivity, "api_calls": len(usage),
             "api_calls_used": sum(c["status"] == "used" for c in usage),
             "api_calls_superseded": sum(c["status"] == "superseded" for c in usage),
             "tokens": tokens, "returned_models": sorted({c["returned_model"] for c in usage}),
             "finalist_rounds": dict(Counter(c["round"] for c in finalists)),
             "finalist_sites": dict(Counter(c["site"] for c in finalists)),
             "pose_count": len(cards) * 5 + len(finalists) * 5}
    (args.run / "experiment_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    reasons = {item["id"]: item for item in final_answer["shortlist"]}
    rows = []
    for rank_number, c in enumerate(finalists, 1):
        r = repeat[c["id"]]
        explanation = reasons[c["id"]]
        rows.append({"rank": rank_number, "consensus_rank_initial": c["consensus_rank"], "id": c["id"], "sequence": c["sequence"],
                     "site": c["site"], "round": c["round"],
                     "llm_priority_final": explanation["priority"],
                     "mean_llm_position": c["mean_llm_position"], "contrast_rep1": c["contrast"],
                     "contrast_rep2": r["contrast"], "support": explanation["support"],
                     "limitation": explanation["limitation"]})
    with (args.run / "shortlist_24.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.run / "shortlist_24.fasta").write_text("".join(f">{r['id']} rank={r['rank']}\n{r['sequence']}\n" for r in rows))
    all_rows = []
    for c in consensus:
        row = {name: c[name] for name in ("id", "site", "sequence", "source", "round", "length",
                                         "gravy", "hydrophobic_run", "contrast", "consensus_rank",
                                         "order1_position", "order2_position", "mean_llm_position")}
        row["llm_priority_order1"] = original_orders[0]["evaluations"][c["id"]]["priority"]
        row["llm_priority_order2"] = original_orders[1]["evaluations"][c["id"]]["priority"]
        for key, values in c["conformers"].items():
            row.update({f"{key}_{name}": value for name, value in values.items()
                        if name != "repeat_contacts_range"})
        all_rows.append(row)
    with (args.run / "candidate_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0])
        writer.writeheader()
        writer.writerows(all_rows)
    columns = ["candidate_id", "sequence", "msa_type_i_signal", "msa_type_ii1_signal", "msa_type_ii2_signal",
               "lewy_signal", "pff_signal", "monomer_signal", "tau_signal", "abeta_signal", "assay", "batch", "replicate"]
    with (args.run / "experimental_binding_matrix_template.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({"candidate_id": c["id"], "sequence": c["sequence"]} for c in finalists)
    table = "| 实验组 | 候选数量 | 三种MSA几何兼容 | 对照差值为正 | 两条件同时满足 | 差值中位数 | 肽键失真姿势数 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    table += "".join(f"| {name} | {g['count']} | {g['msa_geometry_compatible']} | {g['positive_geometric_contrast']} | {g['both_geometry_and_contrast']} | {g['median_contrast']} | {g['peptide_geometry_suspect_poses']} |\n" for name, g in groups.items())
    candidate_table = "| 排名 | ID | 序列 | 位点 | 生成轮次 | LLM最终优先级 | 差值：采样1/采样2 |\n|---:|---|---|---|---:|---:|---:|\n"
    candidate_table += "".join(f"| {r['rank']} | {r['id']} | {r['sequence']} | {r['site']} | {r['round']} | {r['llm_priority_final']} | {r['contrast_rep1']}/{r['contrast_rep2']} |\n" for r in rows)
    text = f"""# MSA-SynPep 闭源 LLM 生成与排序实验报告

日期：2026-10-06。代码和结果服务器根目录：`/media/yjz/htr/synuclein`。本次结果独立保存于 `results/llm_20261006/`。

## 1. 这次到底做了什么

本项目中的MSA指多系统萎缩。短肽是由少量氨基酸组成的分子，候选binder希望识别MSA相关α-synuclein纤维的形状，可作为后续构象检测和研究工具的起点。

目标是找出对三种 MSA α-synuclein 纤维形状有兼容证据、对 Lewy/PFF 对照形状兼容证据较弱的短肽。LLM 提出序列、读取计算证据进行排序；PepGLAD 和 OpenMM 负责生成及调整三维姿势。本次候选池由两轮LLM生成的120条新序列组成，完成五构象评估、120条排序和24条优先候选的第二次结构采样。

本次使用用户指定的闭源模型 `{usage[0]['requested_model']}`，网关返回模型版本为 `{', '.join(stats['returned_models'])}`。序列提案和候选优先级都由LLM产生。四个新增脚本形成独立流程，结构工具提供判断所需的测量证据。

本轮保持模型权重固定，使用闭源模型的API推理和公开预训练的蛋白结构采样器。四个任务表位的中心残基39、80、66、50分别对应site_1至site_4。

| 结构面板 | 来源 | 本任务中的角色 |
|---|---|---|
| MSA Type I | [6XYO，RCSB PDB](https://www.rcsb.org/structure/6XYO) | 希望识别的构象 |
| MSA Type II-1 | [6XYP，RCSB PDB](https://www.rcsb.org/structure/6XYP) | 希望识别的构象 |
| MSA Type II-2 | [6XYQ，RCSB PDB](https://www.rcsb.org/structure/6XYQ) | 希望识别的构象 |
| Lewy fold | [8A9L，RCSB PDB](https://www.rcsb.org/structure/8A9L) | 同蛋白的非MSA构象对照 |
| 体外rod纤维 | [6CU7，RCSB PDB](https://www.rcsb.org/structure/6CU7) | 同蛋白的另一种纤维对照 |

## 2. 设计依据与任务适配

**关键判断：同一条 α-synuclein 序列对应多种疾病构象。** 输入蛋白序列无法直接区分这些形状，所以输入加入同源残基编号、各构象表面暴露面积和局部 Cα 距离矩阵。距离矩阵与 PDB 的平移旋转无关，表达表位形状；LLM 对其理解仍需结构工具和实验检验。

生成分两轮，每轮四个表位各15条。第一轮覆盖不同长度和局部形状／电荷／极性假设。第二轮读取第一轮60条的真实五构象几何证据，修改表现较弱的序列思路，同时保留多样性。每条序列的提出理由与原始 API 响应均保存。

结构评估用官方 `fixseq.ckpt`：固定LLM提出的序列，分别针对6XYO、6XYP、6XYQ、8A9L、6CU7的同源表位独立采样姿势。模型输入与最小化复合物都包含中心层和两个相邻蛋白层，使采样过程看到纤维层间的空间约束。8A9L使用原始PDB的BIOMT操作1/2/3重建α-syn的三层蛋白部分；该PDB中身份未确定的碎片B/C保留在原始数据中。随后用CHARMM36做受约束局部最小化，按10 kJ mol⁻¹ nm⁻¹的力容差收敛，`maxIterations=0`按[OpenMM官方定义](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LocalEnergyMinimizer.html)表示迭代至收敛。本次初筛每候选每构象一个姿势，Top24增加第二个独立随机采样。两块RTX3090分担候选，PepGLAD推理与距离矩阵用CUDA，OpenMM最小化用NVIDIA OpenCL。各GPU使用3个并行的结构准备／最小化进程。文件读写、PDB整理、少量SASA计算在CPU完成。

本次第二轮提案读取的是第一轮300步预评估的几何反馈，实际输入保存在`api/generation_r2_*_request.json`。最终排序使用全部600个初筛姿势以收敛设置重新优化后的证据；肽链质量检查促成了这项计算修正。复现本次序列提案可读取原始API响应，生成新的实验时按当前收敛流程计算反馈。

**排序输入是结构证据卡。** 隐去生成轮次和提出理由，用随机编号交给LLM。卡片含五构象的接触、冲突、肽键失真、极性与电荷接近程度及序列化学性质。LLM为每条候选给出0–100的实验优先级，代码按数值从高到低排列；相同数值按固定随机编号排列。这些值表达LLM的综合判断，含义是当前候选池中的实验优先级。两次完整排序使用相反输入顺序，最终名次取平均；平均名次是排序稳定性的汇总规则。生成条件、指标定义、采样预算仍属于研究者的实验设计。

## 3. 与已有工作的关系

| 参考工作 | 借鉴的通用思想 | 本项目的具体设计 |
|---|---|---|
| [Wallace 等，Chemical Science 2024](https://doi.org/10.1039/D3SC06245G) | 短肽可用于检验 α-syn 构象识别 | 明确采用三种MSA与两种同蛋白对照的计算证据面板 |
| [PepGLAD，NeurIPS 2024 官方实现](https://github.com/THUNLP-MT/PepGLAD) | 公开固定序列的全原子姿势生成 | 对LLM提出的序列进行五构象独立采样 |
| [ProteinMCP，Protein Science 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13140703/) | LLM调度蛋白工具并利用计算指标做决策 | 去除生成来源信息，直接做跨构象证据排序，加入顺序敏感性与去对照消融 |
| [Lee等，Natural-Language-Guided Generator-Agnostic Shortlisting，2026公开稿](https://arxiv.org/abs/2608.20755) | 用预先计算的结构／界面指标指导LLM生成排序策略；作者注明ICML2026 GenBio Workshop接收 | 本项目让LLM直接比较候选的五构象证据，同时加入序列提案反馈及两次姿势审计 |

代码为本项目新增的集成实现，上游PepGLAD保持来源及许可证。本设计是一项针对MSA场景的组合研究方案，其学术新颖性需进一步文献比对及实验支持。引用已有思想并清楚区分具体改动符合学术署名要求；原创性评价取决于与既有工作的实质差异。

“LLM读取结构指标做筛选”已有先例。本项目可研究的差异集中在同一蛋白的疾病构象选择性、几何证据反馈的序列提出、来源屏蔽与跨构象重复采样。对贡献的论证需要证明这些任务适配提高了真实选择性候选的发现效率。

## 4. 指标如何理解

接触密度是4.5Å内不同肽／蛋白残基对数量除以肽长。冲突密度是小于2Å的重原子对数量除以肽长。中央链接触使五种构象的比较对象一致；蛋白组装体的冲突用于判断姿势能否放进该纤维。8A9L的评估组装体是依据PDB重建的三层α-syn蛋白部分，其余目标用沉积蛋白层数；完整组装体SASA和冲突用于各自构象的背景解释。未知碎片、非蛋白辅因子、结构未解析区域和人为片段末端是模型背景的限制。

“对照差值”=最弱MSA中央链接触密度−最强对照中央链接触密度。这是可审计的几何描述量。三种MSA几何兼容的统计定义为各构象中心链及组装体2Å冲突为0、接触密度至少0.5、肽链相邻残基C–N距离偏离近似理想值1.33Å最多0.3Å。这个宽松肽键阈值检查链的完整性，完整立体化学还需要更全面的评价。这些阈值用于报告统计，LLM排序读取连续原始指标并权衡跨构象证据。LLM的权衡仍受到输入指标、任务提示和采样条件的影响。

极性原子接近程度提供局部作用线索，氢键判定还需要角度；电荷接近程度是简化代理量。本次计算证据支持局部结构假设。真实亲和力、疾病选择性、单体／tau／Aβ交叉结合及药用性质由相应实验确定。

组装体冲突使用同一组坐标状态：优化后的三个模型蛋白层，加上其他层的原始坐标。模型蛋白层的Cα位移RMSD随证据卡一起给出，帮助LLM识别通过扭曲目标结构换取接触的候选。全复合物总能量保留在日志中，排名使用局部几何证据。

这里的positive/negative是目标构象面板的定义：MSA作为希望识别的构象，Lewy/PFF作为需要区分的对照。候选肽的真实结合阳性／阴性标签需要实验产生；本次数据集保存计算假设和结构证据。

## 5. 实际计算实验结果

{table}

表中阈值指标用于描述结果，评价对象与LLM排序使用的几何输入有关。它们提供计算检查依据；评价排序的真实收益需要对同一实验面板做结合测量。本轮生成与反馈的比较也受到候选预算及随机采样影响。

初筛Top24同时满足几何兼容与正对照差值的数量为 **{groups['跨构象 LLM 共识 Top24']['both_geometry_and_contrast']}/24**。当前输出定位为实验优先假设。初筛600个姿势中，肽键偏离超过0.3Å的姿势有 **{groups['全部 LLM 120 条']['peptide_geometry_suspect_poses']}个**；第二次120个姿势对应 **{groups['Top24 第二次结构采样']['peptide_geometry_suspect_poses']}个**。优化设置允许迭代至收敛，但严重异常的原始采样仍可能留下失真，LLM会看到这个连续指标。对照姿势失真时，其结合证据属于未知，须结合进一步采样及实验判断。

输入顺序改变后，两份Top24共同候选为 **{sensitivity['order_top24_overlap']}/24**；全部候选的平均名次变化为 **{sensitivity['mean_absolute_position_change']}**。跨构象共识Top24与“只看MSA”的Top24重叠 **{sensitivity['positive_only_top24_overlap']}/24**。两种输入条件使用相同的120条LLM候选，这些量衡量稳定性和对照信息带来的决策差异。

两次排序同时包含输入顺序变化与模型输出随机性；该对照度量整体排序稳定性。分离各自影响需要增加同一输入的重复调用。仅MSA消融为一次调用，决策差异也包含模型随机变化。

Top24生成轮次分布：`{stats['finalist_rounds']}`；位点分布：`{stats['finalist_sites']}`。第二次采样保持初次候选名单，再调用LLM读取每候选每构象的两次原始观测，调整24条内部的优先级并说明稳定性与限制。最终名单的候选范围在第二次采样前已确定。

## 6. 候选名单

最终LLM给出的排序逻辑（模型意见）：{final_answer['decision_logic']}

{candidate_table}

每条候选的LLM支持证据、限制和两次采样差值见 `shortlist_24.csv`。完整120条排名见 `consensus_ranking.jsonl`，完整五构象卡片见 `all_evidence.jsonl`。五构象的原始及最小化PDB保存在 `poses/`；逐姿势能量与几何指标见 `*_poses.jsonl`。总计完成 **{stats['pose_count']}** 个局部复合物姿势。

## 7. API 用量与费用

完成 **{len(usage)}次**计费用量记录，其中 **{stats['api_calls_used']}次**用于最终结果：8次批量生成、2次跨构象完整排序、1次仅MSA排序、1次基于两次姿势观测的24条精排。另外 **{stats['api_calls_superseded']}次**是输出契约修订前的排序调用，费用统计包含它们。输入 `{tokens['prompt_tokens']:,}` tokens，输出 `{tokens['completion_tokens']:,}` tokens，共 `{tokens['total_tokens']:,}` tokens。每次返回的缓存及推理token信息见 `api/usage.jsonl`。

网关响应提供token用量，货币账单由ChatAnywhere计费。报告保留真实token统计；实际金额以供应商账单为准。可按供应商输入／输出／缓存单价换算费用。通过15条一批、短理由、复用已完成API阶段、GPU计算结构、排序源信息屏蔽来控制调用规模。

## 8. 可复现方式

认证从环境变量 `SYNPep_API_KEY` 读取，密钥保存在运行时临时文件中。公开代码、请求和结果包含研究内容及模型输出。

```bash
cd /media/yjz/htr/synuclein
export SYNPep_API_KEY='<在运行环境配置你的密钥>'
.venv/bin/python llm_design.py prepare --run results/llm_20261006
.venv/bin/python llm_design.py generate --run results/llm_20261006 --round 1 --resume-api
.venv/bin/python llm_run.py --run results/llm_20261006 --start round1_eval
```

完整重跑GPU计算使用上述命令；`--resume-api`复用同一命名阶段的现有响应。修改输入或模型时采用新的结果目录。`llm_run.py --start`允许从指定完成边界继续运行。

## 9. 当前能提出的贡献及验证计划

可研究的贡献是：用同序列的构象差异指导LLM序列提案，用五构象实际采样证据指导LLM排序，并检验来源屏蔽、顺序变化和去对照信息后的决策变化。此次结果提供一套可复现的实现及可检验假设。

湿实验应让本轮LLM候选中“只看MSA”与“跨构象”推荐的候选在同批次、同条件下测量三种MSA与Lewy/PFF的结合，同时加入随机或乱序肽等实验对照。随后比较固定候选预算下的MSA选择性命中率及Top-K富集，并用独立批次验证。获得实验标签后，可评估LLM反馈迭代的增益、校准排序及训练监督ranker。
"""
    (args.run / "REPORT.md").write_text(text)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "replicates", "report"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "report":
        replicates(args)
        subprocess.run([sys.executable, str(ROOT / "llm_design.py"), "rank", "--run", str(args.run),
                        "--view", "replicated", "--evidence", "replicated_evidence.jsonl", "--resume-api"], check=True)
        report(args)
    else:
        {"select": select, "replicates": replicates}[args.command](args)


if __name__ == "__main__":
    main()
