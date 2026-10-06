# MSA-SynPep：闭源 LLM 候选生成与排序

当前流程用闭源 LLM 分两轮生成 120 条 10–25 aa 短肽，再依据五种 α-synuclein 构象的实测计算证据进行 LLM 排序，选出 24 条优先候选。MSA 指多系统萎缩。研究目标是提出偏好 MSA 纤维构象的短肽假设，用于后续结合实验。

## 本轮流程

1. 读取公开结构及四个任务表位，构造同源残基的构象上下文。
2. LLM 每个表位提出 15 条肽，共 60 条；GPU 计算其五构象姿势和局部几何证据。LLM 读取第一轮反馈，再提出 60 条。
3. LLM 对这 120 条新序列做跨构象排序。反转输入顺序重复排序，取两次名次的均值，选择 Top24。
4. Top24 在五种构象上各增加一个独立姿势，LLM 读取两次观测，完成 24 条内部精排。

全部候选序列由 LLM 提出。PepGLAD 官方 `fixseq.ckpt` 保持序列固定，提供三维姿势；OpenMM 在 GPU 上按收敛设置优化，距离计算使用 CUDA。三层蛋白环境体现纤维的相邻层约束。Lewy 三层 α-syn 蛋白结构由原始 PDB 的 BIOMT 操作重建。

排序输入含接触、碰撞、肽键失真、目标结构位移及序列化学性质。LLM 为每条候选给出 0–100 的实验优先级，程序按数值排列，同分按固定随机编号排列。这些值是综合判断，真实亲和力和选择性由结合实验确定。仅 MSA 信息消融也使用相同的 120 条新序列。

## 代码与输入

- `llm_design.py`：构象上下文、两轮 API 序列提案、完整候选评价及排序。
- `llm_evaluate.py`：固定序列姿势采样、GPU 优化、计算证据卡。
- `llm_run.py`：两块 GPU 并行调度和阶段续跑。
- `llm_report.py`：排序汇总、Top24 复采样精排、CSV/FASTA 和 Markdown 报告。
- `data/pdb/`：公开原始结构。MSA 为 6XYO/6XYP/6XYQ，对照为 8A9L/6CU7。
- `results/selected_epitopes.csv`、`results/site_*_pocket.json`：任务表位定义，中心残基 39、80、66、50。
- `vendor/PepGLAD/`：上游代码、许可证及官方预训练权重，来源为 [PepGLAD](https://github.com/THUNLP-MT/PepGLAD)。

## 运行

服务器根目录为 `/media/yjz/htr/synuclein`，使用两块 RTX 3090。已配置 Python 3.8.5、PyTorch 1.13.0+cu117、NumPy 1.23.5、SciPy 1.10.1、Biopython 1.80、FreeSASA 2.2.1、torch-scatter 2.1.2、OpenMM 8.1.1、PDBFixer 1.9.0。PepGLAD 及几何计算用 CUDA，OpenMM 用 NVIDIA OpenCL。

```bash
cd /media/yjz/htr/synuclein
export SYNPep_API_KEY='<在运行环境配置你的密钥>'
.venv/bin/python llm_design.py prepare --run results/llm_20261006
.venv/bin/python llm_design.py generate --run results/llm_20261006 --round 1 --resume-api
.venv/bin/python llm_run.py --run results/llm_20261006 --start round1_eval
```

`--api-base` 默认 `https://api.chatanywhere.tech/v1`，`--model` 默认用户指定的 `gpt-6.1-sol`。密钥从环境变量读取。`--resume-api` 复用同一目录的已完成响应；更换实验输入或模型时创建新的结果目录。

`llm_run.py --start` 从指定阶段继续。`--start reoptimize` 使用保存的原始姿势重新优化 120 条新候选，随后继续排名、复采样和报告。新实验从 `round1_eval` 开始即可。

## 实验输出

本轮正式结果目录为 `results/llm_20261006/`：

- `REPORT.md`：完整实验汇报、设计依据、实际结果和解释。
- `round1_candidates.jsonl`、`round2_candidates.jsonl`：两轮各 60 条 LLM 序列与提案理由。
- `all_evidence.jsonl`、`candidate_scores.csv`：120 条候选的五构象证据。
- `consensus_ranking.jsonl`：两次跨构象排序的平均名次。
- `shortlist_24.csv`、`shortlist_24.fasta`：最终实验优先名单及 LLM 的支持和限制。
- `poses/`、`*_poses.jsonl`：600 个初筛姿势及 120 个 Top24 第二次姿势的原始坐标、优化坐标和指标。
- `api/`：实际 API 请求、响应和供应商返回的 token 用量；费用按供应商账单结算。
- `experimental_binding_matrix_template.csv`：后续湿实验录入模板，结合信号字段留待实验测量。

报告明确区分参考工作的通用思路与 MSA 场景的具体设计。当前结果是可复现的计算假设，后续贡献评估围绕真实 MSA 选择性命中率展开。
