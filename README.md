# MSA-SynPep：LLM设计与正反向模型评价

当前设计用闭源LLM提出短肽，用PepGLAD预测固定序列的结合姿势，再用ProteinMPNN评价原序列与该姿势的匹配程度。LLM读取三种MSA构象和两种同蛋白对照的证据，给出实验优先级。MSA指多系统萎缩。

## 当前流程

1. 用同源残基编号、局部形状、表面暴露及五构象差异构建四个表位的上下文。
2. LLM分两轮各生成60条10–25 aa肽；生成模式的第二轮读取第一轮几何指标及逆折叠逐位置反馈。
3. PepGLAD保持序列固定，分别在6XYO、6XYP、6XYQ、8A9L、6CU7的同源表位采样，GPU最小化并评价几何。
4. ProteinMPNN在完整三层受体+短肽环境中评价原序列，再保留同一肽骨架、移除受体，评价肽单独环境。每个条件使用8种匹配的肽位置解码顺序。
5. LLM读取完整证据两次，反转输入顺序，以平均名次选择Top24；一次去逆折叠信息排序用于比较决策变化。
6. Top24在每种构象补足第二姿势，重新进行几何/逆折叠评价，LLM结合两个观察精排。

ProteinMPNN仅评价短肽链K；受体序列固定并先进入解码条件。complex_nll是原序列按肽长平均的负对数概率，越低表示序列与骨架更匹配。context_gain=peptide_only_nll−complex_nll，表达模型从靶标环境得到的支持。分数是计算证据，实际亲和力和选择性由结合实验确认。

当前几何资格沿用宽松规则：中央链及组装体2Å冲突均为0，肽链C–N距离对近似1.33Å的最大偏离≤0.3Å。完整原始分数保留，资格不足姿势的逆折叠证据在排序卡片中标为未知。对照姿势未知时继续采样与实验测量。

## 代码和目录

- `design.py`：表位上下文、LLM序列提案、LLM完整证据/去信息/最终排序。
- `structure.py`：固定序列复合物采样、GPU最小化和几何指标。
- `inverse.py`：ProteinMPNN多链原序列评分、肽单独评分、逐位置分数与解码顺序变化。
- `evidence.py`：多姿势卡片与跨构象描述量。
- `pipeline.py`：两块GPU调度、当前完整流程和续跑阶段。
- `report.py`：Top24选择、消融比较、CSV/FASTA与Markdown报告。
- `data/pdb/`：公开结构原始文件。
- `data/epitopes/`：四个任务表位定义及其选择依据。
- `data/inputs/`：固定输入序列、结构坐标及生成来源。`PROVENANCE.json`说明这些输入的产生方式。
- `vendor/PepGLAD/`：固定序列姿势采样器及许可证。
- `vendor/ProteinMPNN/`：官方模型实现、许可证、安全张量格式权重及`SOURCE.json`。
- `results/current/`：当前版本正式结果；根目录保留当前六个模块。

## 运行环境

服务器工程为`/media/yjz/htr/synuclein`，使用两块RTX3090。GPU环境为Python3.8.5、PyTorch1.13.0+cu117、NumPy1.23.5、SciPy1.10.1、Biopython1.80、FreeSASA2.2.1、torch-scatter2.1.2、OpenMM8.1.1、PDBFixer1.9.0。

PepGLAD、ProteinMPNN及几何计算使用CUDA；OpenMM使用NVIDIA OpenCL。PDB读写和小规模输入整理使用CPU。结构采样批量为2，每块GPU安排1个最小化进程。

ProteinMPNN使用[官方v_48_020模型](https://github.com/dauparas/ProteinMPNN)。原模型张量通过临时PyTorch2.4环境以`weights_only=True`转换为服务器兼容格式，浮点训练元数据保存于`SOURCE.json`；当前权重安全加载同样使用`weights_only=True`。模型参数固定，推理坐标扰动为0。

## 运行方式

固定已有序列和初次坐标，重新评价并重新排序：

```bash
cd /media/yjz/htr/synuclein
export SYNPep_API_KEY='<在运行环境配置密钥>'
.venv/bin/python pipeline.py --run results/current --mode reuse
```

从序列提案开始运行当前完整设计：

```bash
.venv/bin/python pipeline.py --run results/new_experiment --mode generate
```

`--start evaluate|inverse|rank|replicate|report`指定已完成边界后的阶段。已有API响应仅用于同一实验目录、相同命名阶段的续跑；修改输入或模型时使用新的实验目录。默认API地址为`https://api.chatanywhere.tech/v1`，模型为`gpt-6.1-sol`，凭据从环境变量读取。

复用模式的输入序列原来采用几何反馈生成。本次固定这些序列，评价新增逆折叠证据对排序的影响；当前生成模式的第二轮已接入逆折叠反馈。生成模式按当前序列独立采样第二姿势。

## 输出与解释

当前正式报告为`results/current/REPORT.md`。`pose_scores.csv`和`residue_scores.csv`保存逐姿势/逐位置指标；`candidate_scores.csv`保存120条汇总；`shortlist_24.csv`及FASTA保存最终名单。`api/`保存本次请求、响应及token用量，原序列生成来源单独作为输入保存。

去逆折叠信息对照和输入顺序比较反映计算决策及稳定性。检验准确性增益需要让推荐候选在同一条件下测量三种MSA与两种对照的结合，比较固定候选预算下的选择性命中率。

## 来源

- [ProteinMPNN，Science 2022](https://doi.org/10.1126/science.add2187)，[官方实现](https://github.com/dauparas/ProteinMPNN)。
- [PepGLAD，NeurIPS 2024官方实现](https://github.com/THUNLP-MT/PepGLAD)。
- [Wallace等，Chemical Science 2024](https://doi.org/10.1039/D3SC06245G)。

本项目实现MSA场景的多构象证据整合及靶标环境对原序列的支持比较，来源和实验边界在报告中明确记录。
