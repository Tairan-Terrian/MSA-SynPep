# MSA-SynPep Phase 1：AI 筛选与构象特异打分实验报告

**日期：**2026-09-30  
**服务器目录：**`/media/yjz/htr/synuclein`  
**结果目录：**`/media/yjz/htr/synuclein/results`

## 一、我们要解决的科学问题

同一种 α-synuclein 蛋白可以折成多种纤维形状。MSA 与帕金森病、路易体痴呆的病理纤维虽然来自同一蛋白，表面形状却不同。项目的目标是找出短肽：它在 MSA 纤维表面贴合得好，在其他构象表面贴合得较差。这样的肽可以作为区分病理构象的研究工具；后续经过实验验证，可探索诊断探针等用途。

[Wallace 等，2024](https://doi.org/10.1039/D3SC06245G) 已展示针对某一 α-synuclein 纤维构象设计短肽并进行实验验证的路线。本阶段把目标明确设为 MSA 患者来源纤维，先形成可复现的计算候选集。**构象特异性最终由真实结合实验判定。**

## 二、目标、输入与交付物

本次对应方案的前四个计算步骤：结构库、MSA 特异表位、短肽生成、计算交叉筛选。实验结合矩阵、主动学习更新和独立样本验证属于后续湿实验阶段。`results/experimental_binding_matrix_template.csv` 提供记录模板，测量值留待实验填入。

| 结构 | 角色 | 使用方式 |
|---|---|---|
| [6XYO](https://www.rcsb.org/structure/6XYO) | MSA Type I | 表位发现、短肽生成与主复合物优化 |
| [6XYP](https://www.rcsb.org/structure/6XYP)、[6XYQ](https://www.rcsb.org/structure/6XYQ) | MSA Type II-1 / II-2 | 检查候选对其他 MSA 构象的兼容性 |
| [8A9L](https://www.rcsb.org/structure/8A9L) | PD/PDD/DLB 的 Lewy fold | 同蛋白、不同病理构象的高难度对照 |
| [6CU7](https://www.rcsb.org/structure/6CU7) | 体外形成的 α-synuclein rod 纤维 | 体外 PFF 的结构代理对照 |
| [8Q2L](https://www.rcsb.org/structure/8Q2L)、[6SHS](https://www.rcsb.org/structure/6SHS) | tau、Aβ 纤维 | 非 α-syn 蛋白对照的结构存档与后续实验设计 |

单体 α-synuclein 是构象变化明显的无序蛋白，本阶段将其列为湿实验对照，采用真实单体样本测定结合。tau 与 Aβ 没有可用的同源残基映射；它们已进入结构库，结合选择性将在直接交叉结合实验中验证。体外 PFF 的实际结构取决于制备条件；6CU7 是明确注明的结构代理。

## 三、实现方法

### 1. 结构库

从 RCSB 下载 7 个公开 PDB 文件。脚本 `prepare.py` 提取疾病、折叠类型、链和残基范围，写入 `results/structure_inventory.csv`。6XYO/6XYP/6XYQ 各提供多层纤维片段；8A9L 主要提供单层 α-synuclein 链。下游比较分别在对应的单链或全纤维尺度进行。

### 2. 表位

以 6XYO 的内部重复层链 C 为参考，用 FreeSASA 计算纤维环境下的残基暴露面积，并将链 C 与 8A9L 链 A 的同源残基局部距离矩阵比较。表位中心同时满足 MSA Type I、Type II-1、Type II-2 三种结构的最小暴露面积 ≥50 Å²，再按照 Type I 暴露与 Lewy fold 局部形状差异排序。中心间距至少 13 Å，从相邻纤维层收集 10 Å 内可接触残基。表位定义保存在 `site_*_pocket.json`。

| 表位 | 中心残基 | 口袋残基数 | Type I 暴露面积（Å²） | 三种 MSA 最小暴露面积（Å²） | 与 Lewy fold 局部距离差（Å） |
|---|---:|---:|---:|---:|---:|
| site_1 | 39 | 18 | 122.89 | 83.89 | 11.82 |
| site_2 | 80 | 15 | 100.98 | 89.32 | 13.69 |
| site_3 | 66 | 14 | 95.55 | 95.55 | 21.32 |
| site_4 | 50 | 16 | 86.41 | 86.41 | 18.04 |

表中距离差是局部几何差异描述量，反映结构变化幅度；它本身不代表结合亲和力。首次运行曾选取残基 74，其在 MSA Type II 中的暴露面积仅约 35–40 Å²。已将该次结果归档到 `results/pilot_typeI_only/`，并以三种 MSA 都暴露的残基 50 取代。原有相同口袋、相同随机种子的三个表位结果继续使用。

### 3. 短肽生成与结构优化

采用公开的 [PepGLAD 模型和权重](https://github.com/THUNLP-MT/PepGLAD)进行结合位点条件下的全原子短肽序列—构象共设计。选择该模型的理由是它直接支持短肽长度范围并提供公开推理权重，适合本阶段快速构建可运行基线。每个表位生成 30 条、长度 10–25 aa，总计 **120 条**；两个 RTX 3090 分担生成任务，种子为 `20260930`。

PepGLAD 的公开权重来自通用蛋白—短肽复合物基准，MSA 病理纤维属于迁移应用；模型输出用于提出候选。MSA 纤维沉积结构中的部分非蛋白辅因子没有进入生成模型，实验阶段需检验其影响。

生成的肽—MSA 复合物随后用 OpenMM CHARMM36 力场优化。优化在两块 RTX 3090 的 OpenCL 设备上并行，纤维重原子受到较强位置约束，肽受到较弱约束并调整局部构象。几何质量以优化后能量、接触和小于 2 Å 的重原子冲突衡量。高能量或明显冲突的结构进入质量剔除集。

### 4. 构象交叉筛选与排名

候选肽首先在原始 MSA Type I 复合物中计算接触和冲突。随后使用表位同源残基的局部刚体对齐，将肽的姿态转移至 MSA Type II-1、Type II-2、Lewy fold 与 6CU7，再计算相同指标。接触定义为肽—目标重原子距离 ≤4.5 Å 的不同残基对；冲突定义为距离 <2.0 Å 的重原子对。

每个构象的兼容性代理量为 `I = (接触数 − 3 × 冲突数) / 肽长`。MSA 得分取三种 MSA 构象中较低的一项；Lewy fold 比较同源内部链，6CU7 比较完整纤维。两个对照的偏好差分别写入 `lewy_margin` 和 `pff_margin`，最终 `selectivity_proxy` 取较低偏好差。排名同时考虑最弱 MSA 构象得分、选择性代理量、疏水性与连续疏水残基数。几何合格定义为优化后总能量低于 10⁸ kJ/mol、MSA Type I 中无小于 2 Å 的重原子冲突、总接触数至少达到肽长，且内部链 C 接触数至少达到肽长的四分之一；化学筛选定义为 GRAVY 疏水指数 ≤0.8 且最长连续疏水片段 ≤4 aa。**A 级**还要求三种 MSA 构象均具有正的兼容性得分、两种对照偏好差均为正、局部姿态转移对齐 RMSD 均 ≤5 Å，且满足接触与化学筛选；其余几何合格、化学筛选通过的候选标为 **B 级探索候选**。

**解释边界：**上述量是结构兼容性代理指标，单位和物理意义均不同于 Kd、EC50 或结合自由能。姿态转移评估固定结合模式在其他构象上的兼容程度；肽在对照构象上重新寻找新结合位点的可能性需要实验或更完整的重对接检验。8A9L 主要提供单层沉积模型，Lewy fold 对照采用同源链比较。有限层数的 MSA 坐标可能产生纤维端面效应，内部链接触门槛用于降低该影响；患者样本中的修饰、辅因子和不同纤维制备条件需要后续实验覆盖。

## 四、实际结果

| 指标 | 本次结果 |
|---|---:|
| 纳入结构 | 7 个 PDB |
| 选定 MSA 表位 | 4 个 |
| PepGLAD 生成肽 | 120 条 |
| 几何质量合格 | 76 条 |
| 三种 MSA 构象兼容性代理分均为正 | 19 条 |
| A 级候选 | 2 条 |
| 结构优先候选 | 18 条 |
| 实验筛选面板 | 24 条 |

实验面板按照每个位点至少 4 条的覆盖要求，再用综合排名补足到 24 条；其中 18 条三种 MSA 构象兼容性代理分均为正，列在 `results/priority_candidates.csv` 与 `.fasta`。另外 6 条用于探索位点和分级边界。全部 120 条的序列、结构路径和指标见 `results/candidate_scores.csv`；完整面板见 `results/shortlist_24.csv` 与 `.fasta`。下表列出 24 条面板候选。`MSA三型`表示三种 MSA 构象的兼容性代理分均为正；`对照对齐`表示 Lewy/PFF 姿态转移的局部对齐 RMSD 均 ≤5 Å。

| 候选 ID | 位点 | 序列 | 等级 | MSA三型 | 对照对齐 | ΔLewy | ΔPFF | 排名分 |
|---|---|---|---|---|---|---|---|---|
| site_1_022 | site_1 | AMKKTYQVEVTV | A | 是 | 可靠 | 1.50 | 3.83 | 2.25 |
| site_1_021 | site_1 | SIEDDNLWRAFPLDNIQVLHNQRR | A | 是 | 可靠 | 1.04 | 0.83 | 1.54 |
| site_1_024 | site_1 | QYMEEIWRTSF | B | 是 | 可靠 | 0.64 | 9.27 | 1.09 |
| site_1_005 | site_1 | PAHNPIPPYEM | B | 是 | 可靠 | 0.09 | 3.27 | 0.55 |
| site_1_002 | site_1 | ILRYTRGEKRNEYDDLRPH | B | 是 | 可靠 | 0.21 | 3.58 | 0.47 |
| site_3_012 | site_3 | NHSPILLLTALTGH | B | 是 | 可靠 | -0.07 | 8.14 | 0.36 |
| site_3_002 | site_3 | TPSIILRLRRGEKRNEYSD | B | 是 | 可靠 | 0.05 | 18.32 | 0.21 |
| site_1_019 | site_1 | ADLKLLLRKARL | B | 是 | 可靠 | -0.08 | -0.08 | 0.00 |
| site_1_011 | site_1 | IIECHTLFDPD | B | 是 | 可靠 | -0.46 | 4.09 | -0.36 |
| site_2_020 | site_2 | SAPYGMLRLEYS | B | 是 | 低置信 | 13.33 | 8.50 | 8.75 |
| site_4_025 | site_4 | IADARPPIRFGTILQ | B | 是 | 低置信 | 5.93 | 19.87 | 6.47 |
| site_2_003 | site_2 | VPRERNQWCGKWQV | B | 是 | 低置信 | 8.14 | 5.00 | 5.57 |
| site_2_004 | site_2 | RSAFVRLYENLLDNWA | B | 是 | 低置信 | 4.94 | 11.50 | 5.19 |
| site_2_002 | site_2 | ESIILRLTRGEKRNEYSDS | B | 是 | 低置信 | 2.05 | 14.37 | 2.32 |
| site_2_025 | site_2 | SLIIDARPPIRFGTI | B | 是 | 低置信 | 0.53 | 5.00 | 0.67 |
| site_2_019 | site_2 | EDWENHADLKLL | B | 是 | 低置信 | 1.50 | 0.17 | 0.25 |
| site_2_012 | site_2 | HSPILLLTALRGHG | B | 是 | 低置信 | -0.14 | 13.57 | 0.07 |
| site_2_005 | site_2 | VRLNQIPAHQP | B | 是 | 低置信 | 3.36 | -0.64 | -0.45 |
| site_3_027 | site_3 | VNGASDPLNDVKEVGYKSRELRAR | B | 待验证 | 可靠 | 7.62 | 6.54 | 5.92 |
| site_1_006 | site_1 | CEDHEGNVERERGQINFS | B | 待验证 | 可靠 | 4.50 | 8.67 | 3.61 |
| site_3_015 | site_3 | CHFQTKKNWWDDGPLAHLNVWSET | B | 待验证 | 可靠 | 3.54 | 5.46 | -2.12 |
| site_4_020 | site_4 | APYGMLRLQYLA | B | 待验证 | 低置信 | 7.50 | 20.08 | 5.58 |
| site_4_001 | site_4 | AKVEALRLPHL | B | 待验证 | 低置信 | 8.27 | 14.18 | 4.64 |
| site_4_023 | site_4 | TWGLSYIKNPELAVPKV | B | 待验证 | 低置信 | 1.71 | 12.59 | 1.06 |

初次 Type-I-only 运行的未优化分数保存在 `results/pilot_typeI_only/candidate_scores_pre_relax.csv`，用于审计表位修正与结构优化前后的差别。最终排名以修正后的表位和优化后结构为准。优化后 PDB 在 `results/relaxed/`，原始 PDB 在 `results/generated/`，每条结构的能量见 `minimization_gpu0.jsonl`、`minimization_gpu1.jsonl` 和 `minimization_gpu0_site4.jsonl`。

## 五、结论与下一步判断

前四个计算步骤已形成可运行链条：公开结构 → 四个表位 → 120 条 AI 生成短肽 → GPU 优化及三种 MSA / 两种 α-syn 对照的结构筛选。`priority_candidates.csv` 中的 18 条可作为首批重点评估对象，其中 2 条具备更完整的 A 级计算证据；`shortlist_24.csv` 另纳入 6 条探索候选，用于检查模型和表位选择的边界。科学结论目前停留在结构假设层面。

下一步建议在同一实验体系下对 MSA、PD/PDD/DLB、体外 PFF、单体、tau 与 Aβ 测量结合信号，并加入乱序肽及背景对照。每条肽的真实 MSA 特异性可按统一标准定义为 MSA 结合信号与最高对照结合信号的差值或比值。获得首轮结合矩阵后，再训练文档中的 MSA-SynPep Ranker，开展主动学习迭代。当前没有可用于训练特异性监督模型的 MSA 结合标签，故本次使用透明的结构代理排序，并保留全部原始指标供后续校准。

## 六、主要来源

1. [Wallace HM 等，*Chemical Science*，2024](https://doi.org/10.1039/D3SC06245G)：构象特异 α-synuclein 短肽设计与实验验证先例。
2. [Schweighauser M 等，*Nature*，2020](https://www.nature.com/articles/s41586-020-2317-6)：MSA Type I/II 病理纤维结构。
3. [Yang Y 等，*Nature*，2022](https://www.nature.com/articles/s41586-022-05319-3)：PD/PDD/DLB Lewy fold 结构。
4. [Kong X 等，NeurIPS 2024 / arXiv](https://arxiv.org/abs/2402.13555)：PepGLAD 全原子短肽生成模型。
