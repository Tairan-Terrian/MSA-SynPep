# MSA-SynPep：正向结构生成与逆折叠评价实验报告

日期：2026-10-09。服务器工程：`/media/yjz/htr/synuclein`。当前结果：`results/current/`。

## 1. 研究目的与当前设计

MSA指多系统萎缩。研究目标是提出偏好MSA相关α-syn纤维构象的短肽，用于后续构象识别实验。构象面板包括MSA结构6XYO、6XYP、6XYQ，以及同蛋白对照8A9L和6CU7。positive/negative表达希望识别和需要区分的目标结构；短肽的真实结合标签由实验产生。

当前流程：闭源LLM提出序列 → PepGLAD固定序列生成复合物姿势 → GPU局部最小化及几何评价 → ProteinMPNN逆折叠评价原序列 → 闭源LLM跨构象排序 → Top24第二姿势 → 两姿势精排。

本次以已有120条闭源LLM序列及600个初次姿势为固定输入，重新计算GPU几何和ProteinMPNN证据，并重新调用LLM排序。已有第二次姿势仅在候选进入本次Top24后复用；其余候选新增第二次采样。序列提出时的反馈来自先前几何计算。当前generate模式已接入逆折叠反馈；本次输入序列保持原样，因此本次结果评估的是新增验证与排序环节。

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

唯一候选数：**120**。已评价复合物姿势：**720**。每个姿势包含完整复合物与肽单独两种条件，每种8个解码顺序，总计**11520**次模型条件评价。几何资格不足姿势：**311**。

Top24第二姿势中，本次新采样**65**个，复用固定输入对应坐标**55**个；所有姿势均由当前代码重新评价。最终Top24每种MSA构象都至少存在一个合格姿势，其中**7/24**条在三种MSA的两次采样中全部合格。至少一次合格与两次稳定合格代表不同的证据强度。

| 实验组 | 候选数 | 三种MSA均有合格姿势 | 五构象均有合格姿势 | NLL对照差为正 | 环境增益对照差为正 |
|---|---:|---:|---:|---:|---:|
| 全部120条，初次姿势 | 120 | 30 | 19 | 11 | 9 |
| 去逆折叠信息 Top24，初次姿势 | 24 | 20 | 11 | 5 | 6 |
| 完整证据 Top24，初次姿势 | 24 | 18 | 13 | 10 | 9 |
| 完整证据 Top24，两次姿势 | 24 | 24 | 21 | 12 | 10 |


完整证据和去逆折叠信息Top24重叠：**14/24**。两次完整排序Top24重叠：**21/24**，全池平均名次变化**5.87**。这些量描述当前计算流程的决策与稳定性。去信息对照只有一次LLM调用，差异中包含LLM随机变化；该实验支持方法行为分析，真实命中率由湿实验评估。

## 5. 当前24条实验优先候选

LLM给出的当前判断：分数是相对实验优先级，不是结合或选择性概率。综合全部姿势，先看三种MSA的无冲突几何覆盖、真实接触、受体位移和C-N偏差，再比较同序列在MSA与8A9L、6CU7上的NLL及环境增益；可信的强控制姿势不能被较差重复平均掉，不合格控制只代表兼容性未知。小差异须对照解码波动和独立姿势差异，后者常更大；接触、极性及电荷邻近不能单独当作亲和力。首批建议llm_r2_site_2_008、llm_r2_site_2_012、llm_r2_site_3_001，并以llm_r1_site_3_003或llm_r1_site_3_014扩展site_3序列/长度多样性；llm_r1_site_2_005作为同系列备选。llm_r1_site_2_004、llm_r1_site_2_000虽局部几何可靠，但控制兼容性削弱疾病区分。所有序列疏水连续段均为1，不能据此额外区分。当前证据只支持结构模型假设，需对三种MSA及两个控制开展匹配条件实验验证。

| 排名 | ID | 序列 | 位点 | LLM优先级 | NLL对照差 | 环境增益对照差 |
|---:|---|---|---|---:|---:|---:|
| 1 | llm_r2_site_2_008 | HFSPGQDVTHSN | site_2 | 90 | 0.5456 | 0.79025 |
| 2 | llm_r2_site_2_012 | RFAQEADQVR | site_2 | 82 | 0.47415 | -0.01881 |
| 3 | llm_r2_site_3_001 | KDTYSPGTLEK | site_3 | 73 | 0.07959 | 0.1187 |
| 4 | llm_r1_site_3_003 | EQYSPNGTWQK | site_3 | 68 | -0.01648 | -0.05895 |
| 5 | llm_r1_site_2_005 | RFSPGQDVTRSN | site_2 | 66 | 0.07702 | -0.04397 |
| 6 | llm_r1_site_3_014 | KEQSWTNQPGTQFNSQTEK | site_3 | 64 | 0.0038 | -0.22862 |
| 7 | llm_r1_site_2_000 | RFSGSGQDVTRS | site_2 | 62 | -0.16047 | -0.0203 |
| 8 | llm_r1_site_2_012 | RQNSGNGSQEVSRNQTS | site_2 | 58 | 0.14819 | 0.12933 |
| 9 | llm_r1_site_2_004 | KYSQGSGNGSGQDVTRNQS | site_2 | 57 | 证据不足 | 证据不足 |
| 10 | llm_r1_site_2_011 | KQNSGSGQDITRNQS | site_2 | 56 | 0.1069 | 0.08359 |
| 11 | llm_r1_site_2_001 | RYSNGSGNDVTRN | site_2 | 54 | -0.31657 | -0.14824 |
| 12 | llm_r1_site_2_007 | RFSNPGSQDVTRNS | site_2 | 51 | 0.05425 | 0.11863 |
| 13 | llm_r2_site_2_007 | RFSPGQEVTRSN | site_2 | 48 | -0.04983 | 0.03688 |
| 14 | llm_r1_site_2_014 | RYNQSGNGSGSNQDVTRNQSTS | site_2 | 46 | 0.04789 | -0.08711 |
| 15 | llm_r2_site_2_013 | RFAQEADQVRSAQNT | site_2 | 41 | 证据不足 | 证据不足 |
| 16 | llm_r2_site_3_002 | EKSFTPGTQLEK | site_3 | 40 | 0.27675 | 0.22778 |
| 17 | llm_r2_site_2_006 | RFSPGQDVTESN | site_2 | 38 | 证据不足 | 证据不足 |
| 18 | llm_r2_site_3_012 | DEKSYTQAPGTQLSNQTEK | site_3 | 37 | 0.12024 | 0.20116 |
| 19 | llm_r1_site_2_006 | KYSPNGQDVSRNQ | site_2 | 35 | -0.04841 | 0.23946 |
| 20 | llm_r1_site_3_001 | EKWNTDGNWQK | site_3 | 34 | -0.06166 | -0.52986 |
| 21 | llm_r1_site_3_009 | EQNWTSTQNGFQSTK | site_3 | 32 | 0.33022 | -0.09914 |
| 22 | llm_r1_site_3_011 | EKSYQTPGNWQSEK | site_3 | 29 | -0.44866 | -0.37058 |
| 23 | llm_r1_site_3_010 | KEQWSPGTFQEK | site_3 | 23 | -0.08599 | 0.00186 |
| 24 | llm_r1_site_3_007 | EQWQTSHNTQYQKE | site_3 | 18 | -0.1275 | -0.48126 |


详细支持与限制见shortlist_24.csv；所有姿势均保留原始分数及资格标记。证据不足表示相应对照差无法可靠汇总。

## 6. GPU、费用与复现

PepGLAD、OpenMM及几何计算沿用CUDA/NVIDIA OpenCL，两块RTX3090并行。ProteinMPNN在CUDA上分担姿势，各姿势8种解码顺序合并成批。文件整理和PDB解析在CPU进行。当前服务器其他任务占用显存，结构采样批量设为2，各GPU安排1个最小化进程。

本次新增API调用**4**次，输入**122,269** tokens，输出**8,814** tokens，总计**131,083** tokens。模型返回版本：`gpt-6.1-sol-2026-09-29`。序列来源的历史请求作为data/inputs/generation_provenance保存，当前用量统计覆盖本次调用。实际货币费用以供应商账单为准。

```bash
cd /media/yjz/htr/synuclein
export SYNPep_API_KEY='<运行时密钥>'
.venv/bin/python pipeline.py --run results/current --mode reuse
```

从新序列开始的当前设计：`.venv/bin/python pipeline.py --run results/new_experiment --mode generate`。第二轮提案读取几何及逆折叠逐位置反馈。生成模式的第二姿势来自本次序列，复用模式仅复用与固定输入对应的姿势。已完成API命名阶段在同一实验目录继续运行时复用；输入或模型改变时使用新的实验目录。

## 7. 输出与贡献验证

all_evidence.jsonl保存全部候选的初筛卡片；initial_inverse.jsonl及replicate_inverse.jsonl保存逐姿势/逐位置证据；pose_scores.csv与residue_scores.csv便于分析；consensus_ranking.jsonl、shortlist_24.csv及FASTA保存当前排名。experimental_binding_matrix_template.csv提供五构象的实验信号录入字段。

可检验的贡献是：利用正向姿势与逆向序列适配的一致性，结合靶标环境对原序列的支持及同蛋白构象对照，改善MSA选择性候选的选择。后续应让完整证据与去逆折叠信息推荐的候选在同一条件下测量三种MSA和两种对照，比较相同预算下的选择性命中率和Top-K富集。当前报告给出计算证据、排序变化和实验候选，真实结合数据是确认增益的依据。
