# MSA-SynPep 三模块实验汇报

本轮在清空服务器旧 results 后，从头生成候选、预测结构、计算指标并完成排序。运行目录为 bounded_20261013；启动日期为2026-10-13，完成日期为2026-10-14，正式运行约71.9分钟。

已得到：**α-syn结合肽Top10、LIR Top10、100条完整融合序列、最终融合肽Top10**。全部序列均标注长度。

模块一生成的是用于抓取α-syn的短肽。α-syn本身是研究靶标；LIR负责与LC3B相容，Linker将两个功能模块连接成完整嵌合肽。

## 本轮流程

1. **结合肽模块：LLM生成 → PepGLAD结构评价 → LLM排序。** 用三种MSA结构（6XYO、6XYP、6XYQ）与两种纤维对照（8A9L、6CU7）比较四个候选区域，LLM两批共生成120条新肽，每个区域30条。五种纤维先各预测一个姿势，再对20条重点候选复核第一姿势并补第二姿势；入围肽另计算9A1A的8个有限单体模型。LLM读取本轮结构证据，选出Top10。
2. **LIR模块：LLM有限变体生成 → LC3B模板相容性评价 → LLM排序。** 以人p62片段SGGDDDWTHLSS为来源，保留WTHL核心，改变1–3个侧翼残基，生成30条新变体。使用2ZJD的LC3B–p62实验复合物映射核心位置，每条检查3个模板样本，并与天然12肽和文献20肽参考比较，选出Top10。模板放置提供已知识别方式的先验。
3. **融合模块：连接组装 → 完整结构及短模拟评价 → LLM排序与复核。** 两组Top10形成100对，每对比较4/8/12 aa的GSGS连接子及两种模块顺序，共600个方案；每个方案在五种纤维上各采样8个连接构象，共24,000个有限构象。先检查完整肽与LC3B、纤维能否同时容纳，再将12对合格模型在共同靶标6XYP上各模拟10 ps，LLM选出最终Top10，每条再以不同随机种子复核10 ps。

本轮使用两块RTX 3090进行模型推理及适合并行的科学计算；ProteinMPNN已移出主流程。额外位置调整集中于重点复核池，一次20步；侧链最小化最多150步。

## 生物学信息具体放在哪里

**设计与解释原则写入英文提示词，真实计算观察作为紧凑输入。** 完整坐标和逐残基记录保存在结构文件中。生成说明提出可检查的假设；PepGLAD实际接收序列、长度、受体坐标与口袋。

- [结合肽生成提示词](prompts/binder_generate.txt)：表面侧链互补、层数和跨链遮挡、连接末端、溶解性与单体风险。
- [结合肽排序提示词](prompts/binder_rank.txt)：MSA覆盖、对照兼容性、重复支持、实际接触和连接空间；失败对照保留为证据缺口。
- [LIR生成提示词](prompts/lir_generate.txt)：来源与片段边界、WTHL核心、有限侧翼替换、未修饰单LIR和天然参考单列。
- [LIR排序提示词](prompts/lir_rank.txt)：双口袋相容性、模板先验、参考对照、条件能量与真实亲和力的区别。
- [融合排序提示词](prompts/fusion_rank.txt)：完整几何、LC3B遮挡、双界面接触保持、模块连接风险及膜环境的证据范围。

提示词组织参考了“先整理靶标资料，再形成具体设计说明”的思路。本项目采用LLM序列提案与PepGLAD结构评价。文章中直接将自然语言输入结构生成器的方式属于另一种模型接口。[Designing Protein Structures from Natural Language，2026年预印本](https://arxiv.org/abs/2610.13917)

## 实际结构证据

结合肽初筛完成600个姿势，重点复核完成200个姿势，单体对照完成160个姿势。初筛的完整几何资格为0/600；有限复核后有16条序列在至少一种MSA构象上获得合格姿势，其中1条覆盖三种MSA。入选Top10的19个合格纤维姿势均未在4.5 Å内接触人为封端原子；完整α-syn缺失尾部的影响仍待评估。四个区域中心为58、43、94、85，其中94靠近部分结构的解析边界。

LIR及参考共33条序列、99个模板样本，每条均有1个几何及双口袋合格样本。侧翼扰动样本的失败意味着重复支持有限。LC3B模板来自[2ZJD](https://www.rcsb.org/structure/2ZJD)（人LC3B、鼠p62），识别核心映射至人p62片段；本轮评价使用未修饰线性L肽。

100个代表融合模型中，20个支持两种MSA构象，46个支持一种，34个在本轮有限采样中获得的支持为0。6XYP上共有40个合格完整模型；按照几何覆盖、可行采样比例及来源多样性安排12个模型进入短模拟。此资源调度规则由项目设置，最终相对名次由LLM依据实际观察给出。

## 模块一：α-syn结合肽Top10

|名次|编号|完整序列|长度 aa|
|---|---|---|---|
|1|B067|EQTSDNAGKRSPATNQ|16|
|2|B002|NEDSGQETSNQS|12|
|3|B006|SGDQENSTEDQSNTS|15|
|4|B064|QDYTENGSARQST|13|
|5|B004|SDEQPGNEDTQSGS|14|
|6|B062|EQTSDNAGKRS|11|
|7|B001|SDQEGNDESTS|11|
|8|B116|ARKTQGDESNPAGQTNASGS|20|
|9|B049|NSRLQGTEQSNTGS|14|
|10|B068|NDRTQESGAKSTGAPQN|17|

详细证据及逐条理由：[模块一报告](module1/REPORT.md)、[结构证据表](module1/top10_evidence.csv)。

## 模块二：LIR Top10

|名次|编号|完整序列|长度 aa|
|---|---|---|---|
|1|L015|SGGDDDWTHLES|12|
|2|L012|SGGDDDWTHLTS|12|
|3|L010|SGGDDEWTHLSS|12|
|4|L021|TGGDDDWTHLTS|12|
|5|L014|SGGDDDWTHLDS|12|
|6|L011|SGGDDDWTHLAS|12|
|7|L022|EGGDDDWTHLDS|12|
|8|L005|SSGDDDWTHLSS|12|
|9|L004|SAGDDDWTHLSS|12|
|10|L028|SGGEDDWTHLTS|12|

详细证据及逐条理由：[模块二报告](module2/REPORT.md)、[参考比较](module2/reference_comparison.csv)。[文献亲和力](https://pmc.ncbi.nlm.nih.gov/articles/PMC4990988/)3.2/10.9 μM对应20肽WT/L341V构建；这些测量属于各自的文献构建。L341V是弱结合参考。

## 模块三：完整融合肽Top10

|名次|编号|完整融合序列|长度 aa|
|---|---|---|---|
|1|F01_08|EQTSDNAGKRSPATNQGSGSSSGDDDWTHLSS|32|
|2|F01_02|EQTSDNAGKRSPATNQGSGSSGGDDDWTHLTS|32|
|3|F01_05|SGGDDDWTHLDSGSGSGSGSEQTSDNAGKRSPATNQ|36|
|4|F02_09|NEDSGQETSNQSGSGSSAGDDDWTHLSS|28|
|5|F05_01|SGGDDDWTHLESGSGSGSGSSDEQPGNEDTQSGS|34|
|6|F02_04|TGGDDDWTHLTSGSGSNEDSGQETSNQS|28|
|7|F02_03|SGGDDEWTHLSSGSGSGSGSNEDSGQETSNQS|32|
|8|F10_06|NDRTQESGAKSTGAPQNGSGSSGGDDDWTHLAS|33|
|9|F05_04|TGGDDDWTHLTSGSGSSDEQPGNEDTQSGS|30|
|10|F05_05|SGGDDDWTHLDSGSGSSDEQPGNEDTQSGS|30|

每条的来源模块、连接子、连接方向及位置标注见[完整Top10表](module3/top10.csv)。100条组合序列见[100条完整融合肽](module3/complete_pairs_100.csv)，600种方案见[连接方案表](module3/all_linker_variants.csv)。

## 两次短模拟的实际表现

|名次|融合肽|来源模块|初筛保持比例|复核保持比例|
|---|---|---|---|---|
|1|F01_08|B067 + L005|1.00|0.91|
|2|F01_02|B067 + L012|1.00|1.00|
|3|F01_05|B067 + L014|0.82|1.00|
|4|F02_09|B002 + L004|0.73|0.82|
|5|F05_01|B004 + L015|1.00|0.18|
|6|F02_04|B002 + L021|0.27|0.64|
|7|F02_03|B002 + L010|0.27|0.45|
|8|F10_06|B068 + L011|0.45|0.18|
|9|F05_04|B004 + L021|0.45|0.18|
|10|F05_05|B004 + L014|0.36|1.00|

保持比例表示：11个采样时刻中，结合肽–α-syn和LIR–LC3B两侧各保留初始残基接触至少一半的时刻比例；包含0 ps的基线。1.00表示11个时刻均达到，0.18表示2个时刻达到。两次模拟使用不同随机种子。

前4条在两次短观察中均保持较多双界面接触。F05_01由1.00变为0.18，F05_05由0.36变为1.00，体现了有限轨迹的波动。**最终排名来自初筛后的LLM排序，复核结果用于显示重复性和解释范围。** Top10后段适合作为探索候选，当前计算支持强弱有别。

22条轨迹均具有非空的两侧接触基线，数值有限，终点保留标准主链和Ile/Thr侧链手性。模拟使用固定纤维重原子、可动LC3B及融合肽、300 K、1 fs、AMBER14/OBC2隐式溶剂。10 ps观察的是局部接触变化；长时间稳定性、体内半衰期、真实选择性、细胞递送和自噬降解效果仍需后续证据。

## 排序一致性及API用量

反转输入后，结合肽和LIR的Top10各重合9/10，融合肽重合10/10。Top10表示当前候选池内的相对计算优先级。

本轮有效实验调用9次API，输入154,857 tokens、输出54,341 tokens，合计209,198 tokens。正式运行约71.9分钟。修复过程中已弃置请求的消耗另列于[过程核查说明](AUDIT_NOTES.md)。

## 证据范围与文件

本轮产物为未修饰、单LIR的完整嵌合肽。实际结合亲和力、MSA选择性及降解效率保持待验证。正常单体只检查了8个有限模型；LC3B膜锚定、其他ATG8成员、CPP与递送过程属于后续研究范围。

[详细计算报告](DETAILED_REPORT.md)保留完整方法和证据表；[数据完整性核查](integrity_audit.json)记录最终检查；[过程核查说明](AUDIT_NOTES.md)说明能量文字更正、结构假设和弃置请求消耗。API目录保留本轮实际请求、响应和用量，module3/stability目录保留22条轨迹与逐时刻记录。

来源：[Wallace 2024](https://doi.org/10.1039/D3SC06245G)、[PepGLAD](https://github.com/THUNLP-MT/PepGLAD)、[2ZJD](https://www.rcsb.org/structure/2ZJD)、[人p62](https://www.uniprot.org/uniprotkb/Q13501/entry)、[WT/L341V实验](https://pmc.ncbi.nlm.nih.gov/articles/PMC4990988/)、[9A1A](https://www.rcsb.org/structure/9A1A)。
