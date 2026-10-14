# MSA-SynPep 三模块候选设计

模块一：LLM生成120条α-syn结合肽，PepGLAD五构象初筛，先复核20条，合格候选不足10条时扩至最多40条；入围序列计算8个有限单体对照，LLM输出Top10。
模块二：LLM生成30条未修饰p62 LIR侧翼变体，实验模板集合、天然及弱结合参考评价，LLM Top10。
模块三：10×10=100组，比较600种连接长度/方向方案，最多20组短模拟筛查，LLM Top10与独立短轨迹复核。

## 本轮实验结果

本轮于2026-10-14完成，正式运行约72分钟。当前仓库保存`results/bounded_20261013/`这一轮完整结果。

- [完整实验报告](results/bounded_20261013/REPORT.md)与[详细计算报告](results/bounded_20261013/DETAILED_REPORT.md)。
- [α-syn结合肽Top10](results/bounded_20261013/module1/top10.csv)及[FASTA序列](results/bounded_20261013/module1/top10.fasta)。
- [LIR Top10](results/bounded_20261013/module2/top10.csv)及[FASTA序列](results/bounded_20261013/module2/top10.fasta)。
- [100条完整融合肽](results/bounded_20261013/module3/complete_pairs_100.csv)与[600种连接方案](results/bounded_20261013/module3/all_linker_variants.csv)。
- [最终融合肽Top10](results/bounded_20261013/module3/top10.csv)、[FASTA序列](results/bounded_20261013/module3/top10.fasta)及[两次短模拟结果](results/bounded_20261013/module3/top10_stability_summary.csv)。
- [汇报下载包](results/bounded_20261013/presentation_package.zip)、[本轮完成状态](results/bounded_20261013/status.json)及[过程核查说明](results/bounded_20261013/AUDIT_NOTES.md)。

本轮生成120条结合肽和30条LIR变体，形成两组Top10与100对融合肽。12条完整模型完成10 ps筛查，最终10条各完成不同随机种子的10 ps复核。前4条在两次短观察中较一致，后段候选存在较大波动。序列文件均标注长度和完整融合位置。

## 完整英文提示词

- 模块一：[结合肽生成](prompts/binder_generate.txt)、[结合肽排序](prompts/binder_rank.txt)。
- 模块二：[LIR生成](prompts/lir_generate.txt)、[LIR排序](prompts/lir_rank.txt)。
- 模块三：[完整融合肽排序](prompts/fusion_rank.txt)。

生物学设计原则写入提示词；坐标计算得到的真实观察作为紧凑证据输入。详细说明见[提示词设计](docs/PROMPT_DESIGN.md)。实际API请求、响应和用量保存在本轮结果的`api/`目录，方法说明更正记录在过程核查文件中。

## 运行

服务器路径为/media/yjz/htr/synuclein，Python3.8虚拟环境和两块RTX3090。凭据通过SYNPep_API_KEY传入。

    .venv/bin/python pipeline.py --name bounded_20261013

入口清空本项目results，候选、处理结构、姿势、API记录与排名从头产生。原始公开结构与预训练模型作为固定研究输入。

每次新版本完整重跑的约定见[运行规则](docs/RUN_POLICY.md)。本次GitHub同步上传已完成实验的原始产物。

## 计算安排

初筛每条结合肽在五纤维采样一次。基础几何处理是一次30步Cα扭转拟合；受体位置搜索仅用于入围池的一次20步有限调整。侧链最小化最多150步。GPU完成神经推理、几何、相容性与动力学。LLM三个排名均审查反转输入后的Top10交集。

结构信息包括三层/五层中央SASA比较、侧链方向/电荷/疏水描述和跨链口袋；9A1A八模型是有限单体对照。单体与其他生理状态的整体保护仍需验证。LIR结构模板先验、文献20肽测量及新12肽变体保持各自的证据范围。

## 输出

results/本次运行/module1与module2分别有top10.csv、top10.fasta、top10_evidence.csv、REPORT.md；module3有100组完整序列、600种方案、筛查及复核轨迹、完整融合肽Top10。根目录REPORT.md和presentation_package.zip用于独立汇报。完整英文提示词位于prompts，本次实际请求/响应在api。

Top10表示计算优先级。实测亲和力、选择性、递送和降解效果待验证。
