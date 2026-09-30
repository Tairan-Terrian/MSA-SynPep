# MSA-SynPep Phase 1 计算流程

本项目用公开的 α-synuclein 纤维结构建立构象对照，选择 MSA 纤维表位，用 PepGLAD 预训练模型生成 10–25 aa 短肽，并用 GPU 能量最小化与构象姿态转移指标排序。`results/` 保存全部实验输出。

## 目录

- `data/pdb/`：RCSB 原始坐标，包括 MSA 6XYO/6XYP/6XYQ、PD/PDD/DLB 8A9L、体外 α-syn 纤维 6CU7、tau 8Q2L、Aβ 6SHS。
- `vendor/PepGLAD/`：上游模型源码及官方 `codesign.ckpt` 权重。
- `prepare.py`：结构清单、三种 MSA 构象共同暴露面积、局部折叠差异和表位定义。
- `generate.py`：PepGLAD GPU 批量生成肽序列与全原子构象。
- `minimize.py`：OpenMM OpenCL GPU 优化复合物，固定纤维骨架并允许肽调整。
- `score.py`：MSA 三类构象和两类同蛋白对照的姿态转移筛选、几何质量及候选分层。
- `results/REPORT.md`：研究背景、参数、结果、解释边界和湿实验衔接。
- `results/pilot_typeI_only/`：初次只按 Type I 表面选择表位的完整归档；后续检查发现残基 74 在 Type II 中被遮挡，最终表位改为残基 50。

## 运行环境

服务器 Python 3.8.5、PyTorch 1.13.0+cu117、NumPy 1.23.5、SciPy 1.10.1、Biopython 1.80、FreeSASA 2.2.1、torch-scatter 2.1.2、OpenMM 8.1.1、PDBFixer 1.9.0。OpenMM 经 NVIDIA OpenCL 设备运行；PepGLAD 经 CUDA 运行。模型使用上游 [PepGLAD](https://github.com/THUNLP-MT/PepGLAD) 发布的权重。

仓库包含模型权重、输入结构和实验结果。`.venv/` 是服务器本地的 Python 运行环境，需按上述版本在目标机器上重新创建；`__pycache__/` 是 Python 运行时缓存。上游 PepGLAD 的源码、许可证及权重保存在 `vendor/PepGLAD/`。

## 复现命令

在 `/media/yjz/htr/synuclein` 下运行：

```bash
.venv/bin/python prepare.py
.venv/bin/python generate.py --site site_1 --samples 30 --gpu 0 --batch-size 2
.venv/bin/python generate.py --site site_2 --samples 30 --gpu 1 --batch-size 2
.venv/bin/python generate.py --site site_3 --samples 30 --gpu 0 --batch-size 2
.venv/bin/python generate.py --site site_4 --samples 30 --gpu 1 --batch-size 2
.venv/bin/python minimize.py --gpu 0 --sites site_1 site_3
.venv/bin/python minimize.py --gpu 1 --sites site_2 site_4
.venv/bin/python score.py
```

生成时每个位点使用种子 `20260930`。四个位点各生成 30 条，总计 120 条。最小化采用 OpenMM CHARMM36、非周期 1.2 nm 截断、蛋白骨架约束 1000 kJ mol⁻¹ nm⁻²、肽约束 20 kJ mol⁻¹ nm⁻²、最多 300 次迭代。表位筛选、结合姿态转移和候选排序采用本项目脚本给出的确定规则。

## 结果说明

`candidate_scores.csv` 提供每条肽的接触、冲突、局部对齐 RMSD、构象偏好代理指标、化学性质和质量等级。`priority_candidates.csv` 与 `.fasta` 汇集三种 MSA 结构兼容性得分均为正的候选；`shortlist_24.csv` 与 `.fasta` 是包含探索候选的 24 条实验筛选面板。`experimental_binding_matrix_template.csv` 是后续湿实验录入模板；每个结合信号字段由实验测量填入。计算代理指标描述结构兼容性，真实 MSA 选择性由结合实验判定。
