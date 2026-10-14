# 固定研究参考

- 2ZJD.pdb：RCSB公开LC3B–p62复合物结构，https://www.rcsb.org/structure/2ZJD 。LC3B来自人，p62结构肽来自鼠；本次人序列按核心基序映射。
- SQSTM1_Q13501.fasta：人p62的UniProt参考序列，https://www.uniprot.org/uniprotkb/Q13501/entry 。
- LC3B_Q9GZQ8.fasta：人LC3B的UniProt参考序列，https://www.uniprot.org/uniprotkb/Q9GZQ8/entry 。
- ramachandran_probabilities.npz：由本地CCTBX RamachandranEval导出的固定统计概率表。包含general、glycine、cispro、transpro、prepro、ileval六类，φ/ψ网格为−179°至179°，步长2°。正式实验使用GPU插值；general允许阈值0.0005，其余0.001，采样使用概率≥0.02的区域。本次肽键设置trans，Pro采用transpro。源码依据：https://github.com/cctbx/cctbx_project/blob/master/mmtbx/validation/ramalyze.py 。

这些公开或统计参考作为原始研究输入。处理后的结构、候选、评价和轨迹在每次正式实验中重新计算。
