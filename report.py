"""Actual-count Chinese report and portable presentation package."""
import zipfile
from pathlib import Path
import numpy as np
from core import ROOT,read_json,read_jsonl,write_json,write_csv

def table(rows,columns):
    lines=['|'+'|'.join(label for key,label in columns)+'|','|'+'|'.join('---' for _ in columns)+'|']
    for row in rows:
        lines.append('|'+'|'.join(('未知' if row.get(key) is None else str(row[key])).replace('|','/') for key,label in columns)+'|')
    return '\n'.join(lines)

def generate(run):
    run=Path(run);config=read_json(run/'configuration.json')
    b=read_jsonl(run/'module1/top10.jsonl');l=read_jsonl(run/'module2/top10.jsonl');f=read_jsonl(run/'module3/top10.jsonl')
    bc=read_jsonl(run/'module1/all_candidates.jsonl');lc=read_jsonl(run/'module2/candidates.jsonl')
    bp=read_jsonl(run/'module1/all_poses.jsonl');lp=read_jsonl(run/'module2/all_poses.jsonl');pairs=read_jsonl(run/'module3/complete_pairs_100.jsonl')
    screen=read_jsonl(run/'module3/screen_stability.jsonl');final=read_jsonl(run/'module3/final_stability.jsonl')
    calls=read_jsonl(run/'api/usage.jsonl')
    totals={'calls':len(calls),'input_tokens':sum(c['usage']['prompt_tokens'] for c in calls),
        'output_tokens':sum(c['usage']['completion_tokens'] for c in calls),'total_tokens':sum(c['usage']['total_tokens'] for c in calls)}
    write_json(run/'api/usage_summary.json',totals)
    agreements={m:read_json(run/m/'ranking_order_agreement.json') for m in ('module1','module2','module3')}
    bsummary=[]
    for row in b:
        p=[x for x in bp if x['id']==row['id']]
        counts={t:sum(x['geometry_eligible'] for x in p if x['target']==t) for t in ('6XYO','6XYP','6XYQ','8A9L','6CU7')}
        monomers=[x for x in p if x['target'].startswith('9A1A_')]
        bsummary.append(dict(row,**counts,monomer_eligible=sum(x['geometry_eligible'] for x in monomers),monomer_samples=len(monomers)))
    lsummary=[]
    for row in l:
        p=[x for x in lp if x['id']==row['id']]
        lsummary.append(dict(row,valid_two_pocket=sum(x['geometry_eligible'] and x['core_two_pocket'] for x in p),samples=len(p)))
    write_csv(run/'module1/top10_evidence.csv',bsummary);write_csv(run/'module2/top10_evidence.csv',lsummary)
    fsummary=[]
    for row in f:
        s=next(x for x in screen if x['id']==row['id']);r=next(x for x in final if x['id']==row['id'])
        fsummary.append(dict(row,screen_dual_retention=s['simultaneous_retention_fraction'],recheck_dual_retention=r['simultaneous_retention_fraction'],
            recheck_alpha_flips=r['final_peptide_geometry']['chirality_flips'],recheck_beta_flips=r['final_peptide_geometry']['beta_stereo_flips']))
    fcolumns=[('rank','名次'),('id','编号'),('binder_id','结合肽'),('lir_id','LIR'),('module_order','顺序'),
        ('linker_sequence','连接子'),('sequence','完整融合序列'),('total_length','长度'),('screen_dual_retention','10 ps双界面保持'),('recheck_dual_retention','独立复核保持')]
    write_csv(run/'module3/top10_stability_summary.csv',[{k:r[k] for k,label in fcolumns} for r in fsummary])
    controls=[]
    for identifier in ('p62_natural_12','p62_WT_20','p62_L341V_20'):
        p=[x for x in lp if x['id']==identifier]
        energy=[x['mm_gb_binding_proxy_kcal_mol'] for x in p if x['geometry_eligible'] and x['core_two_pocket'] and x['mm_gb_binding_proxy_kcal_mol'] is not None]
        controls.append({'id':identifier,'length':len(p[0]['sequence']),'qualified_two_pocket':sum(x['geometry_eligible'] and x['core_two_pocket'] for x in p),
            'mean_conditional_MMGB_kcal_mol':round(float(np.mean(energy)),3) if energy else None})
    write_csv(run/'module2/reference_comparison.csv',controls)
    summary={'binder_candidates':len(bc),'binder_initial_poses':len(read_jsonl(run/'module1/initial_poses.jsonl')),
        'binder_current_poses':len(bp),'binder_top10':len(b),'LIR_candidates':len(lc),'LIR_poses_including_controls':len(lp),'LIR_top10':len(l),
        'fusion_pairs':len(pairs),'linker_schemes':sum(len(r['linker_variants']) for r in pairs),'screen_trajectories':len(screen),
        'fusion_top10':len(f),'recheck_trajectories':len(final),'order_agreement':agreements,'api_usage':totals}
    write_json(run/'experiment_summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=run/'figures';directory.mkdir(exist_ok=True)
    matrix=np.zeros((10,10));md=np.full((10,10),np.nan);lookup={r['id']:r for r in pairs}
    for r in pairs:matrix[r['binder_rank']-1,r['lir_rank']-1]=len(r['joint_supported_MSA_targets'])
    for r in screen:
        pair=lookup[r['id']];md[pair['binder_rank']-1,pair['lir_rank']-1]=r['simultaneous_retention_fraction']
    for name,values,title,maximum,label in [('joint_geometry',matrix,'MSA conformers with joint-qualified geometry',3,'Supported conformers'),
        ('screen_retention',md,'10 ps dual-interface retention (gray: uncomputed)',1,'Fraction of sampled frames')]:
        fig,ax=plt.subplots(figsize=(7,6));cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dddddd')
        im=ax.imshow(values,vmin=0,vmax=maximum,cmap=cmap)
        ax.set(xticks=range(10),xticklabels=[r['id'] for r in l],yticks=range(10),yticklabels=[r['id'] for r in b],xlabel='LIR',ylabel='Binder',title=title)
        ax.tick_params(axis='x',rotation=45)
        for r in f:ax.plot(r['lir_rank']-1,r['binder_rank']-1,'s',mfc='none',mec='white',ms=18,mew=1.5)
        fig.colorbar(im,ax=ax,label=label);fig.tight_layout();fig.savefig(directory/(name+'.png'),dpi=180);plt.close(fig)
    details=lambda rows:'\n\n'.join('**{} {}**\n\n支持：{}\n\n取舍：{}\n\n未知项：{}'.format(r['rank'],r['id'],r['support'],r['tradeoffs'],r['uncertainty']) for r in rows)
    btable=table(bsummary,[('rank','名次'),('id','编号'),('sequence','完整序列'),('length','长度'),('6XYO','6XYO合格姿势'),('6XYP','6XYP'),('6XYQ','6XYQ'),('8A9L','8A9L'),('6CU7','6CU7'),('monomer_eligible','合格单体姿势')])
    ltable=table(lsummary,[('rank','名次'),('id','编号'),('sequence','完整序列'),('length','长度'),('valid_two_pocket','合格双口袋姿势'),('samples','模板样本数')])
    ftable=table(fsummary,fcolumns)
    for module,title,body,rows in [('module1','α-syn结合肽',btable,b),('module2','LIR',ltable,l),('module3','完整融合肽',ftable,f)]:
        (run/module/'REPORT.md').write_text('# '+title+'实验结果\n\n'+body+'\n\n'+details(rows)+'\n')
    contexts=read_json(run/'inputs/surface_contexts.json');selection=read_json(run/'module3/screen_selection.json')
    source='[Wallace 2024](https://doi.org/10.1039/D3SC06245G)、[PepGLAD](https://github.com/THUNLP-MT/PepGLAD)、[LC3B–p62 2ZJD](https://www.rcsb.org/structure/2ZJD)、[p62序列](https://www.uniprot.org/uniprotkb/Q13501/entry)、[WT/L341V实验](https://pmc.ncbi.nlm.nih.gov/articles/PMC4990988/)、[单体集合9A1A](https://www.rcsb.org/structure/9A1A)。'
    methods="""## 方法与生物学范围

输入为三种MSA、两个同蛋白纤维对照及9A1A八个有限单体模型。表面计算使用组装体遮挡，记录三层/五层SASA差异。骨架形状和共同暴露提出四个区域；侧链方向、电荷组及疏水排列作为结构描述输入LLM。系统性抗体表位、竞争结合及当前区域突变标签列为资料缺口。

LLM两批各生成60条10–25 aa结合肽。PepGLAD在双GPU上按批量4条固定序列预测，初筛五纤维各一个姿势。基础几何处理在四个低成本骨架初始化中选择Cα匹配最好的一条，进行30步仅对Cα的扭转角拟合。记录原始坐标、标准化位移和资格。初始20条复核本次第一姿势并补第二姿势，合格序列不足10条时将复核池扩到最多40条；符合条件的提案接受一次20步有限位置调整，平移每轴±1 Å，旋转四元数向量分量±0.06。固定骨架无严重重叠时最多150步GPU侧链最小化。采用标准几何加氢后执行AMBER优化。额外位置调整集中到复核池。

资格覆盖Cα/Ile/Thr手性、允许主链角、C–N偏离<0.05 Å、trans平面偏离<10°、重原子<2 Å严重重叠为0，以及标准化Cα位移≤3 Å。这些是本轮计算阈值。姿势失败保留未知信息。末端方向开放度和延伸点净空是连接空间代理。

LIR以人p62 332–343片段SGGDDDWTHLSS为结构设计模板，固定WTHL，侧翼改变1–3位。首轮单LIR且未修饰。LC3B–p62实验复合物提供核心识别先验，三个侧翼模板样本检查相容性。天然12肽及文献20肽WT、L341V单独作为参考。文献3.2/10.9 μM测量属于20肽构建，L341V为弱结合对照。条件MM/GBSA包含OBC2的ACE表面积溶剂化项，省略构象熵；校准方向与缺失值保留。其他家族成员、修饰依赖和膜环境保持未知。

两个Top10形成100组，每组比较4/8/12残基GSGS及两个模块顺序，每个方案在五纤维上采样8个连接构象。完整模型检查两处界面、源模块资格及LC3B与纤维重叠。按跨MSA合格覆盖、可行比例和连接长度选择每组代表方案。合格三元模型表示空间相容性。

最多20组入围融合肽使用共同MSA靶标和同一协议的10 ps局部模拟：固定纤维重原子、LC3B和融合肽可动、AMBER14/OBC2隐式溶剂、300 K、1 fs。最终10组各做一次独立10 ps复核，记录两处接触与同时保持比例。LLM根据实际观察排序；反转同一池输入再排一次，记录Top10交集。
"""
    report='# MSA-SynPep 三模块独立实验报告\n\n运行：'+config['run_name']+'；日期：2026-10-13。全部候选与计算来自本次清空结果后的独立运行。\n\n'
    report+='## 实际完成数量\n\n'+table([{'item':k,'count':v} for k,v in summary.items() if isinstance(v,int)],[('item','项目'),('count','数量')])+'\n\n'
    report+=methods+'\n本次四个区域中心：'+str([r['center_residue'] for r in contexts])+'；融合筛查共同靶标：'+selection['simulation_target']+'。\n\n'
    report+='## 两组Top10与最终Top10\n\n### α-syn结合肽\n\n'+btable+'\n\n### LIR\n\n'+ltable+'\n\n### 完整融合肽\n\n'+ftable+'\n\n'
    report+='## 排序稳定性与API记录\n\n反转输入的Top10交集：'+str({m:a['top10_overlap'] for m,a in agreements.items()})+'。本次API用量：'+str(totals)+'。\n\n'
    report+='![三元几何覆盖](figures/joint_geometry.png)\n\n![有限轨迹观察](figures/screen_retention.png)\n\n'
    report+='## 结果解释\n\nTop10是本次模型与有限采样下的相对计算优先级。单体集合仅覆盖有限状态；模板、固定受体和有限连接采样构成模型先验。短轨迹观察局部接触保持，体内半衰期、细胞降解效果和长时间稳定性保持未知。实际Kd、选择性、递送、胞质释放、自噬通量、DC50和Dmax待验证。本轮序列是未修饰线性L肽，含单LIR，CPP和膜上招募分别作为后续事项。\n\n'
    report+='## 来源与可审查记录\n\n'+source+'\n\n完整英文模板位于汇报包prompts；服务器api目录保存实际请求和响应。各模块REPORT.md提供逐条理由。\n'
    (run/'REPORT.md').write_text(report)
    archive=run/'presentation_package.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for relative in ('REPORT.md','configuration.json','experiment_summary.json','api/usage_summary.json'):
            z.write(run/relative,relative)
        for module in ('module1','module2','module3'):
            for pattern in ('top10*.csv','top10.fasta','REPORT.md','ranking_order_agreement.json','*comparison.csv','*stability.csv','complete_pairs_100.csv','pairs_100.csv'):
                for p in (run/module).glob(pattern):z.write(p,str(p.relative_to(run)))
        for p in directory.glob('*.png'):z.write(p,str(p.relative_to(run)))
        for p in (ROOT/'prompts').glob('*.txt'):z.write(p,'prompts/'+p.name)
        for row in f:
            for target in row['joint_supported_MSA_targets']:
                p=Path(row['structures'][target]);z.write(p,'fusion_structures/'+p.name)
            for r in [x for x in screen+final if x['id']==row['id']]:
                for key in ('initial_pdb','final_pdb'):
                    p=Path(r[key]);z.write(p,'stability/'+row['id']+'/'+p.name)
        z.write(run/'inputs/layer_sensitivity.csv','inputs/layer_sensitivity.csv')
    print('REPORT_AND_PACKAGE_COMPLETE',str(archive),flush=True)
