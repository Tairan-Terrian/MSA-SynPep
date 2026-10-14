"""Fresh three-module experiment with bounded computation and explicit evidence."""
import argparse,shutil,subprocess,sys,time,os
from concurrent.futures import ThreadPoolExecutor
for variable,value in (('OMP_NUM_THREADS','1'),('OPENBLAS_NUM_THREADS','1'),('MKL_NUM_THREADS','1'),('OPENMM_CPU_THREADS','2')):
    os.environ[variable]=value
from core import ROOT,write_json,read_jsonl,write_jsonl,write_csv
import targets,binder,lir,fusion

def parallel(run,script,arguments,label):
    handles=[];processes=[]
    for gpu in (0,1):
        handle=(run/'logs'/('{}_gpu{}.log'.format(label,gpu))).open('w');handles.append(handle)
        processes.append(subprocess.Popen([sys.executable,str(ROOT/script),'--run',str(run),'--gpu',str(gpu)]+arguments,
            cwd=str(ROOT),stdout=handle,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for handle in handles:handle.close()
    if any(codes):raise RuntimeError('{} workers exited {}; see logs'.format(label,codes))

def collect(run,module,pattern,destination):
    rows=[]
    for gpu in (0,1):rows.extend(read_jsonl(run/module/pattern.format(gpu)))
    write_jsonl(run/module/destination,rows);return rows

def main(args):
    started=time.monotonic();shutil.rmtree(ROOT/'results')
    run=ROOT/'results'/args.name
    for folder in ('logs','module1','module2','module3','api'):(run/folder).mkdir(parents=True,exist_ok=True)
    config={'version':'three_modules_bounded_20261013','run_name':args.name,'date':'2026-10-13',
        'binder_candidates':120,'initial_fibril_poses':600,'binder_review_pool':20,'binder_max_review_pool':40,'review_poses':200,
        'monomer_reference':'9A1A','monomer_models':8,'LIR_generated_candidates':30,'LIR_reference_constructs':3,
        'LIR_template_ensemble_size':3,'module1_top':10,'module2_top':10,'fusion_pairs':100,'schemes_per_pair':6,
        'linker_conformations_per_scheme_target':8,'screen_md_ps':10.,'final_md_ps':10.,'final_md_repeats':1,
        'fusion_screen_pool_max':20,'GPUs':[0,1],'API_model':'gpt-6.1-sol','API_completion_budget':24000,'pose_prediction_batch_size':4,
        'geometry_fit_steps':30,'geometry_fit_starts':4,'local_position_adjustment_steps':20,
        'local_position_adjustment_scope':'review pool only, bounded translation and rotation',
        'hydrogen_placement':'OpenMM default geometric and steric preparation before AMBER energy minimization','geometry_proposal_CA_RMSD_gate_a':3.,'sidechain_minimization_max_iterations':150,
        'model_inference':'CUDA','force_field_compute':'NVIDIA OpenCL','ProteinMPNN':'removed',
        'outputs':'All candidates, observations and ranking responses generated in the empty current results directory.'}
    write_json(run/'configuration.json',config)
    def status(stage,**values):
        write_json(run/'status.json',dict(stage=stage,completed=stage=='complete',**values));print('STAGE',stage,flush=True)
    status('preparing');targets.prepare(run)
    def generate_binders():
        result=[]
        for number in (1,2):result.extend(binder.generate(run,number))
        write_jsonl(run/'module1/all_candidates.jsonl',result);return result
    status('generating_sequences')
    with ThreadPoolExecutor(max_workers=2) as pool:
        bfuture=pool.submit(generate_binders);lfuture=pool.submit(lir.generate,run)
        bcandidates=bfuture.result();lcandidates,lcontrols=lfuture.result()
    status('initial_structure_screen',binder_candidates=len(bcandidates),LIR_candidates=len(lcandidates))
    with ThreadPoolExecutor(max_workers=2) as pool:
        bf=pool.submit(parallel,run,'structure.py',['--module','module1','--phase','initial','--candidates',str(run/'module1/all_candidates.jsonl')],'binder_initial')
        lf=pool.submit(parallel,run,'structure.py',['--module','module2','--phase','initial','--candidates',str(run/'module2/all_sequences.jsonl')],'lir_template')
        bf.result();lf.result()
    initial=collect(run,'module1','initial_gpu{}.jsonl','initial_poses.jsonl')
    lposes=collect(run,'module2','initial_gpu{}.jsonl','all_poses.jsonl')
    review=binder.review_pool(run,bcandidates,initial)
    status('focused_binder_review',initial_poses=len(initial),review_pool=len(review))
    parallel(run,'structure.py',['--module','module1','--phase','review','--candidates',str(run/'module1/review_candidates.jsonl')],'binder_review')
    rposes=collect(run,'module1','review_gpu{}.jsonl','review_poses.jsonl')
    supported={p['id'] for p in rposes if p['target'] in ('6XYO','6XYP','6XYQ') and p['geometry_eligible']}
    if len(supported)<10:
        write_jsonl(run/'module1/review_poses_part1.jsonl',rposes)
        original={r['id'] for r in review};expanded=binder.review_pool(run,bcandidates,initial,limit=40)
        extra=[r for r in expanded if r['id'] not in original]
        write_jsonl(run/'module1/extra_review_candidates.jsonl',extra)
        status('expanded_binder_review',supported_sequences=len(supported),additional_candidates=len(extra))
        parallel(run,'structure.py',['--module','module1','--phase','review','--candidates',str(run/'module1/extra_review_candidates.jsonl')],'binder_review_extra')
        rposes+=collect(run,'module1','review_gpu{}.jsonl','review_poses_part2.jsonl')
        write_jsonl(run/'module1/review_poses.jsonl',rposes);review=expanded
    parallel(run,'structure.py',['--module','module1','--phase','monomer','--candidates',str(run/'module1/review_candidates.jsonl')],'monomer_review')
    mposes=collect(run,'module1','monomer_gpu{}.jsonl','monomer_poses.jsonl')
    reviewed={r['id'] for r in review};bposes=[p for p in initial if p['id'] not in reviewed]+rposes+mposes
    write_jsonl(run/'module1/all_poses.jsonl',bposes)
    status('module_ranking')
    with ThreadPoolExecutor(max_workers=2) as pool:
        bf=pool.submit(binder.rank,run,review,bposes);lf=pool.submit(lir.rank,run,lcandidates,lcontrols,lposes)
        bf.result();lf.result()
    status('two_top10_complete')
    fusion.pairs(run);parallel(run,'fusion.py',['--samples','8'],'fusion_assembly')
    rows=collect(run,'module3','pairs_gpu{}.jsonl','complete_pairs_100.jsonl')
    fields=('id','binder_id','lir_id','binder_rank','lir_rank','binder_sequence','lir_sequence','linker_sequence',
        'module_order','sequence','total_length','binder_start','binder_end','linker_start','linker_end','lir_start','lir_end','joint_supported_MSA_targets')
    write_csv(run/'module3/complete_pairs_100.csv',[{k:r[k] for k in fields} for r in rows])
    variants=[dict(v,pair_id=r['id']) for r in rows for v in r['linker_variants']]
    write_jsonl(run/'module3/all_linker_variants.jsonl',variants);write_csv(run/'module3/all_linker_variants.csv',variants)
    screen_rows=fusion.screen_pool(run,rows);status('fusion_screen_dynamics',pairs=len(rows),review_pool=len(screen_rows))
    parallel(run,'stability.py',['--rows',str(run/'module3/screen_candidates.jsonl'),'--duration-ps','10','--repeats','1','--phase','screen'],'screen_dynamics')
    screen=collect(run,'module3','screen_gpu{}.jsonl','screen_stability.jsonl');write_csv(run/'module3/screen_stability.csv',screen)
    status('fusion_ranking');fusion.rank(run,screen_rows,screen)
    status('top10_dynamics_recheck')
    parallel(run,'stability.py',['--rows',str(run/'module3/top10.jsonl'),'--duration-ps','10','--repeats','1','--phase','final'],'final_dynamics')
    final=collect(run,'module3','final_gpu{}.jsonl','final_stability.jsonl');write_csv(run/'module3/final_stability.csv',final)
    from report import generate
    generate(run)
    status('complete',elapsed_hours=round((time.monotonic()-started)/3600,3),binder_top=10,LIR_top=10,complete_pairs=100,fusion_top=10)
    print('EXPERIMENT_COMPLETE',run,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='bounded_20261013');main(parser.parse_args())
