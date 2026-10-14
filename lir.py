"""Module two: reference-conditioned LIR variants, LC3B evidence, Top10."""
from pathlib import Path
import numpy as np
from core import (ALPHABET,read_json,write_jsonl,write_csv,object_schema,llm,
                  evidence_cards,rank_candidates)

def generate(run):
    folder=Path(run)/'module2';folder.mkdir(exist_ok=True)
    dossier=read_json(Path(run)/'inputs/lir_reference_dossier.json')
    text={'type':'string'}
    item=object_schema({'sequence':{'type':'string','pattern':'^[{}]{{6}}WTHL[{}]{{2}}$'.format(ALPHABET,ALPHABET)},
       'design_group':text,'evidence_ids':{'type':'array','items':text},'design_specification':text,
       'testable_structural_expectation':text,'preferred_attachment_end':{'type':'string','enum':['N','C','either']},
       'attachment_hypothesis':text,'modification_dependence':text,'uncertainty':text})
    schema=object_schema({'batch_plan':text,'reference_baseline':object_schema({'sequence':{'type':'string','enum':[dossier['reference_seed']]},
        'role':{'type':'string','enum':['natural_reference_outside_designed_variants']}}),
        'designed_variants':{'type':'array','minItems':30,'maxItems':30,'items':item}})
    inputs={'LIR_REFERENCE_DOSSIER_JSON':dossier,'LC3B_INTERFACE_CONTEXT_JSON':{'core':'WTHL','pocket':dossier['pocket']},
       'CURRENT_DESIGN_CONFIGURATION_JSON':{'candidate_count':30,'length':12,'core_positions':[7,8,9,10],
                    'mutable_positions':[1,2,3,4,5,6,11,12],'mutations_per_variant':[1,3],'shortlist_size':10},
       'CURRENT_EXPERIMENT_FEEDBACK_JSON':{}}
    answer=llm(run,'lir_generation','lir_generate.txt',inputs,schema,9000)
    candidates=[];seed=dossier['reference_seed']
    for index,row in enumerate(answer['designed_variants'],1):
        mutations=[{'position':i+1,'from':a,'to':b} for i,(a,b) in enumerate(zip(seed,row['sequence'])) if a!=b]
        candidates.append(dict(row,id='L{:03d}'.format(index),length=len(row['sequence']),
                      reference_id='human_p62_332_343',mutation_list=mutations,modifications=[],source='closed_LLM_current_experiment'))
    controls=[{'id':'p62_natural_12','sequence':seed,'length':12,'source':'natural_reference_fragment'}]
    controls.extend(dict(c,length=len(c['sequence']),source='matched_literature_control') for c in dossier['matched_literature_controls'])
    write_jsonl(folder/'candidates.jsonl',candidates)
    write_jsonl(folder/'reference_controls.jsonl',controls)
    write_jsonl(folder/'all_sequences.jsonl',candidates+controls)
    return candidates,controls

def rank(run,candidates,controls,poses):
    dossier=read_json(Path(run)/'inputs/lir_reference_dossier.json')
    cards=evidence_cards(candidates+controls,poses,['LC3B'])
    dossier['current_reference_observations']=[c for c in cards if c['id'] in {r['id'] for r in controls}]
    observed={identifier:[p['mm_gb_binding_proxy_kcal_mol'] for p in poses if p['id']==identifier
               and p['geometry_eligible'] and p['core_two_pocket'] and p['mm_gb_binding_proxy_kcal_mol'] is not None]
           for identifier in ('p62_WT_20','p62_L341V_20')}
    means={identifier:float(np.mean(values)) if values else None for identifier,values in observed.items()}
    dossier['conditional_energy_control_calibration']={'mean_potential_proxies':means,
          'interpretable_sample_counts':{identifier:len(values) for identifier,values in observed.items()},
          'known_affinity_direction_reproduced':means['p62_WT_20']<means['p62_L341V_20'] if all(value is not None for value in means.values()) else None,
          'interpretation':'Use calibration to bound energy-based conclusions; a failed known direction leaves affinity-improvement hypotheses unresolved.'}
    supported={p['id'] for p in poses if p['geometry_eligible'] and p['core_two_pocket']}
    pool=[c for c in candidates if c['id'] in supported]
    write_jsonl(Path(run)/'module2/evidence_cards.jsonl',cards)
    write_csv(Path(run)/'module2/pose_metrics.csv',poses)
    print('LIR_SUPPORTED_POOL',len(pool),'of',len(candidates),flush=True)
    return rank_candidates(run,'module2',pool,poses,['LC3B'],dossier)
