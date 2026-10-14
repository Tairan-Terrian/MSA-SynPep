"""Module one: LLM sequences, target-conditioned pose evidence, Top10 ranking."""
from pathlib import Path
from core import (ALPHABET,read_json,read_jsonl,write_jsonl,write_csv,object_schema,llm,
                  evidence_cards,rank_candidates,MSA,TARGETS,write_json)

def generate(run,round_number=1):
    folder=Path(run)/'module1';folder.mkdir(exist_ok=True)
    contexts=read_json(Path(run)/'inputs/surface_contexts.json')
    sites=[c['site_id'] for c in contexts]
    dossier=read_json(Path(run)/'inputs/binder_dossier.json')
    compact=[]
    for context in contexts:
        conformers={}
        for target in TARGETS:
            row=context['conformers'][target]
            conformers[target]={'chain':row['central_chain'],'center_sasa_a2':row['center_sasa_a2'],
                'three_five_layer_sasa_difference_a2':row['center_layer_sasa_difference_a2'],
                'cross_chain_pocket':row['cross_chain_pocket'],
                'surface_columns':['residue','aa','sasa_a2','charge_group','hydrophobic_group','sidechain_direction_in_local_frame'],
                'surface_rows':[[r['residue'],r['aa'],r['sasa_a2'],r['charge_group'],r['hydrophobic_group'],r['sidechain_direction']] for r in row['surface_chemistry']]}
        compact.append({'site_id':context['site_id'],'evidence_id':context['evidence_id'],'center_residue':context['center_residue'],
            'selection_observation':context['joint_selection_observation'],'conformers':conformers,
            'boundary_note':'The resolved fibril core has missing full-length tails; centers near a resolved boundary carry additional uncertainty.'})
    dossier={key:dossier[key] for key in ('objective','MSA_targets','controls','labels','site_selection','monomer_controls','experimental_references','unavailable_evidence')}
    dossier['geometry_data_scope']='Full coordinates, pockets and mappings remain in this experiment inputs. Generation uses calculated descriptors, not mental reconstruction of atomic coordinates.'
    dossier['assembly_scope']='Middle rung of the longest-resolved protofilament, with paired-protofilament occlusion; local-frame sidechain directions are independent of global PDB axes.'
    text={'type':'string'}
    item=object_schema({'sequence':{'type':'string','pattern':'^[{}]{{10,25}}$'.format(ALPHABET)},
        'site_id':{'type':'string','enum':sites},'evidence_ids':{'type':'array','items':text},
        'design_specification':text,'peptide_anchor_hypotheses':text,
        'expected_MSA_vs_control_difference':text,'testable_structural_expectation':text,
        'design_group':text,'preferred_attachment_end':{'type':'string','enum':['N','C','either']},
        'attachment_hypothesis':text,'uncertainty':text})
    schema=object_schema({'batch_plan':text,'candidates':{'type':'array','minItems':60,'maxItems':60,'items':item}})
    feedback={}
    if round_number==2:
        prior=read_jsonl(folder/'round1_candidates.jsonl')
        feedback={'existing_sequences':[c['sequence'] for c in prior],
                  'state':'Batch two expands sequence diversity; structural evaluation follows both batches.'}
    inputs={'BINDER_RESEARCH_DOSSIER_JSON':dossier,'CONFORMER_SURFACE_COMPARISON_JSON':compact,
       'CURRENT_DESIGN_CONFIGURATION_JSON':{'candidate_count':60,'length_range':[10,25],
                  'distribution':'15 candidates per supplied site where four sites are available; balanced length and recognition hypotheses',
                  'peptide_chemistry':'linear unmodified L amino acids; explicit free peptide termini'},
       'CURRENT_EXPERIMENT_FEEDBACK_JSON':feedback}
    answer=llm(run,'binder_generation_round{}'.format(round_number),'binder_generate.txt',inputs,schema,15000)
    candidates=[]
    for index,row in enumerate(answer['candidates'],1):
        candidates.append(dict(row,id='B{:03d}'.format((round_number-1)*60+index),
                    length=len(row['sequence']),source='closed_LLM_current_experiment',generation_round=round_number))
    write_jsonl(folder/('round{}_candidates.jsonl'.format(round_number)),candidates)
    return candidates

def review_pool(run,candidates,poses,limit=20):
    cards=evidence_cards(candidates,poses,TARGETS);lookup={c['id']:c for c in candidates}
    def priority(card):
        valid={t:[p for p in poses if p['id']==card['id'] and p['target']==t and p['geometry_eligible']] for t in TARGETS}
        coverage=sum(bool(valid[t]) for t in MSA)
        contact=sum(max([p['contacts_per_residue'] for p in valid[t]] or [0]) for t in MSA)
        proposals=[p for p in poses if p['id']==card['id'] and p['target'] in MSA]
        fit_support=sum(p['backbone_reconstruction_rmsd_a']<=3 and p['fixed_geometry_collisions']<=40 for p in proposals)
        fit=sum(min(p['backbone_reconstruction_rmsd_a'],10)+min(p['fixed_geometry_collisions'],100)/20 for p in proposals)
        return (coverage,fit_support,contact,-fit,-card['longest_hydrophobic_run'])
    selected=sorted(cards,key=priority,reverse=True)[:limit]
    pool=[lookup[c['id']] for c in selected]
    write_jsonl(Path(run)/'module1/review_candidates.jsonl',pool)
    write_json(Path(run)/'module1/review_selection.json',{'rule':'Geometry-supported MSA coverage, then near-compatible proposal coverage (C-alpha RMSD <=3 A and <=40 fixed-atom overlaps), valid contacts, proposal fit, hydrophobic-run length. This schedules review; near-compatible proposals remain invalid for final rank until they pass geometry.',
        'selected_ids':[c['id'] for c in pool],'pool_size':len(pool)})
    return pool

def rank(run,candidates,poses):
    dossier=read_json(Path(run)/'inputs/binder_dossier.json')
    target_names=list(TARGETS)+sorted({p['target'] for p in poses if p['target'].startswith('9A1A_')})
    cards=evidence_cards(candidates,poses,target_names)
    supported={c['id'] for c in cards if any(c['eligible_counts'][target]>0 for target in MSA)}
    pool=[c for c in candidates if c['id'] in supported]
    write_jsonl(Path(run)/'module1/evidence_cards.jsonl',cards)
    write_csv(Path(run)/'module1/pose_metrics.csv',poses)
    print('BINDER_SUPPORTED_POOL',len(pool),'of',len(candidates),flush=True)
    dossier['priority']='Prefer all-three-MSA coverage; retain target-specific limitations for partial coverage. Monomer ensemble is a limited off-target panel.'
    return rank_candidates(run,'module1',pool,poses,target_names,dossier)
