"""Shared data, API, and evidence utilities for one complete new experiment."""

import csv
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGETS = ('6XYO', '6XYP', '6XYQ', '8A9L', '6CU7')
MSA = TARGETS[:3]
CONTROLS = TARGETS[3:]
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
CLIENT_DATE = '2026-10-13'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                           for k,v in row.items()} for row in rows])


def write_fasta(path, rows):
    Path(path).write_text(''.join('>{} length={}\n{}\n'.format(row['id'],len(row['sequence']),row['sequence'])
                               for row in rows))


def chemistry(sequence):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    run = longest = 0
    for aa in sequence:
        run = run + 1 if aa in 'IVLFWYM' else 0
        longest = max(longest, run)
    return {'length':len(sequence), 'gravy':round(ProteinAnalysis(sequence).gravy(),3),
            'charge_count_KR_minus_DE':sum(sequence.count(a) for a in 'KR')-sum(sequence.count(a) for a in 'DE'),
            'longest_hydrophobic_run':longest}


def object_schema(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def llm(run, name, template, inputs, schema, output_tokens=10000):
    prompt = (ROOT/'prompts'/template).read_text()
    for key, value in inputs.items():
        prompt = prompt.replace('{{'+key+'}}', json.dumps(value,ensure_ascii=False,separators=(',',':')))
    body = {'model':'gpt-6.1-sol', 'messages':[
                {'role':'system','content':'Use supplied scientific observations to propose and compare peptide hypotheses. Return the required JSON object. Keep scientific rationales concise.'},
                {'role':'user','content':prompt}],
            'reasoning_effort':'low','max_completion_tokens':24000,
            'response_format':{'type':'json_schema','json_schema':{'name':name,'strict':True,'schema':schema}}}
    folder = Path(run)/'api'
    folder.mkdir(exist_ok=True)
    write_json(folder/(name+'_request.json'),body)
    request = urllib.request.Request('https://api.chatanywhere.tech/v1/chat/completions',
              data=json.dumps(body).encode(), headers={'Content-Type':'application/json',
              'Authorization':'Bearer '+os.environ['SYNPep_API_KEY']})
    started=time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        result=json.load(response)
    write_json(folder/(name+'_response.json'),result)
    usage={'call':name,'requested_model':body['model'],'returned_model':result['model'],
           'usage':result['usage'],'elapsed_seconds':round(time.monotonic()-started,2)}
    with (folder/'usage.jsonl').open('a') as handle:
        handle.write(json.dumps(usage)+'\n')
    print('LLM_COMPLETED',name,usage['usage'],flush=True)
    return json.loads(result['choices'][0]['message']['content'])


POSE_FIELDS = ('repeat','geometry_eligible','rama_outliers','chirality_flips','beta_stereo_flips','self_severe_clashes','cn_max_deviation_a',
               'omega_max_deviation_deg','severe_clashes','contacts_per_residue',
               'polar_proximity_per_residue','opposite_charge_pairs_per_residue',
               'backbone_reconstruction_rmsd_a','core_two_pocket','core_rmsd_a','mm_gb_binding_proxy_kcal_mol',
               'N_end_open_fraction','C_end_open_fraction','N_end_clearance_a','C_end_clearance_a',
               'local_adjustment_displacement_a','observed_residue_contacts')


def evidence_cards(candidates, poses, targets):
    result=[]
    for candidate in candidates:
        grouped={t:[p for p in poses if p['id']==candidate['id'] and p['target']==t] for t in targets}
        counts={t:sum(p['geometry_eligible'] for p in values) for t,values in grouped.items()}
        card=dict(candidate)
        card.update(chemistry(candidate['sequence']))
        card['pose_columns']=list(POSE_FIELDS)
        def observed_contacts(pose):
            merged={}
            for position,chain,residue,aa in pose['observed_residue_contacts']:
                merged.setdefault((position,residue,aa),set()).add(chain)
            return [[position,residue,aa,''.join(sorted(chains))] for (position,residue,aa),chains in sorted(merged.items())][:16]
        card['poses']={t:[[observed_contacts(p) if field=='observed_residue_contacts' else p.get(field) for field in POSE_FIELDS]
                            for p in values if p['geometry_eligible']] for t,values in grouped.items()}
        card['sample_counts']={t:len(values) for t,values in grouped.items()}
        card['invalid_pose_columns']=['repeat','rmsd_a','rama_outliers','alpha_flips','beta_flips','self_clashes','interchain_clashes']
        card['invalid_pose_summaries']={t:[[p['repeat'],p['backbone_reconstruction_rmsd_a'],p['rama_outliers'],p['chirality_flips'],
            p['beta_stereo_flips'],p['self_severe_clashes'],p['severe_clashes']] for p in values if not p['geometry_eligible']] for t,values in grouped.items()}
        card['eligible_counts']=counts
        card['uncomputed_targets']=[t for t,values in grouped.items() if len(values)==0]
        result.append(card)
    return result


METRICS = {
 'geometry_eligible':'Canonical stereocenters, continuous planar trans backbone, allowed torsions, zero nonlocal self overlaps and interchain heavy-atom distances below 2 A, and C-alpha displacement from the original proposal at most 3 A. This is a stated computational gate.',
 'rama_outliers':'Internal phi/psi outliers under residue-specific CCTBX reference probability tables; termini are excluded.',
 'chirality_flips':'Standard non-Gly residue main-chain handedness opposite to declared L geometry.',
 'beta_stereo_flips':'Ile/Thr side-chain stereocenter handedness opposite to canonical experimental 2ZJD reference residues.',
 'self_severe_clashes':'Peptide heavy-atom pairs separated by less than 2 A across residues at least two positions apart.',
 'cn_max_deviation_a':'Maximum adjacent peptide C-N bond-length deviation from 1.33 A.',
 'omega_max_deviation_deg':'Maximum departure of peptide omega from trans planarity in degrees.',
 'severe_clashes':'Interchain heavy-atom pairs separated by less than 2 A; this is a severe-overlap screen, not a complete experimental validation.',
 'contacts_per_residue':'Distinct peptide/receptor residue pairs within 4.5 A divided by peptide length.',
 'polar_proximity_per_residue':'N/O atom proximities below 3.5 A divided by peptide length; a proximity proxy.',
 'opposite_charge_pairs_per_residue':'D/E versus K/R charged-atom proximities below 4.5 A divided by peptide length.',
 'backbone_reconstruction_rmsd_a':'Peptide C-alpha displacement between raw pose proposal and valid continuous-chain reconstruction.',
 'core_two_pocket':'Both W/L side-chain centroids placed within 3 A of mapped experimental pocket-anchor side-chain centroids.',
 'core_rmsd_a':'WTHL core C-alpha RMSD from the experimental LC3B-bound motif after target alignment.',
 'mm_gb_binding_proxy_kcal_mol':'AMBER/OBC2 complex potential minus receptor and peptide potentials at the same bound coordinates, divided by 4.184. A conditional MM/GBSA compatibility proxy including the OBC2 implicit ACE surface-area term and omitting conformational entropy; lower is more favorable in this approximation. Template-derived pocket placement is prior information.',
 'N_end_open_fraction':'Fraction of 48 directions at 4 A from the N-terminal atom with receptor clearance >2.5 A; a local attachment-space proxy.',
 'C_end_open_fraction':'Same attachment-space proxy at the C terminus.',
 'N_end_clearance_a':'Receptor clearance 3 A along the N-terminal outward bond direction.',
 'C_end_clearance_a':'Receptor clearance 3 A along the C-terminal outward bond direction.',
 'local_adjustment_displacement_a':'RMS heavy-atom displacement during one 20-step bounded review adjustment; this is limited refinement, not independent evidence.',
 'observed_residue_contacts':'Ranking rows: [peptide position, target residue, target amino acid, contacting chain letters] within 4.5 A. Equivalent-rung contacts are aggregated, at most 16 rows are shown for qualified poses; full atom-derived lists are in pose outputs. Invalid samples show reasons rather than uninterpretable contact maps.',
 'scope':'Scores and ranks describe computational compatibility. Measured affinity, cellular degradation, DC50 and in-vivo half-life remain unknown.'}


def rank_candidates(run, module, candidates, poses, targets, dossier):
    dossier={key:value for key,value in dossier.items() if key not in ('structure_sources','pocket','target_pdb','capped_pdb','core_reference_ca','core_reference_sidechain_centroids')}
    cards=evidence_cards(candidates,poses,targets)
    lookup={c['id']:c for c in candidates}
    fields={'candidate_id':{'type':'string','enum':list(lookup)},
            'support':{'type':'string'},'tradeoffs':{'type':'string'},'uncertainty':{'type':'string'}}
    schema=object_schema({'ranked_candidates':{'type':'array','items':object_schema(fields),'minItems':10,'maxItems':10},
                          'decision_summary':{'type':'string'},'feedback_for_next_round':{'type':'string'}})
    inputs={
      'BINDER_RESEARCH_DOSSIER_JSON':dossier,'LIR_REFERENCE_DOSSIER_JSON':dossier,
      'METRIC_DEFINITIONS_JSON':METRICS,'CURRENT_EXPERIMENT_BINDER_EVIDENCE_JSON':cards,
      'CURRENT_EXPERIMENT_LIR_EVIDENCE_JSON':cards}
    template='binder_rank.txt' if module=='module1' else 'lir_rank.txt'
    if len(candidates)<10:
        raise ValueError('{} has {} candidates with interpretable target evidence; ten supported candidates are required'.format(module,len(candidates)))
    answer=llm(run,module+'_ranking_order1',template,inputs,schema,6500)
    inputs['CURRENT_EXPERIMENT_BINDER_EVIDENCE_JSON']=list(reversed(cards))
    inputs['CURRENT_EXPERIMENT_LIR_EVIDENCE_JSON']=list(reversed(cards))
    check=llm(run,module+'_ranking_order2',template,inputs,schema,6500)
    ids1=[r['candidate_id'] for r in answer['ranked_candidates']];ids2=[r['candidate_id'] for r in check['ranked_candidates']]
    agreement={'primary_order':ids1,'reversed_input_order':ids2,'top10_overlap':len(set(ids1)&set(ids2)),
        'interpretation':'Input-order sensitivity audit on the same candidate pool; rank remains a computational priority.'}
    write_json(Path(run)/module/'ranking_order_agreement.json',agreement)
    write_json(Path(run)/module/'llm_ranking_order2.json',check)
    selected=[]
    for rank,item in enumerate(answer['ranked_candidates'],1):
        row=dict(lookup[item['candidate_id']])
        row.update(item)
        row.update(chemistry(row['sequence']))
        row['rank']=rank
        selected.append(row)
    write_json(Path(run)/module/'llm_ranking.json',answer)
    write_jsonl(Path(run)/module/'top10.jsonl',selected)
    write_csv(Path(run)/module/'top10.csv',selected)
    write_fasta(Path(run)/module/'top10.fasta',selected)
    return selected
