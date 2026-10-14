"""Module three: 100 peptide pairs, linker ensembles, complete-chimera evidence."""
import argparse,copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from core import (read_json,read_jsonl,write_json,write_jsonl,write_csv,write_fasta,
                  chemistry,object_schema,llm,TARGETS,MSA,METRICS)
from geometry import (AA,BB,read_structure,save_structure,tensor,angles_from_chain,RamaReference,
                      peptide,rotamers,backbone,align,atom_data,interface_metrics,geometry_metrics)

LINKERS=('GSGS','GSGSGSGS','GSGSGSGSGSGS')
ORDERS=('binder_linker_lir','lir_linker_binder')

def pairs(run):
    binders=read_jsonl(Path(run)/'module1/top10.jsonl');lirs=read_jsonl(Path(run)/'module2/top10.jsonl')
    rows=[]
    for b in binders:
        for l in lirs:
            rows.append({'id':'F{:02d}_{:02d}'.format(b['rank'],l['rank']),
                         'binder_id':b['id'],'lir_id':l['id'],'binder_rank':b['rank'],'lir_rank':l['rank'],
                         'binder_sequence':b['sequence'],'lir_sequence':l['sequence'],
                         'binder_length':len(b['sequence']),'lir_length':len(l['sequence'])})
    write_jsonl(Path(run)/'module3/pairs_100.jsonl',rows);write_csv(Path(run)/'module3/pairs_100.csv',rows)
    return rows

def best_pose(poses,candidate,target):
    rows=[p for p in poses if p['id']==candidate and p['target']==target]
    return sorted(rows,key=lambda p:(not p['geometry_eligible'],not p.get('core_two_pocket',True),
                                    -p['contacts_per_residue'],p['backbone_reconstruction_rmsd_a']))[0]

def layout(row,linker,order):
    b,l=row['binder_sequence'],row['lir_sequence']
    if order=='binder_linker_lir':
        sequence=b+linker+l;bs=0;ls=len(b)+len(linker);ks=len(b)
    else:
        sequence=l+linker+b;ls=0;bs=len(l)+len(linker);ks=len(l)
    return sequence,bs,ls,ks

def atom_template(sequence,phis,psis,sideangles,device):
    chain=peptide(sequence,phis,psis,device,sidechain_angles=sideangles)
    coefficients=[];residue_indices=[];atom_names=[];prototype_atoms=[]
    for i,residue in enumerate(chain):
        n,ca,c=tensor([residue[name].coord for name in BB],device)
        e1=(n-ca)/torch.linalg.vector_norm(n-ca)
        e2=c-ca-torch.dot(c-ca,e1)*e1;e2=e2/torch.linalg.vector_norm(e2)
        e3=torch.cross(e1,e2,dim=-1);basis=torch.stack((e1,e2,e3))
        for atom in residue:
            coefficients.append((tensor(atom.coord,device)-ca)@basis.T)
            residue_indices.append(i);atom_names.append(atom.name);prototype_atoms.append(atom)
    return chain,torch.stack(coefficients),torch.tensor(residue_indices,device=device),atom_names

def all_atom_coordinates(bb,psi,coefficients,indices,names):
    n=bb[:,:,0];ca=bb[:,:,1];c=bb[:,:,2]
    e1=(n-ca)/torch.linalg.vector_norm(n-ca,dim=-1,keepdim=True)
    e2=c-ca-((c-ca)*e1).sum(-1,keepdim=True)*e1;e2=e2/torch.linalg.vector_norm(e2,dim=-1,keepdim=True)
    e3=torch.cross(e1,e2,dim=-1);basis=torch.stack((e1,e2,e3),-2)
    coordinates=ca[:,indices]+torch.einsum('na,bnac->bnc',coefficients,basis[:,indices])
    from geometry import place
    oxygen=place(n,ca,c,1.23,120.5117,psi+180)
    terminal=place(n[:,-1],ca[:,-1],c[:,-1],1.25,120.5,psi[:,-1])
    for j,name in enumerate(names):
        if name=='N':coordinates[:,j]=n[:,indices[j]]
        elif name=='CA':coordinates[:,j]=ca[:,indices[j]]
        elif name=='C':coordinates[:,j]=c[:,indices[j]]
        elif name=='O':coordinates[:,j]=oxygen[:,indices[j]]
        elif name=='OXT':coordinates[:,j]=terminal
    return coordinates

def sample_variant(row,target,binder_pose,lir_pose,linker,order,reference,device,generator,samples):
    sequence,bs,ls,ks=layout(row,linker,order);nb=len(row['binder_sequence']);nl=len(row['lir_sequence'])
    b_structure=read_structure(binder_pose['pdb']);l_structure=read_structure(lir_pose['pdb'])
    b_chain=b_structure[0]['K'];l_chain=l_structure[0]['K']
    bphi,bpsi=angles_from_chain(b_chain,device);lphi,lpsi=angles_from_chain(l_chain,device)
    phi,psi=reference.sample(sequence,samples,generator)
    phi[:,bs:bs+nb]=bphi;psi[:,bs:bs+nb]=bpsi
    phi[:,ls:ls+nl]=lphi;psi[:,ls:ls+nl]=lpsi
    for start,length in ((bs,nb),(ls,nl)):
        proposed_phi,proposed_psi=reference.sample(sequence,samples,generator)
        if start>0:phi[:,start]=proposed_phi[:,start]
        if start+length<len(sequence):psi[:,start+length-1]=proposed_psi[:,start+length-1]
    phi,psi=reference.project(sequence,phi,psi)
    sideangles={bs+i:v for i,v in rotamers(b_chain,device).items()}
    sideangles.update({ls+i:v for i,v in rotamers(l_chain,device).items()})
    prototype,coefficients,indices,names=atom_template(sequence,phi[0],psi[0],sideangles,device)
    bb=backbone(sequence,phi,psi)
    bca=tensor([r['CA'].coord for r in b_chain if r.resname in AA],device)
    rotation,translation=align(bb[:,bs:bs+nb,1],bca)
    bb=(bb.reshape(samples,-1,3)@rotation+translation[:,None,:]).reshape(samples,len(sequence),3,3)
    coordinates=all_atom_coordinates(bb,psi,coefficients,indices,names)
    lca=tensor([r['CA'].coord for r in l_chain if r.resname in AA],device)
    lc3_rotation,lc3_translation=align(lca.expand(samples,-1,-1),bb[:,ls:ls+nl,1])
    lc3_atoms=[a for a in l_structure[0]['R'].get_atoms() if a.element!='H']
    lc3=tensor([a.coord for a in lc3_atoms],device)
    transformed=lc3[None]@lc3_rotation+lc3_translation[:,None,:]
    receptor_chains=[c for c in b_structure[0] if c.id!='K']
    receptor=tensor([a.coord for c in receptor_chains for a in c.get_atoms() if a.element!='H'],device)
    totals=[];penalties=[]
    nonlocal_mask=(indices[:,None]-indices[None,:]).abs()>=2
    for first in range(0,samples,4):
        pep=coordinates[first:first+4];lc=transformed[first:first+4]
        dr=torch.cdist(pep,receptor[None]);dl=torch.cdist(pep,lc)
        protein=torch.cdist(lc,receptor[None]);self_dist=torch.cdist(pep,pep)
        counts=(dr<2).sum((1,2))+(dl<2).sum((1,2))+(protein<2).sum((1,2))+((self_dist<2)&nonlocal_mask).sum((1,2))//2
        penalty=torch.relu(2.35-dr).square().sum((1,2))+torch.relu(2.35-dl).square().sum((1,2))+torch.relu(2.35-protein).square().sum((1,2))
        totals.append(counts);penalties.append(penalty)
    totals=torch.cat(totals);penalties=torch.cat(penalties)
    selected=int((totals.double()*1000+penalties).argmin())
    atoms=list(prototype.get_atoms())
    for atom,point in zip(atoms,coordinates[selected].cpu().numpy()):atom.coord=point
    lc3_chain=copy.deepcopy(l_structure[0]['R']);lc3_atoms_all=list(lc3_chain.get_atoms())
    transformed_all=tensor([a.coord for a in lc3_atoms_all],device)@lc3_rotation[selected]+lc3_translation[selected]
    for atom,point in zip(lc3_atoms_all,transformed_all.cpu().numpy()):atom.coord=point
    result=copy.deepcopy(b_structure);result[0].detach_child('K');result[0].add(prototype);result[0].add(lc3_chain)
    summary={'target':target,'sampled_linker_conformations':samples,'feasible_fraction':round(float((totals==0).double().mean()),3),
             'minimum_combined_severe_clashes':int(totals[selected]),'packing_overlap_penalty':round(float(penalties[selected]),3),
             'binder_input_pose_eligible':binder_pose['geometry_eligible'],'lir_input_pose_eligible':lir_pose['geometry_eligible']}
    summary.update(geometry_metrics(prototype,reference,device))
    metric=interface_metrics(atom_data([prototype],device),atom_data([c for c in result[0] if c.id!='K'],device),len(sequence))
    summary.update(metric)
    summary['joint_geometry_eligible']=bool(summary['backbone_geometry_eligible'] and summary['minimum_combined_severe_clashes']==0
        and binder_pose['geometry_eligible'] and lir_pose['geometry_eligible'])
    return result,summary

def worker(args):
    torch.cuda.set_device(args.gpu);device=torch.device('cuda:{}'.format(args.gpu));reference=RamaReference(device)
    generator=torch.Generator(device=device).manual_seed(20261013+args.gpu)
    bposes=read_jsonl(args.run/'module1/all_poses.jsonl');lposes=read_jsonl(args.run/'module2/all_poses.jsonl')
    rows=read_jsonl(args.run/'module3/pairs_100.jsonl')[args.gpu::2]
    output=args.run/'module3';(output/'structures').mkdir(exist_ok=True)
    started=time.monotonic()
    with (output/('pairs_gpu{}.jsonl'.format(args.gpu))).open('w') as handle:
        for row in rows:
            lir_pose=best_pose(lposes,row['lir_id'],'LC3B');variants=[];models={}
            for linker in LINKERS:
                for order in ORDERS:
                    key='{}_{}_{}'.format(row['id'],len(linker),order)
                    observations=[];variant_models={}
                    for target in TARGETS:
                        bpose=best_pose(bposes,row['binder_id'],target)
                        structure,summary=sample_variant(row,target,bpose,lir_pose,linker,order,reference,device,generator,args.samples)
                        observations.append(summary);variant_models[target]=structure
                    positive=[r for r in observations if r['target'] in MSA]
                    preference=(sum(r['joint_geometry_eligible'] for r in positive),
                                min(r['feasible_fraction'] for r in positive),
                                -sum(r['minimum_combined_severe_clashes'] for r in positive),-len(linker))
                    variants.append({'variant_id':key,'linker_sequence':linker,'module_order':order,
                          'selection_tuple':list(preference),'observations':observations})
                    models[key]=variant_models
            selected=max(variants,key=lambda r:tuple(r['selection_tuple']))
            sequence,bs,ls,ks=layout(row,selected['linker_sequence'],selected['module_order'])
            selected_row=dict(row,sequence=sequence,total_length=len(sequence),linker_sequence=selected['linker_sequence'],
                          linker_length=len(selected['linker_sequence']),module_order=selected['module_order'],
                          binder_start=bs+1,binder_end=bs+len(row['binder_sequence']),
                          lir_start=ls+1,lir_end=ls+len(row['lir_sequence']),linker_start=ks+1,
                          linker_end=ks+len(selected['linker_sequence']),variant_id=selected['variant_id'],
                          observations=selected['observations'],linker_variants=variants,modifications=[])
            paths={}
            for target,structure in models[selected['variant_id']].items():
                path=output/'structures'/('{}_{}.pdb'.format(row['id'],target));save_structure(structure,path);paths[target]=str(path)
            selected_row['structures']=paths
            selected_row['joint_supported_MSA_targets']=[r['target'] for r in selected['observations'] if r['target'] in MSA and r['joint_geometry_eligible']]
            handle.write(json.dumps(selected_row)+'\n');handle.flush()
            print('FUSION_PAIR',row['id'],'GPU',args.gpu,'variant',selected['variant_id'],'seconds',round(time.monotonic()-started,1),flush=True)

def screen_pool(run,rows):
    target=max(MSA,key=lambda t:sum(t in r['joint_supported_MSA_targets'] for r in rows))
    qualified=[r for r in rows if target in r['joint_supported_MSA_targets']]
    def priority(row):
        observation=next(o for o in row['observations'] if o['target']==target)
        return (len(row['joint_supported_MSA_targets']),observation['feasible_fraction'],-row['binder_rank']-row['lir_rank'])
    pool=[];bc={};lc={}
    for row in sorted(qualified,key=priority,reverse=True):
        if bc.get(row['binder_id'],0)>=3 or lc.get(row['lir_id'],0)>=3:continue
        row=dict(row,simulation_target=target);pool.append(row)
        bc[row['binder_id']]=bc.get(row['binder_id'],0)+1;lc[row['lir_id']]=lc.get(row['lir_id'],0)+1
        if len(pool)==20:break
    write_jsonl(Path(run)/'module3/screen_candidates.jsonl',pool)
    write_json(Path(run)/'module3/screen_selection.json',{'simulation_target':target,'qualified_pair_count':len(qualified),
        'selected_ids':[r['id'] for r in pool],'rule':'Common MSA target with most joint-qualified models, then cross-MSA coverage and feasible fraction; at most three occurrences of either source module in the review pool.'})
    return pool

def rank(run,rows,stability):
    lookup={r['id']:r for r in rows};stability_map={r['id']:r for r in stability}
    cards=[]
    for row in rows:
        card={k:v for k,v in row.items() if k not in ('structures','linker_variants')}
        card.update(chemistry(row['sequence']));card['stability']=stability_map[row['id']];cards.append(card)
    text={'type':'string'}
    item=object_schema({'fusion_id':{'type':'string','enum':list(lookup)},'support':text,'tradeoffs':text,'uncertainty':text})
    schema=object_schema({'ranked_candidates':{'type':'array','minItems':10,'maxItems':10,'items':item},'decision_summary':text})
    if len(rows)<10:raise ValueError('Fusion review has {} supported candidates; ten are required'.format(len(rows)))
    answer=llm(run,'fusion_ranking_order1','fusion_rank.txt',{'COMPLETE_CHIMERA_EVIDENCE_JSON':cards,
             'METRIC_DEFINITIONS_JSON':METRICS},schema,7500)
    check=llm(run,'fusion_ranking_order2','fusion_rank.txt',{'COMPLETE_CHIMERA_EVIDENCE_JSON':list(reversed(cards)),
             'METRIC_DEFINITIONS_JSON':METRICS},schema,7500)
    ids1=[r['fusion_id'] for r in answer['ranked_candidates']];ids2=[r['fusion_id'] for r in check['ranked_candidates']]
    write_json(Path(run)/'module3/ranking_order_agreement.json',{'primary_order':ids1,'reversed_input_order':ids2,
        'top10_overlap':len(set(ids1)&set(ids2))})
    write_json(Path(run)/'module3/llm_ranking_order2.json',check)
    selected=[]
    for rank,item in enumerate(answer['ranked_candidates'],1):
        row=dict(lookup[item['fusion_id']]);row.update(item);row['rank']=rank;row['stability']=stability_map[row['id']]
        selected.append(row)
    folder=Path(run)/'module3';write_json(folder/'llm_ranking.json',answer)
    write_jsonl(folder/'top10.jsonl',selected)
    fields=('rank','id','binder_id','lir_id','binder_sequence','binder_length','linker_sequence','linker_length',
            'lir_sequence','lir_length','module_order','sequence','total_length','binder_start','binder_end',
            'linker_start','linker_end','lir_start','lir_end','support','tradeoffs','uncertainty')
    write_csv(folder/'top10.csv',[{k:r[k] for k in fields} for r in selected]);write_fasta(folder/'top10.fasta',selected)
    return selected

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--samples',type=int,default=8);worker(p.parse_args())
