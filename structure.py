"""GPU poses, one-pass chemical standardization, focused review."""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
import openmm
from openmm import app,unit
from core import ROOT,read_json,read_jsonl,TARGETS
from geometry import (AA,read_structure,save_structure,tensor,angles_from_chain,RamaReference,
                      peptide,standardize,geometry_metrics,atom_data,interface_metrics,align,rotamers,dihedral,
                      focused_adjustment,attachment_metrics)
sys.path.insert(0,str(ROOT/'vendor/PepGLAD'))
from data.codesign import calculate_covariance_matrix
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from data.format import Atom,Block,VOCAB
from utils.const import sidechain_atoms

def model_input(pocket,receptor,sequence):
    lookup={tuple(row) for row in pocket};selected=[];positions=[]
    for chain,blocks in receptor.items():
        for index,block in enumerate(blocks):
            if (chain,block.id[0]) in lookup:
                selected.append(block);positions.append(index+1)
    ligand=[Block(VOCAB.symbol_to_abrv(letter),[Atom(name,[0.,0.,0.],name[0])
             for name in VOCAB.backbone_atoms+sidechain_atoms[letter]]) for letter in sequence]
    mask=torch.tensor([False]*len(selected)+[True]*len(ligand));positions.extend(range(1,len(ligand)+1))
    coordinates=[];symbols=[];atom_masks=[]
    for block in selected+ligand:
        symbol=VOCAB.abrv_to_symbol(block.abrv);atoms={a.name:a.get_coord() for a in block.units}
        centroid=np.mean(list(atoms.values()),axis=0).tolist();names=VOCAB.backbone_atoms+sidechain_atoms.get(symbol,[])
        coordinates.append([atoms.get(name,centroid) for name in names]+[centroid]*(14-len(names)))
        atom_masks.append([name in atoms for name in names]+[False]*(14-len(names)))
        symbols.append(VOCAB.symbol_to_idx(symbol))
    x=torch.tensor(coordinates,dtype=torch.float);atom_mask=torch.tensor(atom_masks,dtype=torch.bool)
    covariance=calculate_covariance_matrix(x[~mask][:,1][atom_mask[~mask][:,1]].numpy())+1e-4*np.eye(3)
    return {'X':x,'S':torch.tensor(symbols),'position_ids':torch.tensor(positions),'mask':mask,
      'atom_mask':atom_mask,'lengths':torch.tensor([len(symbols)]),
      'L':torch.from_numpy(np.linalg.cholesky(covariance)).float().unsqueeze(0)}

def platform(gpu):
    result=openmm.Platform.getPlatformByName('OpenCL')
    result.setPropertyDefaultValue('DeviceIndex',str(gpu));result.setPropertyDefaultValue('Precision','mixed')
    return result

def chiral_force(system,topology,positions,gpu,chains):
    """Intramolecular improper priors preserve canonical stereocenters."""
    coordinates=tensor(positions.value_in_unit(unit.angstrom),'cuda:{}'.format(gpu))
    force=openmm.CustomTorsionForce('stereo_k*(1-cos(theta-theta0))')
    force.addGlobalParameter('stereo_k',1000.);force.addPerTorsionParameter('theta0');force.setForceGroup(1)
    for residue in topology.residues():
        if residue.chain.id not in chains or residue.name not in AA:continue
        atoms={a.name:a.index for a in residue.atoms()};groups=[]
        if residue.name!='GLY':groups.append(('N','C','CA','CB'))
        if residue.name=='THR':groups.append(('CA','OG1','CB','CG2'))
        if residue.name=='ILE':groups.append(('CA','CG1','CB','CG2'))
        for group in groups:
            indices=[atoms[n] for n in group]
            theta=float(dihedral(*coordinates[indices]))*np.pi/180
            force.addTorsion(*indices,[theta])
    system.addForce(force)

def isolated_energy(topology,positions,gpu,forcefield):
    system=forcefield.createSystem(topology,nonbondedMethod=app.NoCutoff,constraints=None)
    integrator=openmm.VerletIntegrator(.001*unit.picoseconds)
    context=openmm.Context(system,integrator,platform(gpu),{'DeviceIndex':str(gpu),'Precision':'mixed'})
    context.setPositions(positions)
    energy=context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del context,integrator
    return energy

def binding_energy(topology,positions,complex_energy,gpu,forcefield):
    receptor=app.Modeller(topology,positions)
    receptor.delete([c for c in receptor.topology.chains() if c.id=='K'])
    peptide_only=app.Modeller(topology,positions)
    peptide_only.delete([c for c in peptide_only.topology.chains() if c.id!='K'])
    target_energy=isolated_energy(receptor.topology,receptor.positions,gpu,forcefield)
    peptide_energy=isolated_energy(peptide_only.topology,peptide_only.positions,gpu,forcefield)
    return (complex_energy-target_energy-peptide_energy)/4.184

def relax_structure(complex_structure,path,gpu,forcefield,estimate_binding=False,max_iterations=150):
    save_structure(complex_structure,path);pdb=app.PDBFile(str(path))
    modeller=app.Modeller(pdb.topology,pdb.positions)
    modeller.addHydrogens(pH=7.0,platform=platform(gpu))
    system=forcefield.createSystem(modeller.topology,nonbondedMethod=app.NoCutoff,constraints=None)
    chiral_force(system,modeller.topology,modeller.positions,gpu,('K',))
    for i,atom in enumerate(modeller.topology.atoms()):
        if atom.element.symbol!='H' and (atom.residue.chain.id!='K' or atom.name in ('N','CA','C','O','OXT','CB')):
            system.setParticleMass(i,0*unit.dalton)
    integrator=openmm.VerletIntegrator(.001*unit.picoseconds)
    simulation=app.Simulation(modeller.topology,system,integrator,platform(gpu),{'DeviceIndex':str(gpu),'Precision':'mixed'})
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance=5*unit.kilojoule_per_mole/unit.nanometer,maxIterations=max_iterations)
    state=simulation.context.getState(getPositions=True,getEnergy=True)
    with path.open('w') as handle:
        app.PDBFile.writeFile(simulation.topology,state.getPositions(),handle,keepIds=True)
    energy=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    amber_energy=simulation.context.getState(getEnergy=True,groups=1).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    topology=simulation.topology;positions=state.getPositions()
    del simulation,integrator
    proxy=binding_energy(topology,positions,amber_energy,gpu,forcefield) if estimate_binding else None
    return energy,proxy

def lir_template(sequence,dossier,reference,device,raw_reference,flank_noise=0.):
    original=[r for r in raw_reference[0]['K'] if r.resname in AA]
    phi_ref,psi_ref=angles_from_chain(raw_reference[0]['K'],device)
    w_index=next(i for i,r in enumerate(original) if r.resname=='TRP')
    phi=torch.full((len(sequence),),-120.,device=device,dtype=torch.float64);psi=torch.full_like(phi,140.)
    for i in range(len(sequence)):
        j=i-6+w_index
        if 0<=j<len(original):
            phi[i]=phi_ref[j];psi[i]=psi_ref[j]
    flank=torch.ones(len(sequence),device=device);flank[6:10]=0
    phi=phi+torch.randn_like(phi)*flank*flank_noise;psi=psi+torch.randn_like(psi)*flank*flank_noise
    phi,psi=reference.project(sequence,phi,psi)
    source_rotamers=rotamers(raw_reference[0]['K'],device)
    mapped_rotamers={i:source_rotamers[i-6+w_index] for i in range(len(sequence))
                    if i-6+w_index in source_rotamers and AA[original[i-6+w_index].resname]==sequence[i]}
    chain=peptide(sequence,phi,psi,device,sidechain_angles=mapped_rotamers)
    source=tensor([list(chain)[i]['CA'].coord for i in range(6,10)],device)
    rotation,translation=align(source,tensor(dossier['core_reference_ca'],device))
    atoms=list(chain.get_atoms());coordinates=tensor([a.coord for a in atoms],device)@rotation+translation
    for atom,coordinate in zip(atoms,coordinates.cpu().numpy()):
        atom.coord=coordinate
    return chain

def evaluate(row,structure,dossier,reference,device):
    chain=structure[0]['K'];metric=geometry_metrics(chain,reference,device)
    metric.update(interface_metrics(atom_data([chain],device),atom_data([c for c in structure[0] if c.id!='K'],device),len(row['sequence'])))
    if row['target']=='LC3B':
        residues=[r for r in chain if r.resname in AA]
        core=tensor([residues[i]['CA'].coord for i in range(6,10)],device)
        metric['core_rmsd_a']=round(torch.sqrt((core-tensor(dossier['core_reference_ca'],device)).square().sum(-1).mean()).item(),3)
        centers=tensor([np.mean([a.coord for a in residues[i] if a.element!='H' and a.name not in ('N','CA','C','O')],axis=0)
                        for i in (6,9)],device)
        metric['core_two_pocket']=bool((torch.linalg.vector_norm(centers-tensor(dossier['core_reference_sidechain_centroids'],device),dim=-1)<3).all())
    metric.update(attachment_metrics(chain,structure[0],device))
    metric['geometry_eligible']=metric['backbone_geometry_eligible'] and metric['severe_clashes']==0 and row['backbone_reconstruction_rmsd_a']<=3.0
    ligand=atom_data([chain],device);receptor=atom_data([c for c in structure[0] if c.id!='K'],device)
    target_residues=[r for c in structure[0] if c.id!='K' for r in c]
    distance=torch.cdist(ligand['xyz'],receptor['xyz'])
    native=ligand['native'][:,None]&receptor['native'][None,:]
    ids=ligand['residue'][:,None]*len(target_residues)+receptor['residue'][None,:]
    observed=[]
    for code in torch.unique(ids[(distance<4.5)&native]).cpu().tolist():
        i,j=divmod(code,len(target_residues));tr=target_residues[j]
        observed.append([i+1,tr.get_parent().id,tr.id[1],AA[tr.resname]])
    metric['observed_residue_contacts']=observed
    return metric

def batch_inputs(samples,device):
    return {key:torch.cat([item[key] for item in samples]).to(device) for key in samples[0]}

def fixed_geometry_collisions(structure,device):
    residues=[r for r in structure[0]['K'] if r.resname in AA]
    atoms=[(i,a) for i,r in enumerate(residues) for a in r if a.name in ('N','CA','C','O','OXT','CB')]
    coordinates=tensor([a.coord for i,a in atoms],device)
    indices=torch.tensor([i for i,a in atoms],device=device)
    receptor=tensor([a.coord for c in structure[0] if c.id!='K' for a in c.get_atoms() if a.element!='H'],device)
    external=int((torch.cdist(coordinates,receptor)<2).sum())
    internal=int(((torch.cdist(coordinates,coordinates)<2)&((indices[:,None]-indices[None,:]).abs()>=2)).sum())//2
    return external+internal

def worker(args):
    torch.cuda.set_device(args.gpu);device=torch.device('cuda:{}'.format(args.gpu))
    torch.manual_seed(20261013+args.gpu+100*int(args.phase=='review'))
    reference=RamaReference(device);ff=app.ForceField('amber14-all.xml','implicit/obc2.xml')
    candidates=read_jsonl(args.candidates)[args.gpu::2]
    metadata=read_json(args.run/'inputs/target_metadata.json')
    contexts={c['site_id']:c for c in read_json(args.run/'inputs/surface_contexts.json')}
    dossier=read_json(args.run/'inputs/lir_reference_dossier.json')
    target_names=([t for t in metadata if t.startswith('9A1A_')] if args.phase=='monomer' else list(TARGETS)) if args.module=='module1' else ['LC3B']
    if args.module=='module1':model=torch.load(ROOT/'vendor/PepGLAD/checkpoints/fixseq.ckpt',map_location='cpu').to(device).eval()
    folder=args.run/args.module/'poses'/args.phase;folder.mkdir(parents=True,exist_ok=True)
    log=args.run/args.module/('{}_gpu{}.jsonl'.format(args.phase,args.gpu));started=time.monotonic();completed=0
    initial=read_jsonl(args.run/'module1/initial_poses.jsonl') if args.module=='module1' and args.phase=='review' else []
    with log.open('w') as handle:
        for target in target_names:
            info=metadata[target] if target!='LC3B' else dossier
            raw_target=read_structure(info['target_pdb'])
            chemical_target=read_structure(info['capped_pdb'])
            blocks=pdb_to_list_blocks(info['target_pdb'],[c.id for c in raw_target[0]],dict_form=True)
            for start in range(0,len(candidates),4):
                batch_candidates=candidates[start:start+4];proposals=[]
                if target=='LC3B':
                    for candidate in batch_candidates:
                        for repeat in range(3):
                            torch.manual_seed(20261013+repeat)
                            chain=lir_template(candidate['sequence'],dossier,reference,device,
                                read_structure(args.run/'inputs/lc3_p62_reference.pdb'),flank_noise=0. if repeat==0 else 8.)
                            proposals.append((candidate,repeat,chain,'experimental_motif_template'))
                else:
                    inputs=[model_input(contexts[c['site_id']]['conformers'][target]['pocket'],blocks,c['sequence']) for c in batch_candidates]
                    batch=batch_inputs(inputs,device)
                    sampling=time.monotonic()
                    with torch.no_grad():
                        xyzs,_,_=model.sample(batch['X'],batch['S'],batch['mask'],batch['position_ids'],batch['lengths'],batch['atom_mask'],
                            L=batch['L'],sample_opt={'energy_func':'default','energy_lambda':.5})
                    print('INFERENCE',target,'GPU',args.gpu,'batch',len(batch_candidates),'seconds',round(time.monotonic()-sampling,2),flush=True)
                    for candidate,xyz in zip(batch_candidates,xyzs):
                        repeat=2 if args.phase=='review' else 1
                        ligand=[Block(VOCAB.symbol_to_abrv(letter),[Atom(name,coordinate,name[0])
                            for name,coordinate in zip(VOCAB.backbone_atoms+sidechain_atoms[letter],coords)])
                            for letter,coords in zip(candidate['sequence'],xyz)]
                        raw_path=folder/('{}_{}_raw{}.pdb'.format(candidate['id'],target,repeat))
                        list_blocks_to_pdb([ligand],['K'],str(raw_path))
                        proposals.append((candidate,repeat,read_structure(raw_path)[0]['K'],'PepGLAD_fixed_sequence'))
                        if args.phase=='review':
                            first=next(r for r in initial if r['id']==candidate['id'] and r['target']==target)
                            proposals.append((candidate,1,read_structure(first['raw_pdb'])[0]['K'],'PepGLAD_fixed_sequence'))
                for candidate,repeat,raw_chain,source in proposals:
                    stage_started=time.monotonic();sequence=candidate['sequence']
                    name='{}_{}_rep{}'.format(candidate['id'],target,repeat)
                    raw_path=folder/(name+'_proposal.pdb');proposal=copy.deepcopy(raw_target);proposal[0].add(copy.deepcopy(raw_chain));save_structure(proposal,raw_path)
                    if target=='LC3B':chain=copy.deepcopy(raw_chain);rmsd=0.
                    else:chain,rmsd=standardize(sequence,raw_chain,reference,device)
                    local_displacement=0.
                    pre=interface_metrics(atom_data([chain],device),atom_data(list(chemical_target[0]),device),len(sequence))
                    if args.phase=='review' and 0<pre['severe_clashes']<=80 and rmsd<=3:
                        local_displacement=focused_adjustment(chain,chemical_target,device,20)
                    complex_structure=copy.deepcopy(chemical_target);complex_structure[0].add(chain)
                    path=folder/(name+'.pdb');save_structure(complex_structure,path)
                    final_ca=tensor([r['CA'].coord for r in chain if r.resname in AA],device)
                    raw_ca=tensor([r['CA'].coord for r in raw_chain if r.resname in AA],device)
                    rmsd=float((final_ca-raw_ca).square().sum(-1).mean().sqrt())
                    row={'id':candidate['id'],'sequence':sequence,'target':target,'repeat':repeat,'phase':args.phase,
                        'pose_source':source,'pdb':str(path),'raw_pdb':str(raw_path),
                        'backbone_reconstruction_rmsd_a':round(rmsd,3),'local_adjustment_displacement_a':round(local_displacement,3),
                        'before_relaxation':geometry_metrics(chain,reference,device),
                        'potential_energy_kj_mol':None,'mm_gb_binding_proxy_kcal_mol':None}
                    constant=fixed_geometry_collisions(complex_structure,device);row['fixed_geometry_collisions']=constant
                    if (target=='LC3B' or args.phase=='review') and constant==0 and rmsd<=3:
                        energy,proxy=relax_structure(complex_structure,path,args.gpu,ff,estimate_binding=target=='LC3B')
                        row.update(potential_energy_kj_mol=round(energy,2),mm_gb_binding_proxy_kcal_mol=round(proxy,3) if proxy is not None else None,
                            relaxation_status='bounded_sidechain_minimization')
                    else:row['relaxation_status']='one_pass_standardization'
                    row.update(evaluate(row,read_structure(path),dossier,reference,device))
                    row['evaluation_seconds']=round(time.monotonic()-stage_started,3)
                    handle.write(json.dumps(row)+'\n');handle.flush();completed+=1
                    print('POSES',args.module,args.phase,'GPU',args.gpu,row['id'],target,'count',completed,
                        'eligible',row['geometry_eligible'],'seconds',round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--module',choices=('module1','module2'),required=True);p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--phase',choices=('initial','review','monomer'),default='initial')
    worker(p.parse_args())
