"""GPU local-receptor dynamics and dual-interface persistence measurements."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
import openmm
from openmm import app,unit
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from core import read_jsonl,write_csv
from geometry import AA,read_structure,save_structure,tensor,add_caps,geometry_metrics,RamaReference
from structure import platform,binding_energy,chiral_force

def local_complex(full,row,device):
    peptide_coords=tensor([a.coord for a in full[0]['K'].get_atoms() if a.element!='H'],device)
    fragments=Structure('local_fibril');fragments.add(Model(0))
    for chain in full[0]:
        if chain.id in ('K','R'):continue
        residues=[r for r in chain if r.resname in AA]
        ca=tensor([r['CA'].coord for r in residues],device)
        near=torch.nonzero(torch.cdist(ca,peptide_coords).min(-1).values<18).flatten()
        if len(near):
            first=max(0,int(near.min())-3);last=min(len(residues),int(near.max())+4)
            fragment=copy.deepcopy(chain)
            keep={r.id for r in residues[first:last]}
            for residue in list(fragment):
                if residue.id not in keep:fragment.detach_child(residue.id)
            fragment.detach_parent();fragments[0].add(fragment)
    model=add_caps(fragments,device)
    for name in ('K','R'):
        chain=copy.deepcopy(full[0][name]);chain.detach_parent();model[0].add(chain)
    for atom in list(model.get_atoms()):
        if atom.element=='H':atom.get_parent().detach_child(atom.id)
    return model

def contact_pairs(x,y,xi,yi):
    distance=torch.cdist(x,y)
    pairs=xi[:,None]*(int(yi.max())+1)+yi[None,:]
    return torch.unique(pairs[distance<4.5])

def retention(current,baseline):
    if len(baseline)==0:return None
    return float(torch.isin(baseline,current).double().mean())

def simulate(row,run,gpu,duration_ps,repeat,phase,index):
    torch.cuda.set_device(gpu);device=torch.device('cuda:{}'.format(gpu))
    folder=run/'module3/stability'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    initial_path=folder/('{}_rep{}_initial.pdb'.format(phase,repeat))
    target=row['simulation_target'];full=read_structure(row['structures'][target])
    model=local_complex(full,row,device);save_structure(model,initial_path)
    pdb=app.PDBFile(str(initial_path));modeller=app.Modeller(pdb.topology,pdb.positions)
    ff=app.ForceField('amber14-all.xml','implicit/obc2.xml')
    modeller.addHydrogens(pH=7.,platform=platform(gpu))
    system=ff.createSystem(modeller.topology,nonbondedMethod=app.NoCutoff,constraints=None)
    chiral_force(system,modeller.topology,modeller.positions,gpu,('K','R'))
    for atom_index,atom in enumerate(modeller.topology.atoms()):
        if atom.residue.chain.id not in ('K','R') and atom.element.symbol!='H':
            system.setParticleMass(atom_index,0*unit.dalton)
    integrator=openmm.LangevinMiddleIntegrator(300*unit.kelvin,1/unit.picosecond,.001*unit.picoseconds)
    random_seed=20261013+100000*int(phase=='final')+repeat*1000+2*index+gpu
    integrator.setRandomNumberSeed(random_seed)
    simulation=app.Simulation(modeller.topology,system,integrator,platform(gpu),{'DeviceIndex':str(gpu),'Precision':'mixed'})
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance=5*unit.kilojoule_per_mole/unit.nanometer,maxIterations=300)
    initial=simulation.context.getState(getPositions=True,getEnergy=True)
    initial_energy=simulation.context.getState(getEnergy=True,groups=1).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    with initial_path.open('w') as handle:
        app.PDBFile.writeFile(modeller.topology,initial.getPositions(),handle,keepIds=True)
    atoms=list(modeller.topology.atoms());b_atoms=[];l_atoms=[];alpha_atoms=[];lc3_atoms=[]
    for i,atom in enumerate(atoms):
        if atom.element.symbol=='H':continue
        name=atom.residue.chain.id
        if name=='K':
            number=int(atom.residue.id)
            if row['binder_start']<=number<=row['binder_end']:b_atoms.append(i)
            if row['lir_start']<=number<=row['lir_end']:l_atoms.append(i)
        elif name=='R' and atom.residue.name in AA:lc3_atoms.append(i)
        elif name!='R' and atom.residue.name in AA:alpha_atoms.append(i)
    b_atoms=torch.tensor(b_atoms,device=device);l_atoms=torch.tensor(l_atoms,device=device)
    alpha_atoms=torch.tensor(alpha_atoms,device=device);lc3_atoms=torch.tensor(lc3_atoms,device=device)
    residue_keys={}
    residue_indices=[]
    for atom in atoms:
        key=(atom.residue.chain.id,atom.residue.id)
        residue_keys.setdefault(key,len(residue_keys));residue_indices.append(residue_keys[key])
    residue_indices=torch.tensor(residue_indices,device=device)
    x=tensor(initial.getPositions().value_in_unit(unit.angstrom),device)
    b_reference=contact_pairs(x[b_atoms],x[alpha_atoms],residue_indices[b_atoms],residue_indices[alpha_atoms])
    l_reference=contact_pairs(x[l_atoms],x[lc3_atoms],residue_indices[l_atoms],residue_indices[lc3_atoms])
    steps=int(duration_ps*1000);interval=1000
    simulation.reporters.append(app.DCDReporter(str(folder/('{}_rep{}.dcd'.format(phase,repeat))),interval))
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin,random_seed)
    observations=[]
    for frame in range(steps//interval+1):
        if frame:simulation.step(interval)
        state=simulation.context.getState(getPositions=True,getEnergy=True)
        x=tensor(state.getPositions().value_in_unit(unit.angstrom),device)
        bp=contact_pairs(x[b_atoms],x[alpha_atoms],residue_indices[b_atoms],residue_indices[alpha_atoms])
        lp=contact_pairs(x[l_atoms],x[lc3_atoms],residue_indices[l_atoms],residue_indices[lc3_atoms])
        br=retention(bp,b_reference);lr=retention(lp,l_reference)
        observations.append({'time_ps':frame,'binder_contact_pairs':len(bp),'lir_contact_pairs':len(lp),
             'binder_contact_retention':br,'lir_contact_retention':lr,
             'simultaneous_retention':br is not None and lr is not None and br>=.5 and lr>=.5,
             'potential_energy_kj_mol':state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)})
    final=simulation.context.getState(getPositions=True)
    final_path=folder/('{}_rep{}_final.pdb'.format(phase,repeat))
    with final_path.open('w') as handle:app.PDBFile.writeFile(modeller.topology,final.getPositions(),handle,keepIds=True)
    write_csv(folder/('{}_rep{}_timeseries.csv'.format(phase,repeat)),observations)
    result={'id':row['id'],'phase':phase,'repeat':repeat,'duration_ps':duration_ps,'random_seed':random_seed,
        'target':target,'solvent_model':'AMBER14/OBC2 implicit solvent','temperature_k':300,
        'timestep_fs':1,'local_fibril_cutoff_a':18,'fibril_heavy_atoms':'fixed',
        'LC3B_and_fusion':'mobile','initial_binder_contact_pairs':len(b_reference),'initial_lir_contact_pairs':len(l_reference),
        'mm_gb_binding_proxy_kcal_mol':None,
        'mean_binder_contact_retention':float(np.mean([r['binder_contact_retention'] for r in observations])) if len(b_reference) else None,
        'mean_lir_contact_retention':float(np.mean([r['lir_contact_retention'] for r in observations])) if len(l_reference) else None,
        'simultaneous_retention_fraction':float(np.mean([r['simultaneous_retention'] for r in observations])),
        'initial_pdb':str(initial_path),'final_pdb':str(final_path),'atom_count':len(atoms),
        'final_peptide_geometry':geometry_metrics(read_structure(final_path)[0]['K'],RamaReference(device),device),
        'scope':'Short local dynamics under a fixed fibril context; compatibility and contact persistence proxies, with membrane, uptake, cellular autophagy and actual degradation outside the simulated system.'}
    del simulation,integrator
    return result

def worker(args):
    rows=read_jsonl(args.rows)[args.gpu::2]
    log=args.run/'module3'/('{}_gpu{}.jsonl'.format(args.phase,args.gpu))
    started=time.monotonic()
    with log.open('w') as handle:
        for index,row in enumerate(rows):
            for repeat in range(1,args.repeats+1):
                result=simulate(row,args.run,args.gpu,args.duration_ps,repeat,args.phase,index)
                handle.write(json.dumps(result)+'\n');handle.flush()
                print('DYNAMICS',args.phase,row['id'],'repeat',repeat,'GPU',args.gpu,
                      'retention',round(result['simultaneous_retention_fraction'],3),
                      'seconds',round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--duration-ps',type=float,required=True)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--phase',required=True);worker(p.parse_args())
