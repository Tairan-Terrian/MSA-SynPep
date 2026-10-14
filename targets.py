"""Prepare raw-source assemblies, jointly compare surfaces, and prepare LC3B."""

import copy
import json
from pathlib import Path
import numpy as np
import torch
import freesasa
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from pdbfixer import PDBFixer
from openmm import app

from core import ROOT, TARGETS, MSA, CONTROLS, write_json, write_csv
from geometry import read_structure,save_structure,AA,tensor,add_caps


def protein_model(source):
    structure=read_structure(source)
    for chain in list(structure[0]):
        for residue in list(chain):
            if residue.resname not in AA:
                chain.detach_child(residue.id)
        if len(chain)==0:
            structure[0].detach_child(chain.id)
    return structure


def prepare_heavy(structure,path,free_c=False):
    save_structure(structure,path)
    fixer=PDBFixer(filename=str(path))
    fixer.findMissingResidues();fixer.missingResidues={}
    fixer.findMissingAtoms();fixer.addMissingAtoms()
    with path.open('w') as handle:
        app.PDBFile.writeFile(fixer.topology,fixer.positions,handle,keepIds=True)
    result=read_structure(path)
    for chain in result[0]:
        for residue in chain:
            if 'OXT' in residue and not free_c:
                residue.detach_child('OXT')
    save_structure(result,path)
    return result


def lewy_assembly(source):
    original=protein_model(source)[0]['A']
    operations={}
    for line in source.read_text().splitlines():
        if line.startswith('REMARK 350   BIOMT'):
            parts=line.split();row=int(parts[2][-1])-1;operation=int(parts[3])
            operations.setdefault(operation,np.zeros((4,4)))[row,:]=list(map(float,parts[4:8]))
    for matrix in operations.values():
        matrix[3,3]=1
    matrices=[operations[3]@operations[3],operations[3],operations[1],operations[2],operations[2]@operations[2]]
    result=Structure('8A9L');result.add(Model(0))
    for name,matrix in zip('ABCDE',matrices):
        chain=copy.deepcopy(original);chain.detach_parent();chain.id=name
        for atom in chain.get_atoms():
            atom.coord=atom.coord@matrix[:3,:3].T+matrix[:3,3]
        result[0].add(chain)
    return result


def local_basis(residue,device):
    n,ca,c=tensor([residue[a].coord for a in ('N','CA','C')],device)
    e1=(n-ca)/torch.linalg.vector_norm(n-ca)
    e2=c-ca-torch.dot(c-ca,e1)*e1;e2=e2/torch.linalg.vector_norm(e2)
    return torch.stack((e1,e2,torch.cross(e1,e2,dim=-1)))


def protofilament_groups(model,device):
    """Separate stacks by sequence boundaries and local backbone orientation."""
    groups=[]
    for chain in model:
        signature=tuple((r.id[1],r.resname) for r in chain)
        basis=local_basis(chain[66],device)
        matched=next((g for g in groups if g['signature']==signature and float((basis*g['basis']).sum())>2.8),None)
        if matched is None:groups.append({'signature':signature,'basis':basis,'chains':[chain]})
        else:matched['chains'].append(chain)
    return sorted(groups,key=lambda g:len(g['signature']),reverse=True)


def central_layers(model,device,count):
    """Keep central rungs separately in every actual protofilament stack."""
    keep=[]
    for group in protofilament_groups(model,device):
        chains=group['chains']
        xyz=tensor([c[66]['CA'].coord for c in chains],device)
        order=torch.linalg.vector_norm(xyz-xyz.mean(0),dim=-1).argsort().cpu().tolist()
        keep.extend(chains[i].id for i in order[:count])
    result=Structure('central_layers');result.add(copy.deepcopy(model))
    for chain in list(result[0]):
        if chain.id not in keep:result[0].detach_child(chain.id)
    return result


def monomer_models(source):
    """Read the IHM atom-site schema directly (it omits PDB insertion codes)."""
    data=MMCIF2Dict(str(source));result=Structure('9A1A')
    models={};chains={};residues={}
    for i,element in enumerate(data['_atom_site.type_symbol']):
        if element=='H' or data['_atom_site.label_alt_id'][i] not in ('.','A'):continue
        model_id=int(data['_atom_site.pdbx_PDB_model_num'][i]);chain_id=data['_atom_site.auth_asym_id'][i]
        number=int(data['_atom_site.label_seq_id'][i]);name=data['_atom_site.label_comp_id'][i]
        if model_id not in models:
            models[model_id]=Model(model_id);result.add(models[model_id])
        ck=(model_id,chain_id)
        if ck not in chains:
            chains[ck]=Chain(chain_id);models[model_id].add(chains[ck])
        rk=(model_id,chain_id,number)
        if rk not in residues:
            residues[rk]=Residue((' ',number,' '),name,' ');chains[ck].add(residues[rk])
        atom=data['_atom_site.label_atom_id'][i]
        xyz=np.array([float(data['_atom_site.Cartn_'+axis][i]) for axis in ('x','y','z')],dtype=np.float32)
        residues[rk].add(Atom(atom,xyz,float(data['_atom_site.B_iso_or_equiv'][i]),float(data['_atom_site.occupancy'][i]),' ',atom.rjust(4),i,element=element))
    return result


def prepare(run):
    device=torch.device('cuda:0')
    inputs=Path(run)/'inputs';inputs.mkdir()
    directory=inputs/'targets';directory.mkdir()
    metadata={};models={};sasa={};sasa3={};inventory=[]
    for key in TARGETS:
        source=ROOT/'data/pdb'/(key+'.pdb')
        structure=lewy_assembly(source) if key=='8A9L' else protein_model(source)
        model=prepare_heavy(structure,directory/(key+'.pdb'))
        groups=protofilament_groups(model[0],device)
        candidates=groups[0]['chains']
        coords=tensor([c[66]['CA'].coord for c in candidates],device)
        central=candidates[int(torch.linalg.vector_norm(coords-coords.mean(0),dim=-1).argmin())].id
        capped=add_caps(model,device)
        save_structure(capped,directory/(key+'_capped.pdb'))
        area=freesasa.calc(freesasa.Structure(str(directory/(key+'.pdb')))).residueAreas()
        sasa[key]={int(number):value.total for number,value in area[central].items()}
        three=central_layers(model[0],device,3);save_structure(three,directory/(key+'_three_layers.pdb'))
        area3=freesasa.calc(freesasa.Structure(str(directory/(key+'_three_layers.pdb')))).residueAreas()
        sasa3[key]={int(number):value.total for number,value in area3[central].items()}
        metadata[key]={'central_chain':central,'chains':[c.id for c in model[0]],
              'target_pdb':str(directory/(key+'.pdb')),'capped_pdb':str(directory/(key+'_capped.pdb')),
              'label':'MSA target' if key in MSA else 'same-protein structural control',
              'source':'https://www.rcsb.org/structure/'+key,
              'protofilament_groups':[[c.id for c in g['chains']] for g in groups],
              'central_selection':'Middle rung within the first longest-resolved protofilament stack; other protofilaments remain in the receptor context.',
              'assembly':'five helical layers from deposited BIOMT transforms' if key=='8A9L' else 'all deposited protein layers and protofilaments',
              'terminal_chemistry':'ACE/NME caps on internal resolved-core boundaries; unresolved full-length tails remain unmodeled'}
        models[key]=model[0]
        for chain in model[0]:
            numbers=[r.id[1] for r in chain]
            inventory.append({'target':key,'class':metadata[key]['label'],'chain':chain.id,
                              'start':min(numbers),'end':max(numbers),'resolved_residues':len(numbers)})
    sensitivity=[{'target':k,'residue':n,'central_chain':metadata[k]['central_chain'],
        'three_layers_sasa_a2':round(sasa3[k][n],3),'five_layers_sasa_a2':round(sasa[k][n],3),
        'absolute_difference_a2':round(abs(sasa3[k][n]-sasa[k][n]),3)} for k in TARGETS for n in sasa[k]]
    write_csv(inputs/'layer_sensitivity.csv',sensitivity)
    common=sorted(set.intersection(*[{r.id[1] for r in models[k][metadata[k]['central_chain']]} for k in TARGETS]))
    scores=[];contexts=[]
    for number in common:
        neighborhood=[n for n in common if abs(number-n)<=5]
        shape={}
        for key in TARGETS:
            chain=models[key][metadata[key]['central_chain']]
            xyz=tensor([chain[n]['CA'].coord for n in neighborhood],device)
            shape[key]=torch.cdist(xyz,xyz)
        difference=min(float((shape[p]-shape[n]).abs().mean()) for p in MSA for n in CONTROLS)
        exposed=min(sasa[p][number] for p in MSA)
        scores.append({'residue':number,'minimum_msa_sasa_a2':round(exposed,2),
                       'minimum_shape_difference_a':round(difference,3),
                       'surface_priority':round(exposed*difference,3)})
    chosen=[]
    for row in sorted(scores,key=lambda r:r['surface_priority'],reverse=True):
        number=row['residue']
        if row['minimum_msa_sasa_a2']>20 and all(abs(number-r['residue'])>=8 for r in chosen):
            chosen.append(row)
        if len(chosen)==4:
            break
    for index,row in enumerate(chosen,1):
        number=row['residue'];site='site_{}'.format(index);by_target={}
        for key in TARGETS:
            chain=models[key][metadata[key]['central_chain']]
            center=tensor(chain[number]['CA'].coord,device)
            pocket=[];observations=[]
            for context_chain in models[key]:
                for residue in context_chain:
                    distance=float(torch.linalg.vector_norm(tensor(residue['CA'].coord,device)-center))
                    if distance<11:
                        pocket.append([context_chain.id,residue.id[1]])
                        observations.append({'chain':context_chain.id,'residue':residue.id[1],
                              'aa':AA[residue.resname],'distance_to_center_a':round(distance,1)})
            neighbors=[n for n in common if abs(number-n)<=5]
            xyz=tensor([chain[n]['CA'].coord for n in neighbors],device)
            matrix=torch.cdist(xyz,xyz).detach().cpu().numpy()
            by_target[key]={'central_chain':chain.id,'pocket':pocket,
                'center_sasa_a2':round(sasa[key][number],2),'central_residue_numbers':neighbors,
                'center_three_layers_sasa_a2':round(sasa3[key][number],2),
                'center_layer_sasa_difference_a2':round(abs(sasa[key][number]-sasa3[key][number]),2),
                'ca_distance_upper_triangle_a':[round(float(matrix[i,j]),1) for i in range(len(neighbors)) for j in range(i+1,len(neighbors))],
                'local_residues':observations}
            chemical=[]
            basis=local_basis(chain[number],device).cpu().numpy()
            for n in neighbors:
                residue=chain[n];side=[a.coord for a in residue if a.name not in ('N','CA','C','O','OXT') and a.element!='H']
                centroid=np.mean(side,axis=0) if side else residue['CA'].coord
                direction=centroid-residue['CA'].coord;norm=float(np.linalg.norm(direction))
                chemical.append({'residue':n,'aa':AA[residue.resname],
                    'sidechain_centroid_offset_a':np.round((centroid-chain[number]['CA'].coord)@basis.T,2).tolist(),
                    'sidechain_direction':np.round((direction/norm)@basis.T,3).tolist() if norm>0 else [0.,0.,0.],
                    'sasa_a2':round(sasa[key][n],2),'charge_group':'positive' if AA[residue.resname] in 'KR' else 'negative' if AA[residue.resname] in 'DE' else 'other',
                    'hydrophobic_group':AA[residue.resname] in 'IVLFWYM'})
            by_target[key]['surface_chemistry']=chemical
            by_target[key]['surface_coordinate_frame']='Center-residue N-CA axis, orthogonalized C-CA axis, right-handed normal; invariant to global PDB rotation.'
            by_target[key]['cross_chain_pocket']=len({c for c,n in pocket})>1
        contexts.append({'site_id':site,'evidence_id':'surface_'+site,'center_residue':number,
                         'joint_selection_observation':row,'conformers':by_target})
    monomers=monomer_models(ROOT/'data/references/9A1A.cif')
    for index,monomer in enumerate(monomers,1):
        key='9A1A_m{}'.format(index);single=Structure(key);copied=copy.deepcopy(monomer);copied.id=0;single.add(copied)
        model=prepare_heavy(single,directory/(key+'.pdb'),free_c=True);chain=next(iter(model[0]))
        metadata[key]={'central_chain':chain.id,'chains':[chain.id],'target_pdb':str(directory/(key+'.pdb')),
            'capped_pdb':str(directory/(key+'.pdb')),'label':'finite monomer ensemble control',
            'source':'https://www.rcsb.org/structure/9A1A','terminal_chemistry':'unmodified full-length monomer reference',
            'scope':'Eight integrative reference models; finite snapshots of a dynamic ensemble; physiological states and modifications are only partly represented.'}
        for context in contexts:
            center=tensor(chain[context['center_residue']]['CA'].coord,device)
            pocket=[[chain.id,r.id[1]] for r in chain if float(torch.linalg.vector_norm(tensor(r['CA'].coord,device)-center))<11]
            context['conformers'][key]={'central_chain':chain.id,'pocket':pocket}
    write_json(inputs/'target_metadata.json',metadata)
    write_json(inputs/'surface_contexts.json',contexts)
    write_csv(inputs/'structure_inventory.csv',inventory)
    write_csv(inputs/'surface_selection.csv',scores)
    lc3=protein_model(ROOT/'data/references/2ZJD.pdb')
    selected=copy.deepcopy(lc3[0]['A']);selected.detach_parent();selected.id='R'
    for residue in list(selected):
        if residue.id[1]>120:
            selected.detach_child(residue.id)
    lc3_model=Structure('LC3B');lc3_model.add(Model(0));lc3_model[0].add(selected)
    lc3_model=prepare_heavy(lc3_model,directory/'LC3B.pdb',free_c=True)
    save_structure(add_caps(lc3_model,device,cap_c=False),directory/'LC3B_capped.pdb')
    reference=copy.deepcopy(lc3[0]['B']);reference.detach_parent();reference.id='K'
    combined=copy.deepcopy(lc3_model);combined[0].add(reference)
    save_structure(combined,inputs/'lc3_p62_reference.pdb')
    core=[r for r in reference if r.resname in ('TRP','THR','HIS','LEU')]
    w=next(r for r in reference if r.resname=='TRP')
    core=[reference[w.id[1]+i] for i in range(4)]
    receptor=list(lc3_model[0]['R']);pocket=[]
    motif=tensor([a.coord for r in core for a in r.get_atoms()],device)
    for residue in receptor:
        coordinates=tensor([a.coord for a in residue.get_atoms()],device)
        if float(torch.cdist(coordinates,motif).min())<7:
            pocket.append(['R',residue.id[1]])
    p62=''.join((ROOT/'data/references/SQSTM1_Q13501.fasta').read_text().splitlines()[1:])
    lir_dossier={'evidence_id':'human_p62_LIR','target_partner':'human LC3B',
       'reference_seed':p62[331:343],'reference_uniprot':'Q13501','human_residue_range':[332,343],
       'core_positions':[7,8,9,10],'core_sequence':'WTHL','structural_reference':'2ZJD:human LC3B / murine p62 peptide',
       'core_reference_ca':[r['CA'].coord.tolist() for r in core],
       'core_reference_sidechain_centroids':[np.mean([a.coord for a in r if a.name not in ('N','CA','C','O')],axis=0).tolist() for r in (core[0],core[3])],
       'pocket':pocket,'target_pdb':str(directory/'LC3B.pdb'),'capped_pdb':str(directory/'LC3B_capped.pdb'),
       'evidence_notes':['Published structures support two hydrophobic anchors and adjacent peptide contacts.',
                         'Flanking-residue modifications have context-dependent effects on LC3B recognition.',
                         'Human 20-residue WT and L341V peptides were measured at approximately 3.2 and 10.9 micromolar in a matched study; these are measurements of the 20-mer constructs.'],
       'matched_literature_controls':[{'id':'p62_WT_20','sequence':p62[331:351],'measured_kd_um':3.2},
                  {'id':'p62_L341V_20','sequence':p62[331:340]+'V'+p62[341:351],'measured_kd_um':10.9}],
       'source_urls':['https://www.rcsb.org/structure/2ZJD','https://pubmed.ncbi.nlm.nih.gov/37530436/',
                      'https://pmc.ncbi.nlm.nih.gov/articles/PMC4990988/'],
       'boundary':'soluble LC3B resolved domain through Gly120; missing N-terminal residues are boundary-capped; membrane PE and autophagosome dynamics are unmodeled'}
    write_json(inputs/'lir_reference_dossier.json',lir_dossier)
    binder_dossier={'objective':'MSA-conformer-preferring alpha-synuclein binder','MSA_targets':list(MSA),
       'controls':list(CONTROLS),'evidence_id':'five_structure_panel','structure_sources':metadata,
       'labels':'Target/control roles identify desired structural preference. Newly proposed peptides have unknown measured binding.',
       'site_selection':'Joint exposed-surface and shape comparison using all three targets and both controls; sites are structural hypotheses.'}
    binder_dossier['monomer_controls']=[k for k in metadata if k.startswith('9A1A_')]
    binder_dossier['experimental_references']=[{'evidence_id':'Wallace_2024','doi':'10.1039/D3SC06245G','pmid':'38846390',
        'source':'https://pubmed.ncbi.nlm.nih.gov/38846390/',
        'observation':'A designed peptide showed conformer-specific binding against other alpha-synuclein amyloids, tau and A-beta40 in the published study.',
        'scope':'Supports the feasibility of conformer recognition. Its experimental labels remain specific to the published peptides and tested fibrils; new MSA sequences have unknown measured binding.'}]
    binder_dossier['unavailable_evidence']=['Systematic mapped antibody epitopes and competition or mutagenesis measurements for the current four surfaces.',
        'Other protein-fibril off-target binding, physiological monomer populations, cellular uptake and actual degradation.']
    write_json(inputs/'binder_dossier.json',binder_dossier)
    print('PREPARED_TARGETS_AND_REFERENCES',len(contexts),'sites',flush=True)
