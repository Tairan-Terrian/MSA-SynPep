"""Continuous L-peptide construction and GPU geometric calculations."""

import copy
import math
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
import PeptideBuilder
from PeptideBuilder import Geometry

from core import ROOT

AA = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
              'ARNDCQEGHILKMFPSTWYV'))
BB = ('N','CA','C')
CHI_ATOMS={
 'R':[('N','CA','CB','CG'),('CA','CB','CG','CD'),('CB','CG','CD','NE'),('CG','CD','NE','CZ')],
 'N':[('N','CA','CB','CG'),('CA','CB','CG','OD1')],
 'D':[('N','CA','CB','CG'),('CA','CB','CG','OD1')],
 'C':[('N','CA','CB','SG')],
 'Q':[('N','CA','CB','CG'),('CA','CB','CG','CD'),('CB','CG','CD','OE1')],
 'E':[('N','CA','CB','CG'),('CA','CB','CG','CD'),('CB','CG','CD','OE1')],
 'H':[('N','CA','CB','CG'),('CA','CB','CG','ND1')],
 'I':[('N','CA','CB','CG1'),('CA','CB','CG1','CD1')],
 'L':[('N','CA','CB','CG'),('CA','CB','CG','CD1')],
 'K':[('N','CA','CB','CG'),('CA','CB','CG','CD'),('CB','CG','CD','CE'),('CG','CD','CE','NZ')],
 'M':[('N','CA','CB','CG'),('CA','CB','CG','SD'),('CB','CG','SD','CE')],
 'F':[('N','CA','CB','CG'),('CA','CB','CG','CD1')],
 'S':[('N','CA','CB','OG')], 'T':[('N','CA','CB','OG1')],
 'W':[('N','CA','CB','CG'),('CA','CB','CG','CD1')],
 'Y':[('N','CA','CB','CG'),('CA','CB','CG','CD1')], 'V':[('N','CA','CB','CG1')]}


def read_structure(path):
    return PDBParser(QUIET=True).get_structure(Path(path).stem, str(path))


def save_structure(structure,path):
    writer=PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))


def tensor(coords,device):
    return torch.as_tensor(np.asarray(coords),dtype=torch.float64,device=device)


def dihedral(a,b,c,d):
    b0=a-b
    b1=c-b
    b2=d-c
    b1=b1/torch.linalg.vector_norm(b1,dim=-1,keepdim=True)
    v=b0-(b0*b1).sum(-1,keepdim=True)*b1
    w=b2-(b2*b1).sum(-1,keepdim=True)*b1
    return torch.atan2((torch.cross(b1,v,dim=-1)*w).sum(-1),(v*w).sum(-1))*180/math.pi


def angles_from_chain(chain,device):
    residues=[r for r in chain if r.resname in AA]
    xyz=tensor([[r[name].coord for name in BB] for r in residues],device)
    phi=torch.full((len(residues),),-120.,device=device,dtype=torch.float64)
    psi=torch.full_like(phi,140.)
    phi[1:]=dihedral(xyz[:-1,2],xyz[1:,0],xyz[1:,1],xyz[1:,2])
    psi[:-1]=dihedral(xyz[:-1,0],xyz[:-1,1],xyz[:-1,2],xyz[1:,0])
    return phi,psi


def category(sequence):
    result=[]
    for index,aa in enumerate(sequence):
        result.append(1 if aa=='G' else 3 if aa=='P' else
                      4 if index+1<len(sequence) and sequence[index+1]=='P' else
                      5 if aa in 'IV' else 0)
    return result


class RamaReference:
    def __init__(self,device):
        data=np.load(ROOT/'data/references/ramachandran_probabilities.npz')
        self.tables=tensor(data['values'],device)
        self.device=device
        self.axes=tensor(data['angles'],device)
        self.favored=[]
        for table in self.tables:
            ids=torch.nonzero(table>=.02)
            self.favored.append(torch.stack((self.axes[ids[:,0]],self.axes[ids[:,1]]),-1))
        native=read_structure(ROOT/'data/references/2ZJD.pdb')[0]['A']
        self.beta_sign={}
        for aa,names in [('THR',('CA','OG1','CG2')),('ILE',('CA','CG1','CG2'))]:
            vectors=tensor([[r[n].coord-r['CB'].coord for n in names] for r in native if r.resname==aa],device)
            volumes=(torch.cross(vectors[:,0],vectors[:,1],dim=-1)*vectors[:,2]).sum(-1)
            self.beta_sign[aa]=float(volumes.median().sign())

    def scores(self,sequence,phi,psi):
        x=((phi+179)/2)%180
        y=((psi+179)/2)%180
        i=x.floor().long();j=y.floor().long()
        dx=x-i;dy=y-j
        k=torch.tensor(category(sequence),device=self.device)
        return ((1-dx)*(1-dy)*self.tables[k,i,j]+dx*(1-dy)*self.tables[k,(i+1)%180,j]+
                (1-dx)*dy*self.tables[k,i,(j+1)%180]+dx*dy*self.tables[k,(i+1)%180,(j+1)%180])

    def project(self,sequence,phi,psi):
        single=phi.ndim==1
        phi=phi[None] if single else phi;psi=psi[None] if single else psi
        result=torch.stack((phi,psi),-1).clone()
        probabilities=self.scores(sequence,phi,psi)
        categories=torch.tensor(category(sequence),device=self.device)
        for k in range(6):
            threshold=.0005 if k==0 else .001
            indices=torch.nonzero((probabilities<threshold)&(categories[None]==k))
            if len(indices):
                current=result[indices[:,0],indices[:,1]]
                delta=(self.favored[k][None]-current[:,None]+180)%360-180
                nearest=delta.square().sum(-1).argmin(-1)
                result[indices[:,0],indices[:,1]]=self.favored[k][nearest]
        return (result[0,:,0],result[0,:,1]) if single else (result[:,:,0],result[:,:,1])

    def sample(self,sequence,batch,generator):
        pairs=[]
        for k in category(sequence):
            choices=torch.randint(len(self.favored[k]),(batch,),device=self.device,generator=generator)
            pairs.append(self.favored[k][choices])
        values=torch.stack(pairs,1)
        return values[:,:,0],values[:,:,1]


def align(source,target):
    s=source-source.mean(dim=-2,keepdim=True)
    t=target-target.mean(dim=-2,keepdim=True)
    u,_,vh=torch.linalg.svd(s.transpose(-1,-2)@t)
    signs=torch.ones(source.shape[:-2]+(3,),device=source.device,dtype=source.dtype)
    signs[...,-1]=torch.linalg.det(u@vh)
    rotation=u@torch.diag_embed(signs)@vh
    translation=target.mean(-2)-torch.matmul(source.mean(-2).unsqueeze(-2),rotation).squeeze(-2)
    return rotation,translation


def place(a,b,c,length,angle,torsion):
    bc=c-b
    bc=bc/torch.linalg.vector_norm(bc,dim=-1,keepdim=True)
    normal=torch.cross(b-a,bc,dim=-1)
    normal=normal/torch.linalg.vector_norm(normal,dim=-1,keepdim=True)
    side=torch.cross(normal,bc,dim=-1)
    angle=angle*math.pi/180
    torsion=torsion*math.pi/180
    return c+length*(-bc*math.cos(angle)+torch.sin(torch.as_tensor(angle,device=c.device))*
           (side*torch.cos(torsion)[...,None]+normal*torch.sin(torsion)[...,None]))


def backbone(sequence,phi,psi):
    batch=phi.shape[0]
    device=phi.device
    geo=Geometry.geometry(sequence[0])
    n=torch.zeros((batch,3),device=device,dtype=phi.dtype)
    ca=n.clone();ca[:,0]=geo.CA_N_length
    c=ca.clone()
    c[:,0]+=-geo.CA_C_length*math.cos(geo.N_CA_C_angle*math.pi/180)
    c[:,1]=geo.CA_C_length*math.sin(geo.N_CA_C_angle*math.pi/180)
    values=[torch.stack((n,ca,c),1)]
    for i in range(1,len(sequence)):
        geo=Geometry.geometry(sequence[i])
        n=place(n,ca,c,geo.peptide_bond,geo.CA_C_N_angle,psi[:,i-1])
        ca=place(values[-1][:,1],values[-1][:,2],n,geo.CA_N_length,geo.C_N_CA_angle,torch.full_like(phi[:,i],180.))
        c=place(values[-1][:,2],n,ca,geo.CA_C_length,geo.N_CA_C_angle,phi[:,i])
        values.append(torch.stack((n,ca,c),1))
    return torch.stack(values,1)


def rotamers(chain,device):
    result={}
    residues=[r for r in chain if r.resname in AA]
    for i,residue in enumerate(residues):
        aa=AA[residue.resname]
        if aa in CHI_ATOMS:
            result[i]=[float(dihedral(*tensor([residue[name].coord for name in group],device))) for group in CHI_ATOMS[aa]]
    return result


def peptide(sequence,phi,psi,device,target_ca=None,sidechain_angles=None):
    geometries=[]
    for i,aa in enumerate(sequence):
        geo=Geometry.geometry(aa)
        if i>0:
            geo.phi=float(phi[i]);geo.psi_im1=float(psi[i-1]);geo.omega=180.
        if sidechain_angles is not None and i in sidechain_angles:
            for atoms,angle in zip(CHI_ATOMS[aa],sidechain_angles[i]):
                key='_'.join(atoms)+'_diangle'
                delta=angle-getattr(geo,key)
                prefix='_'.join(atoms[:3])+'_'
                for attribute in list(geo.__dict__):
                    if attribute.startswith(prefix) and attribute.endswith('_diangle'):
                        setattr(geo,attribute,getattr(geo,attribute)+delta)
        geometries.append(geo)
    structure=PeptideBuilder.initialize_res(geometries[0])
    for geo in geometries[1:]:
        PeptideBuilder.add_residue(structure,geo)
    PeptideBuilder.add_terminal_OXT(structure)
    chain=structure[0]['A']
    chain.id='K'
    if target_ca is not None:
        coords=tensor([r['CA'].coord for r in chain],device)
        rotation,translation=align(coords,target_ca)
        atoms=list(chain.get_atoms())
        fitted=tensor([a.coord for a in atoms],device)@rotation+translation
        for atom,coordinate in zip(atoms,fitted.detach().cpu().numpy()):
            atom.coord=coordinate
    return chain


def quaternion_rotation(q):
    w,x,y,z=(q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)).unbind(-1)
    matrix=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
          2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
          2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)
    return matrix.reshape(-1,3,3).transpose(-1,-2)


def standardize(sequence,raw_chain,reference,device):
    """One bounded torsion fit to C-alpha; no receptor-position search."""
    raw_ca=tensor([r['CA'].coord for r in raw_chain if r.resname in AA],device)
    phi,psi=angles_from_chain(raw_chain,device)
    phi,psi=reference.project(sequence,phi,psi)
    length=len(sequence)
    local=torch.linalg.vector_norm(raw_ca[2:]-raw_ca[:-2],dim=-1)
    helix=torch.cat((local[:1],local,local[-1:]))<6.2
    traced_phi=torch.where(helix,torch.full_like(phi,-60),torch.full_like(phi,-120))
    traced_psi=torch.where(helix,torch.full_like(psi,-40),torch.full_like(psi,140))
    phis=torch.stack((phi,traced_phi,torch.full_like(phi,-60),torch.full_like(phi,-120)))
    psis=torch.stack((psi,traced_psi,torch.full_like(psi,-40),torch.full_like(psi,140)))
    phis,psis=reference.project(sequence,phis,psis)
    with torch.no_grad():
        bb=backbone(sequence,phis,psis);rot,shift=align(bb[:,:,1],raw_ca)
        fitted=bb[:,:,1]@rot+shift[:,None]
        selected=int((fitted-raw_ca).square().sum(-1).mean(-1).argmin())
    phi=phis[selected].clone().requires_grad_();psi=psis[selected].clone().requires_grad_()
    opt=torch.optim.Adam([phi,psi],lr=3.)
    for step in range(30):
        ca=backbone(sequence,phi[None],psi[None])[0,:,1]
        rot,shift=align(ca,raw_ca);fitted=ca@rot+shift
        fidelity=(fitted-raw_ca).square().sum(-1).mean()
        loss=fidelity+.02*(-reference.scores(sequence,phi,psi).clamp_min(1e-6).log()).mean()
        opt.zero_grad();loss.backward();opt.step()
        if step%5==4:
            with torch.no_grad():
                projected_phi,projected_psi=reference.project(sequence,phi,psi)
                phi.copy_(projected_phi);psi.copy_(projected_psi)
    with torch.no_grad():phi,psi=reference.project(sequence,phi,psi)
    chain=peptide(sequence,phi,psi,device,target_ca=raw_ca,sidechain_angles=rotamers(raw_chain,device))
    final_ca=tensor([r['CA'].coord for r in chain],device)
    return chain,float((final_ca-raw_ca).square().sum(-1).mean().sqrt())


def focused_adjustment(chain,receptor,device,steps=20):
    """A bounded local rigid adjustment for the prespecified review pool."""
    atoms=list(chain.get_atoms());points=tensor([a.coord for a in atoms],device).float()[None]
    target=tensor([a.coord for a in receptor.get_atoms() if a.element!='H'],device).float()
    center=points.mean(1,keepdim=True)
    q=torch.tensor([[1.,0.,0.,0.]],device=device,requires_grad=True)
    shift=torch.zeros((1,1,3),device=device,requires_grad=True)
    opt=torch.optim.Adam([q,shift],lr=.035)
    for _ in range(steps):
        moved=(points-center)@quaternion_rotation(q)+center+shift
        loss=torch.relu(2.3-torch.cdist(moved,target[None])).square().sum()+.2*(moved-points).square().mean()
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            q[:,0]=1.;q[:,1:].clamp_(-.06,.06);shift.clamp_(-1.0,1.0)
    with torch.no_grad():moved=(points-center)@quaternion_rotation(q)+center+shift
    displacement=float((moved-points).square().sum(-1).mean().sqrt())
    for atom,point in zip(atoms,moved[0].detach().cpu().numpy()):atom.coord=point
    return displacement


def attachment_metrics(chain,receptor,device):
    residues=[r for r in chain if r.resname in AA]
    xyz=tensor([a.coord for c in receptor if c.id!='K' for a in c.get_atoms() if a.element!='H'],device)
    angles=torch.arange(48,device=device,dtype=torch.float64)*2.3999632297
    z=1-2*(torch.arange(48,device=device,dtype=torch.float64)+.5)/48
    radius=(1-z*z).sqrt();sphere=torch.stack((radius*angles.cos(),radius*angles.sin(),z),-1)
    result={}
    for name,residue,atom in [('N',residues[0],'N'),('C',residues[-1],'C')]:
        tip=tensor(residue[atom].coord,device);ca=tensor(residue['CA'].coord,device)
        direction=(tip-ca)/torch.linalg.vector_norm(tip-ca)
        endpoint=tip+3*direction
        result[name+'_end_clearance_a']=round(float(torch.cdist(endpoint[None],xyz).min()),3)
        result[name+'_end_open_fraction']=round(float((torch.cdist(tip[None]+4*sphere,xyz).min(-1).values>2.5).double().mean()),3)
        result[name+'_end_receptor_atom_contacts']=int((torch.linalg.vector_norm(xyz-tip,dim=-1)<4.5).sum())
    result['preferred_attachment_end']=max(('N','C'),key=lambda end:(result[end+'_end_open_fraction'],result[end+'_end_clearance_a']))
    return result


def geometry_metrics(chain,reference,device):
    residues=[r for r in chain if r.resname in AA]
    sequence=''.join(AA[r.resname] for r in residues)
    xyz=tensor([[r[name].coord for name in BB] for r in residues],device)
    phi,psi=angles_from_chain(chain,device)
    probs=reference.scores(sequence,phi,psi)
    thresholds=torch.tensor([.0005 if k==0 else .001 for k in category(sequence)],device=device)
    outliers=int((probs[1:-1]<thresholds[1:-1]).sum())
    cn=torch.linalg.vector_norm(xyz[1:,0]-xyz[:-1,2],dim=-1)
    omega=dihedral(xyz[:-1,1],xyz[:-1,2],xyz[1:,0],xyz[1:,1])
    flips=0;beta_flips=0
    for residue in residues:
        if residue.resname!='GLY':
            n,ca,c,cb=tensor([residue[a].coord for a in ('N','CA','C','CB')],device)
            volume=torch.dot(torch.cross(n-ca,c-ca,dim=-1),cb-ca)
            flips+=int(float(volume)<0)
        if residue.resname in ('THR','ILE'):
            names=('CA','OG1','CG2') if residue.resname=='THR' else ('CA','CG1','CG2')
            vectors=tensor([residue[n].coord-residue['CB'].coord for n in names],device)
            volume=torch.dot(torch.cross(vectors[0],vectors[1],dim=-1),vectors[2])
            beta_flips+=int(float(volume)*reference.beta_sign[residue.resname]<0)
    deviation=float((cn-1.33).abs().max())
    omega_deviation=float((180-omega.abs()).max())
    atoms=[(i,a) for i,r in enumerate(residues) for a in r if a.element!='H']
    coords=tensor([a.coord for i,a in atoms],device)
    ids=torch.tensor([i for i,a in atoms],device=device)
    self_clashes=int(((torch.cdist(coords,coords)<2)&((ids[:,None]-ids[None,:]).abs()>=2)).sum())//2
    return {'rama_outliers':outliers,'chirality_flips':flips,'beta_stereo_flips':beta_flips,'self_severe_clashes':self_clashes,'cn_max_deviation_a':round(deviation,4),
            'omega_max_deviation_deg':round(omega_deviation,3),
            'backbone_geometry_eligible':outliers==0 and flips==0 and beta_flips==0 and self_clashes==0 and deviation<.05 and omega_deviation<10}


def atom_data(chains,device):
    coordinates=[];residues=[];polar=[];positive=[];negative=[];native=[]
    index=0
    for chain in chains:
        for residue in chain:
            for atom in residue:
                if atom.element!='H':
                    coordinates.append(atom.coord);residues.append(index)
                    native.append(residue.resname in AA)
                    polar.append(atom.element in ('N','O'))
                    positive.append(residue.resname in ('LYS','ARG') and atom.name in ('NZ','NH1','NH2','NE'))
                    negative.append(residue.resname in ('ASP','GLU') and atom.name in ('OD1','OD2','OE1','OE2'))
            index+=1
    return {'xyz':tensor(coordinates,device),'residue':torch.tensor(residues,device=device),
            'native':torch.tensor(native,device=device),
            'polar':torch.tensor(polar,device=device),'positive':torch.tensor(positive,device=device),
            'negative':torch.tensor(negative,device=device)}


def interface_metrics(ligand,receptor,length):
    distances=torch.cdist(ligand['xyz'],receptor['xyz'])
    pairs=ligand['residue'][:,None]*(int(receptor['residue'].max())+1)+receptor['residue'][None,:]
    native=ligand['native'][:,None]&receptor['native'][None,:]
    contacts=int(torch.unique(pairs[(distances<4.5)&native]).numel())
    polar=((distances<3.5)&native&ligand['polar'][:,None]&receptor['polar'][None,:]).sum()
    opposite=((ligand['positive'][:,None]&receptor['negative'][None,:])|
              (ligand['negative'][:,None]&receptor['positive'][None,:]))&(distances<4.5)
    return {'severe_clashes':int((distances<2).sum()),'contacts_per_residue':round(contacts/length,3),
            'polar_proximity_per_residue':round(float(polar)/length,3),
            'opposite_charge_pairs_per_residue':round(float(opposite.sum())/length,3)}


def add_caps(structure,device,cap_c=True):
    result=copy.deepcopy(structure)
    for chain in result[0]:
        residues=[r for r in chain if r.resname in AA]
        first,last=residues[0],residues[-1]
        c,ca,n=tensor([first[a].coord for a in ('C','CA','N')],device)
        prev=place(c,ca,n,1.33,121.7,torch.tensor(-120.,device=device))
        oxygen=place(ca,n,prev,1.23,123.,torch.tensor(180.,device=device))
        methyl=place(ca,n,prev,1.52,116.2,torch.tensor(0.,device=device))
        cap=Residue((' ',first.id[1]-1,' '),'ACE',' ')
        for name,point,element in [('CH3',methyl,'C'),('C',prev,'C'),('O',oxygen,'O')]:
            cap.add(Atom(name,point.detach().cpu().numpy(),0.,1.,' ',name.rjust(4),0,element=element))
        chain.insert(0,cap)
        if cap_c:
            n,ca,c=tensor([last[a].coord for a in ('N','CA','C')],device)
            following=place(n,ca,c,1.33,116.2,torch.tensor(140.,device=device))
            methyl=place(ca,c,following,1.46,121.7,torch.tensor(180.,device=device))
            cap=Residue((' ',last.id[1]+1,' '),'NME',' ')
            for name,point,element in [('N',following,'N'),('CH3',methyl,'C')]:
                cap.add(Atom(name,point.detach().cpu().numpy(),0.,1.,' ',name.rjust(4),0,element=element))
            chain.add(cap)
    return result
