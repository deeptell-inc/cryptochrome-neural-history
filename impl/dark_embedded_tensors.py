"""State-specific electronic tensors in nonuniform saved oxidized protein environments.

Frozen capped flavin geometry, full force-field charges of selected non-FAD
residues, no environment back-polarization or reduced-state equilibration.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from scipy import constants as c
from . import dark_electronic_tensors as base
from . import dark_electronic_response as response
ROOT=base.ROOT;OUT=ROOT/'data/dark-embedded-tensors'
SEEDS=(2026091521,2026091522,2026091523)

def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rotation(x,y):
    u,_,vt=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)))
    return u@np.diag([1,1,np.linalg.det(u@vt)])@vt

def coulomb_efg(origin,positions,charges):
    d=np.asarray(positions)-origin;r=np.linalg.norm(d,axis=1);u=d/r[:,None]
    return np.einsum('n,nij->ij',charges/r**3,3*u[:,:,None]*u[:,None,:]-np.eye(3))

def hessian_hfc_factor():
    return base.hfc_si_prefactor()*3/(8*np.pi)

def environment(seed,frame,radius_A):
    folder=ROOT/f'data/dark-md-coefficients/seed_{seed}';z=np.load(folder/'trajectory.npz')
    meta=json.loads((folder/'metadata.json').read_text());index=list(z['check_frames']).index(frame);x=z['check_positions_nm'][index]*10
    atoms=json.loads((ROOT/'data/dcry-mobile-model/topology.json').read_text())['atoms'];fad={a['name']:a['index'] for a in atoms if a['resname']=='FAD'}
    spec=json.loads((base.OUTPUT/'E_neutral_SQ/input.json').read_text());qm={a['label']:np.array(a['xyz_A']) for a in spec['atoms']}
    names=meta['ring_names'];ref=np.array([qm[{'C4A':'C4X','C5A':'C5X'}.get(n,n)] for n in names]);mobile=x[[fad[n] for n in names]]
    Q=rotation(mobile,ref);center=mobile.mean(0);refcenter=ref.mean(0)
    ids=z['external_ids'];charges=z['external_charges_e'];delta=x[ids]-center;L=np.diag(z['box_nm'])*10;delta-=L*np.round(delta/L)
    xyz=delta@Q+refcenter
    # Include complete residues/waters if ANY atom lies within radius of a ring atom.
    distance=np.min(np.linalg.norm(xyz[:,None,:]-ref[None,:,:],axis=2),axis=1)
    keys=[(atoms[int(i)]['chain'],atoms[int(i)]['residue'],atoms[int(i)]['resname']) for i in ids]
    selected={key for key,d in zip(keys,distance) if d<radius_A};mask=np.array([k in selected for k in keys])
    geom=np.array([a['xyz_A'] for a in spec['atoms']]);closest=float(np.min(np.linalg.norm(xyz[mask,None,:]-geom[None,:,:],axis=2)))
    info=dict(seed=seed,frame=frame,time_ps=float(z['time_ps'][frame]),radius_A=radius_A,selection='whole non-FAD residues/waters with any atom within radius of reference flavin ring',
        charge_count=int(mask.sum()),residue_count=len(selected),net_MM_charge_e=float(charges[mask].sum()),minimum_MM_to_QM_atom_A=closest,
        ring_alignment_RMSD_A=float(np.sqrt(np.mean(np.sum(((mobile-center)@Q+refcenter-ref)**2,axis=1)))),
        state='oxidized MD environment; NOT a reduced-state equilibrium ensemble',QM_geometry='original frozen capped fragment; environment rigidly registered by ring fit',
        omissions=['remaining FAD ribityl/phosphate/adenine charge distribution','MM polarization','QM geometry relaxation','periodic Ewald replicas','state-conditioned environment relaxation'])
    return xyz[mask],charges[mask],ids[mask],info

def case_name(label,seed,frame,radius,scale):return f'{label}__s{seed}__f{frame}__r{radius:g}__l{scale:g}'

def run(label,seed,frame,radius,scale,properties_only=False):
    from pyscf import qmmm,scf
    from pyscf.prop.hfc import uhf as hfc
    start=time.monotonic();name=case_name(label,seed,frame,radius,scale);folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise FileExistsError(folder/'result.json')
    xyz,charges,ids,env=environment(seed,frame,radius);save(folder/'environment.json',env)
    np.savez_compressed(folder/'environment.npz',xyz_A=xyz,charges_e=charges,external_atom_ids=ids)
    mf,meta,nuclei=response.load_mf(label,folder);dm0=mf.make_rdm1();mol=mf.mol
    mf=qmmm.mm_charge(mf,xyz,charges*scale,unit='Angstrom');mf.conv_tol=1e-11;mf.conv_tol_grad=1e-7;mf.max_cycle=120
    # MM potential is static in this calculation: cache hcore rather than recomputing every SCF step.
    hc=mf.get_hcore();mf.get_hcore=lambda *args,**kwargs:hc
    mf.chkfile=str(folder/'scf.chk')
    if properties_only:
        mf.__dict__.update(scf.chkfile.load(mf.chkfile,'scf'));energy=mf.e_tot
        checkpoint_gradient=float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ)))
        if not np.isfinite(checkpoint_gradient) or checkpoint_gradient>=1e-6:
            raise RuntimeError('Checkpoint does not satisfy the SCF stationarity check')
        mf.converged=True
    else:energy=mf.kernel(dm0=dm0)
    if not mf.converged:raise RuntimeError('Embedded SCF not converged: '+name)
    dm=mf.make_rdm1();total=dm.sum(0) if mol.spin else dm;spin=dm[0]-dm[1] if mol.spin else np.zeros_like(dm)
    Vqm,A=response.properties(mf,nuclei,dm);V0,A0=response.properties(mf,nuclei,dm0)
    extxyz=xyz/(base.A0*1e10);Vmm=np.array([coulomb_efg(mol.atom_coord(i),extxyz,charges*scale) for i in nuclei]);V=Vqm+Vmm
    # FC contact prefactor contains 8*pi/3; the Coulomb Hessian formula does not.
    steps=(.001,.0005) if 'pcJ' in label else (.002,.001)
    independent=base.independent_properties(mol,total,spin,nuclei,hessian_hfc_factor(),steps=steps)
    checks=[];rows=[]
    for n,(idx,nameN) in enumerate(zip(nuclei,('N5','N10'))):
        other=extxyz-mol.atom_coord(idx)
        def potential(d):return np.sum(charges*scale/np.linalg.norm(other-d,axis=1))
        H=base.finite_hessian(potential,np.zeros(3),.001)
        vcheck=np.array(independent[n]['EFG_richardson_au'])+base.traceless(H)
        checks.append(dict(atom=nameN,EFG_relative_error=float(np.linalg.norm(vcheck-V[n])/np.linalg.norm(V[n])),
            HFC_relative_error=None if A is None else float(np.linalg.norm(np.array(independent[n]['HFC_richardson_MHz'])-A[n])/np.linalg.norm(A[n]))))
        rows.append(dict(atom=nameN,isotope='14N',EFG_total_au=V[n].tolist(),EFG_QM_au=Vqm[n].tolist(),EFG_MM_au=Vmm[n].tolist(),
            EFG_gas_au=V0[n].tolist(),EFG_total_principal=base.principal_efg(V[n]),
            induced_density_EFG_relative_change=float(np.linalg.norm(Vqm[n]-V0[n])/np.linalg.norm(V0[n])),
            full_EFG_relative_change=float(np.linalg.norm(V[n]-V0[n])/np.linalg.norm(V0[n])),
            HFC_MHz=None if A is None else A[n].tolist(),gas_HFC_MHz=None if A0 is None else A0[n].tolist(),
            isotropic_HFC_MHz=None if A is None else float(np.trace(A[n])/3),gas_isotropic_HFC_MHz=None if A0 is None else float(np.trace(A0[n])/3)))
    S=mf.get_ovlp();electron_number=float(np.einsum('ij,ji',total,S));grad=float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ)))
    assert abs(electron_number-mol.nelectron)<1e-7
    assert np.isfinite(grad) and grad<1e-6,grad
    assert max(x['EFG_relative_error'] for x in checks)<1e-3,checks
    assert max(x['HFC_relative_error'] or 0 for x in checks)<1e-3,checks
    np.savez_compressed(folder/'density.npz',dm=dm,hcore=hc,overlap=S,EFG_total_au=V,HFC_MHz=np.array([]) if A is None else A)
    (folder/'mol.json').write_text(mol.dumps())
    out=dict(label=label,state=meta['state_id'],case=name,seed=seed,frame=frame,radius_A=radius,MM_charge_scale=scale,basis=meta['basis'],
        method='B3LYP/grid4/DF fixed-geometry electrostatic embedding; point MM charges; no electronic environment back-response',
        converged=True,properties_from_checkpoint=properties_only,gradient_norm=grad,electron_count=electron_number,spin_square=float(mf.spin_square()[0]) if mol.spin else 0.,energy_Hartree=float(energy),
        energy_scope='SCF diagnostic only; not a redox free energy or reaction rate',nuclei=rows,checks=checks,elapsed_s=time.monotonic()-start,
        native_calibrated=False,independent_biological_replicates=0,relaxation_spectrum=None,native_neural_gain=None,
        uncertainty='configuration, basis and cutoff contrasts; not physiological posterior intervals')
    save(folder/'result.json',out);save(folder/'independent_checks.json',independent)
    print(name, 'done',round(out['elapsed_s'],1),'s', [(r['atom'],r['isotropic_HFC_MHz']) for r in rows],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='E_neutral_SQ');p.add_argument('--seed',type=int,default=SEEDS[0]);p.add_argument('--frame',type=int,default=0);p.add_argument('--radius',type=float,default=12);p.add_argument('--scale',type=float,default=1);p.add_argument('--properties-only',action='store_true');a=p.parse_args();run(a.label,a.seed,a.frame,a.radius,a.scale,a.properties_only)
