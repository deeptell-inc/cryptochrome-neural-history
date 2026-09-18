"""Separate nitrogen contraction error from embedding truncation sensitivity.

Frozen neutral-SQ flavin only. Not native relaxation or a measured neural gain.
"""
from pathlib import Path
import argparse,json,time,copy
import numpy as np
from . import dark_electronic_tensors as b
from . import dark_electronic_response as response
from . import dark_embedded_tensors as embedded
ROOT=b.ROOT
OUT=ROOT/'data/dark-basis-resolution'
BASIS=('usvpdN','upcj2N','utzvpdN')

def nitrogen_basis(name):
    from pyscf import gto
    if name=='upcj2N':
        basis=gto.basis.parse((ROOT/'evidence/coefficient-identification/pcJ-2_N.nwchem').read_text())
    else: basis=gto.basis.load('def2-tzvpd' if name=='utzvpdN' else 'def2-svpd','N')
    return gto.uncontract(basis)

def run(name,environment='gas',no_df=False,properties_only=False):
    from pyscf import gto,dft,scf,qmmm,lib
    start=time.monotonic();lib.num_threads(1)
    label=f'{name}__{environment}'+('__noDF' if no_df else '')
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise FileExistsError(folder/'result.json')
    source=b.OUTPUT/('E_neutral_SQ_pcJ2N' if name=='upcj2N' else 'E_neutral_SQ')
    spec=json.loads((source/'input.json').read_text());ids=b.validate_input(spec)
    old=gto.loads((source/'mol.json').read_text());basis=copy.deepcopy(old._basis);basis['N']=nitrogen_basis(name)
    mol=gto.M(atom=old.atom,unit=old.unit,charge=old.charge,spin=old.spin,basis=basis,
              max_memory=2200,verbose=4,output=str(folder/'engine.log'))
    # Verify unchanged coordinates before projecting the reference density.
    np.testing.assert_allclose(mol.atom_coords(),old.atom_coords(),atol=1e-12)
    mf=dft.UKS(mol)
    if not no_df:mf=mf.density_fit(auxbasis='def2-universal-jkfit')
    xyz=np.empty((0,3));charges=np.empty(0);env={'boundary':'gas','charge_count':0}
    if environment!='gas':
        radius=12 if environment=='r12' else 1000
        xyz,charges,atomids,env=embedded.environment(2026091521,0,radius)
        if environment=='full':
            z=np.load(ROOT/'data/dark-md-coefficients/seed_2026091521/trajectory.npz')
            assert set(atomids)==set(z['external_ids'])
        mf=qmmm.mm_charge(mf,xyz,charges,unit='Angstrom')
    mf.xc='b3lyp';mf.grids.level=4;mf.conv_tol=1e-11;mf.conv_tol_grad=1e-7;mf.max_cycle=120
    hc=mf.get_hcore();mf.get_hcore=lambda *a,**kw:hc;mf.chkfile=str(folder/'scf.chk')
    if properties_only:
        mf.__dict__.update(scf.chkfile.load(mf.chkfile,'scf'))
        grad=np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ))
        if not np.isfinite(grad) or grad>=1e-6:raise RuntimeError('Nonstationary checkpoint')
        mf.converged=True
    else:
        gas=OUT/f'{name}__gas'
        if environment!='gas' and (gas/'result.json').exists():
            guess=np.load(gas/'density.npz')['dm']
        else:
            density=np.load(source/'density.npz')['dm']
            guess=np.array([scf.addons.project_dm_nr2nr(old,d,mol) for d in density])
        mf.kernel(dm0=guess)
    if not mf.converged:raise RuntimeError('SCF not converged')
    dm=mf.make_rdm1();total=dm.sum(0);spin=dm[0]-dm[1]
    V,A=response.properties(mf,ids,dm)
    direct=np.array([embedded.coulomb_efg(mol.atom_coord(i),xyz/(b.A0*1e10),charges) for i in ids]) if len(charges) else np.zeros_like(V)
    independent=b.independent_properties(mol,total,spin,ids,embedded.hessian_hfc_factor(),steps=(.0005,.00025))
    rho=np.einsum('pi,ij,pj->p',mol.eval_gto('GTOval',mol.atom_coords()[ids]),spin,mol.eval_gto('GTOval',mol.atom_coords()[ids]))
    rows=[]
    for n,atom in enumerate(('N5','N10')):
        rows.append(dict(atom=atom,HFC_MHz=A[n].tolist(),isotropic_HFC_MHz=float(np.trace(A[n])/3),
          EFG_QM_au=V[n].tolist(),EFG_MM_au=direct[n].tolist(),EFG_total_au=(V[n]+direct[n]).tolist(),
          FC_SI_MHz=float(b.hfc_si_prefactor()*rho[n]),
          FC_relative_error=float(abs(b.hfc_si_prefactor()*rho[n]-np.trace(A[n])/3)/abs(np.trace(A[n])/3)),
          HFC_Hessian_relative_error=float(np.linalg.norm(A[n]-independent[n]['HFC_richardson_MHz'])/np.linalg.norm(A[n])),
          EFG_Hessian_relative_error=float(np.linalg.norm(V[n]-independent[n]['EFG_richardson_au'])/np.linalg.norm(V[n]))))
    result=dict(label=label,basis=name,environment=environment,density_fitting=not no_df,
      state='E_neutral_SQ',species='4GU5 dCRY-derived capped flavin in fixed geometry',isotope='14N',
      converged=bool(mf.converged),gradient_norm=float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ))),
      nao=mol.nao,energy_Hartree=float(mf.e_tot),spin_square=float(mf.spin_square()[0]),
      electron_number=float(np.einsum('ij,ji',total,mf.get_ovlp())),nuclei=rows,elapsed_s=time.monotonic()-start,
      geometry_optimized=False,native_calibrated=False,native_T1_s=None,native_T2_s=None,native_gain=None,
      omissions=['SCF stability search','complete basis limit','periodic Ewald images','FAD tail','MM polarization','state-conditioned dynamics'],
      interpretation='basis and finite-cell minimum-image electrostatic-boundary diagnostics, not native identification')
    embedded.save(folder/'environment.json',env);embedded.save(folder/'input.json',spec)
    embedded.save(folder/'independent.json',independent)
    embedded.save(folder/'basis.json',basis)
    (folder/'mol.json').write_text(mol.dumps())
    np.savez_compressed(folder/'density.npz',dm=dm,overlap=mf.get_ovlp())
    embedded.save(folder/'diagnostic.json',result)
    assert abs(result['electron_number']-135)<1e-7 and result['gradient_norm']<1e-6
    for row in rows:
        assert row['FC_relative_error']<1e-7
        assert row['HFC_Hessian_relative_error']<1e-3
        assert row['EFG_Hessian_relative_error']<1e-4
    embedded.save(folder/'result.json',result)
    print(label,[(r['atom'],r['isotropic_HFC_MHz']) for r in rows],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--basis',choices=BASIS,required=True)
    p.add_argument('--environment',choices=['gas','r12','full'],default='gas');p.add_argument('--no-df',action='store_true');p.add_argument('--properties-only',action='store_true')
    a=p.parse_args();run(a.basis,a.environment,a.no_df,a.properties_only)
