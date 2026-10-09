"""Matched fixed-geometry electronic response; never edits source checkpoints."""
from pathlib import Path
import sys,json,time,argparse,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import dark_electronic_response as r
from impl import dark_electronic_tensors as b
from impl import dark_embedded_tensors as e
D=ROOT/'data/field-couplings';FIELD=14e6

def load(state,out):
 from pyscf import gto,dft,scf,qmmm,lib
 lib.num_threads(2)
 label={'H':'H_anionic_HQ','R':'R_oxidized','E':'E_neutral_SQ'}[state]
 src=(ROOT/'data/dark-basis-resolution/upcj2N__full' if state=='E' else e.OUT/e.case_name(label,2026091521,0,12,1))
 meta=json.loads((src/'result.json').read_text());mol=gto.loads((src/'mol.json').read_text());mol.output=str(out/'engine.log');mol.stdout=open(mol.output,'w');mol.verbose=4;mol.max_memory=2600
 mf=(dft.UKS(mol) if mol.spin else dft.RKS(mol)).density_fit(auxbasis='def2-universal-jkfit')
 if state=='E':
  xyz,charges,_,_=e.environment(2026091521,0,1000);mf=qmmm.mm_charge(mf,xyz,charges,unit='Angstrom');h=mf.get_hcore()
 else:h=np.load(src/'density.npz')['hcore']
 mf.get_hcore=lambda *args,**kwargs:h
 mf.xc='b3lyp';mf.grids.level=4;mf.conv_tol=1e-12;mf.conv_tol_grad=1e-8;mf.max_cycle=100
 mf.__dict__.update(scf.chkfile.load(str(src/'scf.chk'),'scf'));mf.converged=True;mf.chkfile=str(out/'new.chk')
 spec=json.loads((ROOT/'data/dark-electronic-tensors'/label/'input.json').read_text());ids=b.validate_input(spec)
 return mf,ids,src,meta,h

def main(state,finite=False,resume=False,acquire_only=False):
 from pyscf.prop.efg import rhf as efg
 from pyscf.prop.hfc import uhf as hfc
 import pyscf
 out=D/state;out.mkdir(parents=True,exist_ok=True)
 if finite and (out/'engine.log').exists() and not (out/'response_engine.log').exists():(out/'engine.log').rename(out/'response_engine.log')
 start=time.monotonic();mf,ids,src,meta,h0=load(state,out)
 origin=np.average(mf.mol.atom_coords(),axis=0,weights=mf.mol.atom_charges())
 with mf.mol.with_common_orig(origin):dip=mf.mol.intor_symmetric('int1e_r',comp=3)
 base=mf.make_rdm1();v0,a0=r.properties(mf,ids,base)
 if not finite:
  dm,checks=r.density_response(mf);total=dm.sum(0) if mf.mol.spin else dm
  dv=np.array([-np.einsum('abij,kji->kab',efg._get_quadrupole_integrals(mf.mol,i),total) for i in ids])
  da=np.array([hfc.make_fcdip(hfc.HyperfineCoupling(mf),dm[:,k],ids,verbose=0) for k in range(3)]).transpose(1,0,2,3) if mf.mol.spin else np.zeros((2,3,3,3))
  baselineV=np.array([n['EFG_QM_au'] for n in meta['nuclei']]);ev=float(np.linalg.norm(v0-baselineV)/np.linalg.norm(baselineV));assert ev<1e-7,ev
  ea=None
  if a0 is not None:
   baselineA=np.array([n['HFC_MHz'] for n in meta['nuclei']]);ea=float(np.linalg.norm(a0-baselineA)/np.linalg.norm(baselineA));assert ea<1e-7,ea
  np.savez_compressed(out/'response.npz',dV_au_per_field_au=dv,dA_MHz_per_field_au=da,V0_au=v0,A0_MHz=np.zeros((2,3,3)) if a0 is None else a0)
  rows=[]
  for i,name in enumerate(('N5','N10')):
   rows.append(dict(atom=name,EFG_max_relative_change_at_14MVm=float(np.linalg.svd(dv[i].reshape(3,9),compute_uv=False)[0]*FIELD/r.FIELD_AU/np.linalg.norm(v0[i])),isotropic_HFC_max_change_MHz_at_14MVm=float(np.linalg.norm(np.trace(da[i],axis1=1,axis2=2)/3)*FIELD/r.FIELD_AU),baseline_isotropic_HFC_MHz=None if a0 is None else float(np.trace(a0[i])/3)))
  result=dict(state=state,source=str(src.relative_to(ROOT)),nuclei=rows,checks=checks,baseline_EFG_relative_error=ev,baseline_HFC_relative_error=ea,field_au_V_m=r.FIELD_AU,geometry='same frozen capped flavin as current operators',environment='R/H radius12 A; E full minimum-image MM cell, as original source',basis='E uncontracted pcJ2 N; R/H def2-svpd',pyscf=pyscf.__version__,seconds=time.monotonic()-start,native_calibrated=False,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src/'scf.chk',src/'mol.json',src/'result.json')})
  (out/'response.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 else:
  mf.conv_tol=1e-11;mf.conv_tol_grad=5e-8
  baseline_gradient=float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ)))
  assert baseline_gradient<1e-6,baseline_gradient
  rows=[]
  # Check a non-axis-aligned direction (all derivative components contribute).
  direction=np.array([1.,2.,3.])/np.sqrt(14);values={}
  for amplitude in ([14e6,7e6] if state in ('H','E') else [14e6]):
   vs=[];aa=[]
   for sign in (1,-1):
    saved=out/f'finite_{amplitude:g}_{sign}.npz'
    if resume and saved.exists():
     cached=np.load(saved);vs.append(cached['V_au']);aa.append(cached['A_MHz']);continue
    field=sign*amplitude/r.FIELD_AU*direction;hf=h0+np.einsum('a,aij->ij',field,dip)
    mf.get_hcore=lambda *args,hh=hf,**kwargs:hh;mf.chkfile=str(out/f'finite_{amplitude:g}_{sign}.chk')
    energy=mf.kernel(dm0=base)
    if not mf.converged:raise RuntimeError('finite-field SCF did not converge')
    grad=float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ)));assert grad<2e-7,grad
    v,a=r.properties(mf,ids,mf.make_rdm1());vs.append(v);aa.append(np.zeros((2,3,3)) if a is None else a)
    np.savez_compressed(saved,V_au=v,A_MHz=aa[-1],field_V_m=sign*amplitude*direction,gradient_norm=grad)
   if acquire_only:continue
   derivative=dict(np.load(out/'response.npz'))
   dv=np.einsum('a,naij->nij',direction,derivative['dV_au_per_field_au'])*amplitude/r.FIELD_AU
   da=np.einsum('a,naij->nij',direction,derivative['dA_MHz_per_field_au'])*amplitude/r.FIELD_AU
   record=dict(amplitude_V_m=amplitude,direction=direction.tolist(),EFG_derivative_relative_error=float(np.linalg.norm((vs[0]-vs[1])/2-dv)/np.linalg.norm(dv)),EFG_even_nonlinearity_relative_to_linear=float(np.linalg.norm((vs[0]+vs[1])/2-v0)/np.linalg.norm(dv)))
   if mf.mol.spin:record.update(HFC_derivative_relative_error=float(np.linalg.norm((aa[0]-aa[1])/2-da)/np.linalg.norm(da)),HFC_even_nonlinearity_relative_to_linear=float(np.linalg.norm((aa[0]+aa[1])/2-a0)/np.linalg.norm(da)))
   rows.append(record);print(json.dumps(record),flush=True)
  (out/('finite_acquisition.json' if acquire_only else 'finite_checks.json')).write_text(json.dumps(dict(state=state,checks=rows,baseline_gradient=baseline_gradient,seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('state',choices=['R','H','E']);p.add_argument('--finite',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--acquire-only',action='store_true');a=p.parse_args();main(a.state,a.finite,a.resume,a.acquire_only)
