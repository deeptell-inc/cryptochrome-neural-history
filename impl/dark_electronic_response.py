"""State-resolved clamped-nuclei electric response and static SQ g tensors.

These are computed coefficients of a specified fragment, not native relaxation
or neural-gain identification. No unseen stochastic input is filled in.
"""
from pathlib import Path
import json,time,argparse,copy,hashlib,shutil
import numpy as np
from scipy import constants as c
from . import dark_electronic_tensors as old
ROOT=old.ROOT;OUT=ROOT/'data/dark-electronic-response'
FIELD_AU=c.physical_constants['atomic unit of electric field'][0]

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def load_mf(label,folder,translation=None):
    from pyscf import gto,dft,scf,lib
    lib.num_threads(1)
    src=old.OUTPUT/label;meta=json.loads((src/'result.json').read_text())
    mol=gto.loads((src/'mol.json').read_text())
    if translation is not None:mol.set_geom_(mol.atom_coords()+translation,unit='Bohr')
    mol.output=str(folder/'engine.log');mol.stdout=open(mol.output,'w');mol.verbose=4;mol.max_memory=2200
    mf=dft.UKS(mol) if mol.spin else dft.RKS(mol)
    if meta['density_fit']:mf=mf.density_fit(auxbasis=meta['auxbasis'])
    mf.xc='b3lyp';mf.grids.level=4;mf.conv_tol=1e-11;mf.max_cycle=100;mf.chkfile=str(folder/'new.chk')
    mf.__dict__.update(scf.chkfile.load(str(src/'scf.chk'),'scf'));mf.converged=True
    spec=json.loads((src/'input.json').read_text());ids=old.validate_input(spec)
    return mf,meta,ids

def properties(mf,ids,dm):
    from pyscf.prop.efg import rhf as efg
    from pyscf.prop.hfc import uhf as hfc
    total=dm.sum(0) if dm.ndim==3 else dm
    V=np.array([efg._get_quad_nuc(mf.mol,i)-np.einsum('abij,ji->ab',efg._get_quadrupole_integrals(mf.mol,i),total) for i in ids])
    A=hfc.make_fcdip(hfc.HyperfineCoupling(mf),dm,ids,verbose=0) if dm.ndim==3 else None
    return V,A

def density_response(mf):
    from pyscf.scf import cphf,ucphf
    from pyscf.prop.polarizability import rhf,uhf
    mol=mf.mol;origin=np.average(mol.atom_coords(),axis=0,weights=mol.atom_charges())
    with mol.with_common_orig(origin):dip=mol.intor_symmetric('int1e_r',comp=3)
    if not mol.spin:
        C=mf.mo_coeff;occ=mf.mo_occ>0;Co=C[:,occ]
        h1=np.einsum('xpq,pi,qj->xij',dip,C,Co,optimize=True);vind=rhf.Polarizability(mf).gen_vind(mf,C,mf.mo_occ)
        mo1=cphf.solve(vind,mf.mo_energy,mf.mo_occ,h1.copy(),np.zeros_like(h1),max_cycle=80,tol=1e-10)[0]
        raw=np.einsum('xai,pa,qi->xpq',mo1,C,Co,optimize=True)*2;dm=raw+raw.transpose(0,2,1)
        v=vind(mo1).reshape(h1.shape);gap=mf.mo_energy[~occ,None]-mf.mo_energy[occ]
        residual=mo1[:,~occ]*gap+h1[:,~occ]+v[:,~occ]
        relative=np.linalg.norm(residual)/np.linalg.norm(h1[:,~occ]);total=dm
    else:
        C=mf.mo_coeff;occ=[o>0 for o in mf.mo_occ];Co=[cc[:,o] for cc,o in zip(C,occ)]
        h1=[np.einsum('xpq,pi,qj->xij',dip,cc,co,optimize=True) for cc,co in zip(C,Co)]
        vind=uhf.Polarizability(mf).gen_vind(mf,C,mf.mo_occ)
        mo1=ucphf.solve(vind,mf.mo_energy,mf.mo_occ,tuple(x.copy() for x in h1),tuple(np.zeros_like(x) for x in h1),max_cycle=80,tol=1e-10)[0]
        raw=[np.einsum('xai,pa,qi->xpq',u,cc,co,optimize=True) for u,cc,co in zip(mo1,C,Co)]
        dm=np.array([x+x.transpose(0,2,1) for x in raw]);total=dm.sum(0)
        flat=np.hstack([x.reshape(3,-1) for x in mo1]);v=vind(flat).reshape(3,-1);cut=mo1[0].shape[1]*mo1[0].shape[2]
        vv=[v[:,:cut].reshape(h1[0].shape),v[:,cut:].reshape(h1[1].shape)];res=[];rhs=[]
        for o,eps,u,h,vv0 in zip(occ,mf.mo_energy,mo1,h1,vv):
            gap=eps[~o,None]-eps[o];res.append((u[:,~o]*gap+h[:,~o]+vv0[:,~o]).ravel());rhs.append(h[:,~o].ravel())
        relative=np.linalg.norm(np.concatenate(res))/np.linalg.norm(np.concatenate(rhs))
    assert relative<1e-7,relative
    traces=np.einsum('...ij,ji->...',dm,mf.get_ovlp());assert np.max(abs(traces))<1e-7
    polar=-np.einsum('aij,bji->ab',dip,total)
    return dm,dict(CPHF_relative_residual=float(relative),electron_number_derivative_max=float(np.max(abs(traces))),
                   polarizability_au=polar.tolist(),polarizability_symmetry_error=float(np.max(abs(polar-polar.T))),field_origin_bohr=origin.tolist())

def response(label):
    from pyscf.prop.efg import rhf as efg
    from pyscf.prop.hfc import uhf as hfc
    folder=OUT/label;folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    if (folder/'response.json').exists():raise FileExistsError(folder/'response.json')
    mf,meta,ids=load_mf(label,folder);dm,checks=density_response(mf)
    total=dm.sum(0) if mf.mol.spin else dm
    DV=np.array([-np.einsum('abij,kji->kab',efg._get_quadrupole_integrals(mf.mol,i),total) for i in ids])
    DA=np.array([hfc.make_fcdip(hfc.HyperfineCoupling(mf),dm[:,k],ids,verbose=0) for k in range(3)]).transpose(1,0,2,3) if mf.mol.spin else None
    np.savez_compressed(folder/'density_response.npz',density_derivative_per_field_au=dm,EFG_derivative_au=DV,HFC_derivative_MHz=np.array([]) if DA is None else DA)
    v0,a0=properties(mf,ids,mf.make_rdm1())
    rows=[]
    for n,i in enumerate(ids):
        rows.append(dict(atom=('N5','N10')[n],EFG_derivative_per_V_m=(DV[n]*old.EFG_AU_SI/FIELD_AU).tolist(),
                         HFC_derivative_MHz_per_V_m=None if DA is None else (DA[n]/FIELD_AU).tolist(),
                         baseline_EFG_au=v0[n].tolist(),baseline_HFC_MHz=None if a0 is None else a0[n].tolist()))
    save(folder/'response.json',dict(label=label,state=meta['state_id'],basis=meta['basis'],density_fit=meta['density_fit'],
        acquisition='analytic coupled KS electric-field response at frozen gas-phase geometry',nuclei=rows,checks=checks,
        tensor_axes='derivative input field axis, tensor row, tensor column; original Cartesian PDB frame',
        field_au_V_m=FIELD_AU,clamped_nuclei=True,native_calibrated=False,elapsed_s=time.monotonic()-start,
        excludes=['geometry and protonation response','inhomogeneous protein/solvent response','electric-field time spectrum','native T1/T2 and neural gain']))
    print('response completed',label,time.monotonic()-start,flush=True)

def finite_check(label,axis=0,field=1e-4,refine=False):
    folder=OUT/label/f'finite_axis{axis}_{field:g}';folder.mkdir(parents=True,exist_ok=True)
    starts={}
    if refine:
        archive=folder/'before_refinement';archive.mkdir(exist_ok=False)
        for p in folder.iterdir():
            if p.is_file():shutil.copyfile(p,archive/p.name)
        starts={sign:np.load(archive/f'density_{sign}.npz')['dm'] for sign in (1,-1)}
    mf,meta,ids=load_mf(label,folder)
    if refine:mf.conv_tol=1e-11;mf.conv_tol_grad=1e-8
    base=mf.make_rdm1();origin=np.average(mf.mol.atom_coords(),axis=0,weights=mf.mol.atom_charges())
    with mf.mol.with_common_orig(origin):dip=mf.mol.intor_symmetric('int1e_r',comp=3)
    h0=mf.get_hcore();values=[]
    for sign in (1,-1):
        mf.get_hcore=lambda *args,ss=sign,**kwargs:h0+ss*field*dip[axis]
        mf.chkfile=str(folder/f'finite_{sign}.chk');energy=mf.kernel(dm0=starts.get(sign,base))
        if not mf.converged:raise RuntimeError('finite-field SCF failed')
        dm=mf.make_rdm1();V,A=properties(mf,ids,dm);values.append((V,A,dm))
        np.savez_compressed(folder/f'density_{sign}.npz',dm=dm,EFG_au=V,HFC_MHz=np.array([]) if A is None else A)
    compare_finite(label,axis,field)

def compare_finite(label,axis=0,field=1e-4):
    # Reuse converged +/- SCFs if the analytic calculation finished later.
    folder=OUT/label/f'finite_axis{axis}_{field:g}'
    values=[]
    for sign in (1,-1):
        q=np.load(folder/f'density_{sign}.npz')
        values.append((q['EFG_au'],q['HFC_MHz'],q['dm']))
    open_shell=values[0][2].ndim==3
    z=np.load(OUT/label/'density_response.npz');der=(values[0][0]-values[1][0])/(2*field);ref=z['EFG_derivative_au'][:,axis]
    errors=dict(EFG_relative_error=float(np.linalg.norm(der-ref)/np.linalg.norm(ref)))
    if open_shell:
        derA=(values[0][1]-values[1][1])/(2*field);refA=z['HFC_derivative_MHz'][:,axis];errors['HFC_relative_error']=float(np.linalg.norm(derA-refA)/np.linalg.norm(refA))
    dmfd=(values[0][2]-values[1][2])/(2*field);dmref=z['density_derivative_per_field_au'][:,axis] if open_shell else z['density_derivative_per_field_au'][axis]
    errors['density_relative_error']=float(np.linalg.norm(dmfd-dmref)/np.linalg.norm(dmref))
    save(folder/'check.json',dict(axis=axis,field_au=field,**errors,scope='independent fully relaxed finite-field SCF versus analytic derivative'))
    print('finite field',label,errors,flush=True)

def three_rhs_adapter(vind,width):
    """Compatibility for properties' fixed-3 RHS callback and variable Krylov blocks.

    The electronic response is linear and independent for each right-hand side.
    Padding a final block with zeros changes no retained response component.
    """
    def apply(x):
        rows=np.asarray(x).reshape(-1,width);out=[]
        for start in range(0,len(rows),3):
            n=min(3,len(rows)-start);padded=np.zeros((3,width),dtype=rows.dtype);padded[:n]=rows[start:start+n]
            out.append(np.asarray(vind(padded)).reshape(3,width)[:n])
        return np.concatenate(out).ravel()
    return apply

def gtensor(label,shift=False):
    from pyscf.prop.gtensor import uks
    from pyscf.prop.nmr import uhf as nmr
    folder=OUT/label/('g_translation' if shift else 'g_tensor');folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();mf,meta,ids=load_mf(label,folder,translation=np.array([2.3,-1.7,.8]) if shift else None)
    assert mf.mol.spin==1
    mf.grids.build(with_non0tab=True)
    obj=uks.GTensor(mf);obj.max_cycle_cphf=80;obj.conv_tol=1e-10
    width=sum(cc.shape[1]*int(np.count_nonzero(oo)) for cc,oo in zip(mf.mo_coeff,mf.mo_occ))
    obj.cphf=three_rhs_adapter(nmr.gen_vind(mf,mf.mo_coeff,mf.mo_occ),width)
    original_fock=obj.get_fock;original_overlap=obj.get_ovlp
    def fock(*args,**kwargs):
        path=folder/'magnetic_fock.npy'
        if path.exists():return np.load(path)
        value=original_fock(*args,**kwargs);np.save(path,value);return value
    def overlap(*args,**kwargs):
        path=folder/'magnetic_overlap.npy'
        if path.exists():return np.load(path)
        value=original_overlap(*args,**kwargs);np.save(path,value);return value
    obj.get_fock=fock;obj.get_ovlp=overlap
    g=obj.kernel();sv=np.linalg.svd(g,compute_uv=False)
    occ=[o>0 for o in mf.mo_occ];co=[cc[:,o] for cc,o in zip(mf.mo_coeff,occ)]
    h=[np.einsum('xpq,pi,qj->xij',hh,cc,ci,optimize=True) for hh,cc,ci in zip(fock(),mf.mo_coeff,co)]
    s=[np.einsum('xpq,pi,qj->xij',overlap(mf.mol),cc,ci,optimize=True) for cc,ci in zip(mf.mo_coeff,co)]
    flat=np.hstack([u.reshape(3,-1) for u in obj.mo10]);v=obj.cphf(flat).reshape(3,width)
    cut=obj.mo10[0].shape[1]*obj.mo10[0].shape[2];vv=[v[:,:cut].reshape(h[0].shape),v[:,cut:].reshape(h[1].shape)]
    residual=[];rhs=[]
    for o,eps,u,hh,ss,vv0 in zip(occ,mf.mo_energy,obj.mo10,h,s,vv):
        b=hh-ss*eps[o];gap=eps[~o,None]-eps[o]
        residual.append((u[:,~o]*gap+b[:,~o]+vv0[:,~o]).ravel());rhs.append(b[:,~o].ravel())
    relative=float(np.linalg.norm(np.concatenate(residual))/np.linalg.norm(np.concatenate(rhs)))
    assert relative<1e-7,relative
    np.savez_compressed(folder/'magnetic_response.npz',alpha=obj.mo10[0],beta=obj.mo10[1])
    save(folder/'result.json',dict(label=label,g_tensor=g.tolist(),principal_g_singular_values=sv.tolist(),
        g_convention='energy mu_B B dot g dot S; singular values define EPR principal values',
        isotropic_trace_g=float(np.trace(g)/3),anisotropy_frobenius=float(np.linalg.norm(g-np.eye(3)*np.trace(g)/3)),
        method='B3LYP, coupled response, GIAO, AMFI+SSO+SOO paramagnetic SOC, no dia_soc2e',
        basis=meta['basis'],density_fit=meta['density_fit'],translation_bohr=[2.3,-1.7,.8] if shift else [0,0,0],
        CPHF_relative_residual=relative,compatibility='zero-padded three-RHS callback for variable Krylov blocks',
        native_calibrated=False,electron_T1_s=None,electron_T2_s=None,elapsed_s=time.monotonic()-start))
    print('g tensor completed',label,sv,time.monotonic()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--mode',choices=['response','finite','finite-refine','finite-compare','g','g-shift'],default='response');p.add_argument('--field',type=float,default=1e-4);p.add_argument('--axis',type=int,default=0);a=p.parse_args()
    if a.mode=='response':response(a.label)
    elif a.mode=='finite':finite_check(a.label,a.axis,a.field)
    elif a.mode=='finite-refine':finite_check(a.label,a.axis,a.field,refine=True)
    elif a.mode=='finite-compare':compare_finite(a.label,a.axis,a.field)
    else:gtensor(a.label,a.mode=='g-shift')
