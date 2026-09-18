"""Fixed-geometry flavin 14N FC+SD and traceless potential Hessian (EFG).

No native calibration, geometry optimization, SOC or relaxation inference.
An independent frozen-density potential finite difference checks every tensor.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
from scipy import constants as c

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/dark-atom-assignment/electronic_inputs'
OUTPUT = ROOT / 'data/dark-electronic-tensors'
STATES = ('E_neutral_SQ', 'R_oxidized', 'H_anionic_HQ', 'alternative_anionic_SQ')
G_N14 = 0.40376100  # Explicit 14N, I=1; PySCF nucprop / EasySpin isotope table.
A0 = c.physical_constants['Bohr radius'][0]
EFG_AU_SI = c.physical_constants['atomic unit of electric field gradient'][0]


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def traceless(t):
    t = np.asarray(t)
    return t - np.eye(3) * np.trace(t) / 3


def principal_efg(t):
    """|Vzz| >= |Vyy| >= |Vxx|; eta=(Vxx-Vyy)/Vzz in [0,1]."""
    vals, axes = np.linalg.eigh(t)
    order = np.argsort(np.abs(vals))
    vals, axes = vals[order], axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    return dict(Vxx_Vyy_Vzz_au=vals.tolist(), axes_columns=axes.tolist(),
                eta=float((vals[0]-vals[1])/vals[2]) if abs(vals[2]) > 1e-14 else None)


def finite_hessian(fun, origin, step):
    """Central differences, no derivative AO integrals; scalar or vector fun."""
    x = np.asarray(origin, float)
    directions = np.eye(3) * step
    zero = np.asarray(fun(x))
    out = np.empty(zero.shape + (3, 3))
    for i in range(3):
        out[..., i, i] = (fun(x+directions[i])-2*zero+fun(x-directions[i]))/step**2
        for j in range(i):
            out[..., i, j] = out[..., j, i] = (
                fun(x+directions[i]+directions[j])-fun(x+directions[i]-directions[j])
                -fun(x-directions[i]+directions[j])+fun(x-directions[i]-directions[j]))/(4*step**2)
    return out


def validate_input(spec):
    atoms = spec['atoms']
    if [a['index'] for a in atoms] != list(range(len(atoms))):
        raise ValueError('Noncontiguous input atom indices')
    nuclei = spec['requested_nuclei']
    if [(n['atom'], n['isotope'], n['spin']) for n in nuclei] != [('N5', '14N', 1.), ('N10', '14N', 1.)]:
        raise ValueError('Expected explicit N5/N10 14N, I=1')
    for n in nuclei:
        a = atoms[n['index']]
        if a['label'] != n['atom'] or a['element'] != 'N':
            raise ValueError('Nucleus/coordinate identity mismatch')
    if spec['spin_2S'] not in (0, 1):
        raise ValueError('Only fixed closed shell / doublet supported')
    return [n['index'] for n in nuclei]


def hfc_si_prefactor():
    """MHz per bohr^-3 spin density for S=1/2, dimensionless spin operators."""
    ge = abs(c.physical_constants['electron g factor'][0])
    muB = c.physical_constants['Bohr magneton'][0]
    muN = c.physical_constants['nuclear magneton'][0]
    return (2/3)*c.mu_0*ge*muB*G_N14*muN/c.h/1e6/A0**3


def tensor_bundle(result):
    """Versioned compatible input for the fixed two-14N identity validator."""
    if result['state_id'] != 'E_neutral_SQ' or result['spin_2S'] != 1:
        raise ValueError('This bundle is the fixed neutral SQ only')
    return dict(state_id=result['state_id'], nuclei=[dict(atom=n['atom'], isotope=n['isotope'], spin=n['spin_I']) for n in result['nuclei']],
                evaluation_site='actual_nuclear_coordinates', units='rad/s', nuclear_dimension=9,
                tensors=[n['HFC_rad_s'] for n in result['nuclei']],
                source=result['label']+'/result.json', quadrupolar_tensors=None,
                EFG_tensors_au=[n['EFG_au'] for n in result['nuclei']],
                EFG_status='calculated static capped-fragment value; native EFG and fluctuations uncalibrated',
                frame=result['frame'], native_calibrated=False)


def independent_properties(mol, dm_total, dm_spin, ids, factor_mhz, steps=(.002, .001)):
    """Finite Hessian of scalar Coulomb potential, self nucleus excluded.

    Checks FC+SD against H(spin potential)-I*tr H; EFG is the traceless
    Hessian of the physical electrostatic potential (nuclei minus electrons).
    Both use int1e_rinv only, independent of analytic derivative integrals.
    """
    rows = []
    for atom_id in ids:
        origin = mol.atom_coord(atom_id)
        other = np.arange(mol.natm) != atom_id
        coords = mol.atom_coords()[other]
        charges = mol.atom_charges()[other]
        def potentials(x):
            with mol.with_rinv_origin(x):
                rinv = mol.intor('int1e_rinv')
            electron = np.einsum('ij,ji', rinv, dm_total)
            spin = np.einsum('ij,ji', rinv, dm_spin)
            nuclear = np.sum(charges/np.linalg.norm(coords-x, axis=1))
            return np.array([nuclear-electron, spin])
        hs = [finite_hessian(potentials, origin, h) for h in steps]
        rich = (4*hs[1]-hs[0])/3
        efgs = [traceless(h[0]) for h in hs]
        hfcs = [factor_mhz*(h[1]-np.eye(3)*np.trace(h[1])) for h in hs]
        rows.append(dict(atom_index=atom_id, step_bohr=list(steps),
                         EFG_steps_au=np.array(efgs).tolist(), EFG_richardson_au=traceless(rich[0]).tolist(),
                         HFC_steps_MHz=np.array(hfcs).tolist(),
                         HFC_richardson_MHz=(factor_mhz*(rich[1]-np.eye(3)*np.trace(rich[1]))).tolist()))
    return rows


def run(state, basis='def2-svpd', density_fit=True, label=None, guess_dir=None):
    import pyscf
    from pyscf import gto, dft, lib, scf
    from pyscf.prop.hfc import uhf
    from pyscf.prop.efg import rhf as efg
    from pyscf.data import nist
    start = time.monotonic()
    lib.num_threads(2)
    spec = json.loads((INPUT / (state+'.json')).read_text())
    ids = validate_input(spec)
    folder = OUTPUT / (label or state)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'result.json').exists():
        raise FileExistsError('Refusing to overwrite a completed calculation: '+str(folder))
    basis_spec = basis
    if basis == 'pcJ-2-N_def2-svpd':
        basis_spec = {'default': 'def2-svpd', 'N': gto.basis.parse(
            (ROOT/'evidence/coefficient-identification/pcJ-2_N.nwchem').read_text())}
    mol = gto.M(atom=[(a['element'], a['xyz_A']) for a in spec['atoms']],
                charge=spec['charge'], spin=spec['spin_2S'], unit='Angstrom',
                basis=basis_spec, max_memory=2500, verbose=4, output=str(folder/'scf.log'))
    assert mol.nelectron == spec['electron_count']
    mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.xc = 'b3lyp'
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    mf.chkfile = str(folder/'scf.chk')
    guess = None
    if guess_dir:
        old = OUTPUT / guess_dir
        old_mol = gto.loads((old/'mol.json').read_text())
        old_dm = np.load(old/'density.npz')['dm']
        guess = np.array([scf.addons.project_dm_nr2nr(old_mol, d, mol) for d in old_dm]) if old_dm.ndim == 3 else scf.addons.project_dm_nr2nr(old_mol, old_dm, mol)
    energy = mf.kernel(dm0=guess)
    if not mf.converged:
        mf = mf.newton()
        mf.max_cycle = 100
        energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError('SCF failed: '+state)
    dm = mf.make_rdm1()
    dm_total = dm.sum(axis=0) if mol.spin else dm
    dm_spin = dm[0]-dm[1] if mol.spin else np.zeros_like(dm)
    overlap = mol.intor('int1e_ovlp')
    # Explicit isotope check; the library FC+SD routine otherwise uses defaults.
    assert uhf.get_nuc_g_factor('N', mass=14) == G_N14 == uhf.get_nuc_g_factor('N')
    factor = nist.ALPHA**2*.5*nist.G_ELECTRON*(nist.HARTREE2J/nist.PLANCK*1e-6)*G_N14*.5*nist.E_MASS/nist.PROTON_MASS
    hfc = uhf.make_fcdip(uhf.HyperfineCoupling(mf), dm, ids, verbose=0) if mol.spin else None
    efgs = efg.kernel(mf, ids)
    rho = np.einsum('pi,ij,pj->p', mol.eval_gto('GTOval', mol.atom_coords()[ids]), dm_spin,
                    mol.eval_gto('GTOval', mol.atom_coords()[ids]))
    direct_fc = hfc_si_prefactor()*rho
    steps = (.00025, .000125) if basis == 'pcJ-2-N_def2-svpd' else (.002, .001)
    independent = independent_properties(mol, dm_total, dm_spin, ids, factor, steps)
    nuclear_rows = []
    for i, atom_id in enumerate(ids):
        row = dict(atom=spec['atoms'][atom_id]['label'], atom_index=atom_id,
                   isotope='14N', spin_I=1, xyz_A=spec['atoms'][atom_id]['xyz_A'],
                   EFG_au=efgs[i].tolist(), EFG_V_m_minus2=(efgs[i]*EFG_AU_SI).tolist(),
                   EFG_principal=principal_efg(efgs[i]),
                   EFG_trace_au=float(np.trace(efgs[i])),
                   EFG_symmetry_residual_au=float(np.max(abs(efgs[i]-efgs[i].T))),
                   EFG_independent_max_error_au=float(np.max(abs(efgs[i]-independent[i]['EFG_richardson_au']))),
                   quadrupole_Q_barn=None, quadrupole_tensor_rad_s=None,
                   CQ_MHz_per_Q_barn=float(principal_efg(efgs[i])['Vxx_Vyy_Vzz_au'][2]*c.e*EFG_AU_SI*1e-28/c.h/1e6),
                   HFC_MHz=None, HFC_rad_s=None,
                   HFC_status='not applicable: no unpaired electron in selected closed shell')
        if hfc is not None:
            iso = float(np.trace(hfc[i])/3)
            eig, axes = np.linalg.eigh(hfc[i])
            row.update(HFC_MHz=hfc[i].tolist(), HFC_rad_s=(2*np.pi*1e6*hfc[i]).tolist(),
                       HFC_status='calculated FC+SD only', isotropic_MHz=iso,
                       spin_dipole_MHz=(hfc[i]-np.eye(3)*iso).tolist(),
                       HFC_principal_MHz=eig.tolist(), HFC_principal_axes_columns=axes.tolist(),
                       independent_FC_MHz=float(direct_fc[i]), spin_density_bohr_minus3=float(rho[i]),
                       FC_independent_relative_error=float(abs(iso-direct_fc[i])/max(abs(iso),1e-8)),
                       HFC_independent_max_error_MHz=float(np.max(abs(hfc[i]-independent[i]['HFC_richardson_MHz']))),
                       HFC_symmetry_residual_MHz=float(np.max(abs(hfc[i]-hfc[i].T))))
        nuclear_rows.append(row)
    result = dict(state_id=state, label=folder.name, status='fixed_geometry_fragment_estimate',
                  xc='b3lyp', basis=basis, density_fit=density_fit, auxbasis='def2-universal-jkfit' if density_fit else None,
                  grids_level=4, scf_conv_tol=1e-10, converged=bool(mf.converged),
                  charge=mol.charge, spin_2S=mol.spin, electron_count=mol.nelectron,
                  integrated_electron_count=float(np.einsum('ij,ji',dm_total,overlap)),
                  integrated_spin_density=float(np.einsum('ij,ji',dm_spin,overlap)),
                  spin_square=float(mf.spin_square()[0]) if mol.spin else 0.,
                  energy_hartree=float(energy), nao=mol.nao, nuclear_g_factor_14N=G_N14,
                  EFG_atomic_unit_V_m_minus2=EFG_AU_SI, geometry_status=spec['geometry_status'],
                  geometry_source=spec['source'], frame='unchanged input Cartesian PDB frame; eigenvectors are columns',
                  protonation={k:spec[k] for k in ('N5_protonated','N1_protonated','N3_protonated')},
                  nuclei=nuclear_rows, input_sha256=hashlib.sha256((INPUT/(state+'.json')).read_bytes()).hexdigest(),
                  elapsed_s=time.monotonic()-start, nuclear_T1_s=None, nuclear_T2_s=None,
                  native_calibrated=False, native_neural_gain=None,
                  scope='Gas-phase capped lumiflavin starting geometry; no optimization, protein/solvent, SOC, native rate or relaxation inference.',
                  conventions={'HFC':'H/h = S dot A_MHz dot I * 1e6; dimensionless spins; positive electron g magnitude',
                               'EFG':'traceless Hessian of electrostatic potential; other nuclei plus negative electron charge; excludes target nuclear self field',
                               'quadrupole':'Q not adopted in this run; CQ=e Q Vzz/h; conversion per barn only; raw library log uses its own tabulated Q'})
    save(folder/'result.json', result)
    save(folder/'independent.json', independent)
    save(folder/'input.json', spec)
    (folder/'mol.json').write_text(mol.dumps())
    np.savez_compressed(folder/'density.npz', dm=dm, overlap=overlap)
    save(folder/'runtime.json',dict(python=platform.python_version(), numpy=np.__version__, pyscf=pyscf.__version__,
                                  platform=platform.platform(), requested_threads=2, effective_pyscf_threads=lib.num_threads(), random_seed='none',
                                  properties_version='0.1.0', guess_dir=guess_dir))
    print(json.dumps(dict(state=state,label=folder.name,elapsed_s=result['elapsed_s'],energy=energy,
                          spin_square=result['spin_square'],nuclei=nuclear_rows)),flush=True)
    assert abs(result['integrated_electron_count']-mol.nelectron)<1e-7
    assert abs(result['integrated_spin_density']-mol.spin)<1e-7
    assert all(abs(r['EFG_trace_au'])<1e-7 and r['EFG_independent_max_error_au']<2e-5 for r in nuclear_rows)
    if mol.spin:
        assert all(r['FC_independent_relative_error']<1e-5 and r['HFC_independent_max_error_MHz']<2e-3 for r in nuclear_rows)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--state', choices=STATES, required=True)
    p.add_argument('--basis', default='def2-svpd')
    p.add_argument('--no-density-fit', action='store_true')
    p.add_argument('--label')
    p.add_argument('--guess-dir')
    a = p.parse_args()
    run(a.state, a.basis, not a.no_density_fit, a.label, a.guess_dir)
