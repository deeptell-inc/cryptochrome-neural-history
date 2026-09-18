"""Sensitivity to SQ coefficients; hold all other states/kinetics fixed."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
import pandas as pd
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import dark_basis_resolution as br
from impl import dark_embedded_tensors as e
from impl import dark_md_coefficients as m
from impl import dark_electronic_tensors as b
from impl import dark_threshold_source as s
from impl import dark_motion_retention as motion
from impl import dark_finite_spike as f

def coefficients(result):
    _,Q,_,_,_,z=m.coefficient_series(2026091521,basis='svpd',external=True,tag='_dense')
    tensors={}
    for state,label in [('R','R_oxidized'),('H','H_anionic_HQ')]:
        d=json.loads((e.OUT/e.case_name(label,2026091521,0,12,1)/'result.json').read_text())
        tensors[state]=np.array([x['EFG_QM_au'] for x in d['nuclei']])*b.EFG_AU_SI
    tensors['E']=np.array([x['EFG_QM_au'] for x in result['nuclei']])*b.EFG_AU_SI
    A=np.array([x['HFC_MHz'] for x in result['nuclei']])*2*np.pi*1e6
    coeff={}
    for state,V in tensors.items():
        transported=Q@V@Q.swapaxes(-1,-2)+z['external_EFG_V_m2']
        coeff[state]=(m.PREF*np.einsum('tnij,kij->tnk',transported,m.B)).reshape(len(Q),10)
    rotated=Q@A@Q.swapaxes(-1,-2)
    coeff['E']=np.column_stack([coeff['E'],np.einsum('tnij,kij->tnk',rotated,m.B).reshape(len(Q),10)])
    return coeff,np.trace(A,axis1=-2,axis2=-1)/3,float(np.diff(z['time_ps'])[0])*1e-12

def main():
    rows=[];retention=[];checks=[]
    cases=[json.loads(p.read_text()) for p in sorted(br.OUT.glob('*/result.json'))]
    for label in ('E_neutral_SQ','E_neutral_SQ_pcJ2N'):
        d=json.loads((e.OUT/e.case_name(label,2026091521,0,12,1)/'result.json').read_text());d['label']='old_'+label+'__r12';cases.append(d)
    for d in cases:
        coeff,iso,dt=coefficients(d);gs={};hs={}
        for state in ('R','H','E','RP'):
            gs[state],hs[state],_,_=m.make_generator(state,coeff['E' if state=='RP' else state],iso,dt,round(1e-12/dt))
        retention.append(dict(condition=d['label'],**m.memory(gs['E'],hs['E'],'E',.012)))
        wr=m.wait_map(gs['R'],10);wh=m.wait_map(gs['H'],10);we=m.wait_map(gs['E'],4)
        _,u=eigh(hs['H']);dephase,_=s.population_projector(u)
        for gamma in (0.,1e8):
            L=gs['RP'].copy()
            for spin in s.S:L+=gamma*motion.dissipator(np.kron(np.kron(np.eye(2),spin),np.eye(9)))
            p,escape=m.reaction_maps(L);cyc=wh@wr@(wr@p+s.TRACE_E@we@escape)
            flow=float(np.max(abs(s.trvec(9)@p+s.trvec(18)@escape-s.trvec(9))))
            mass=float(np.max(abs(s.trvec(9)@cyc-s.trvec(9))))
            J=cyc.reshape(9,9,9,9,order='F').transpose(0,2,1,3).reshape(81,81,order='F')
            cp=float(eigh((J+J.conj().T)/2,eigvals_only=True)[0])
            yref=float((s.trvec(9)@p@s.X0).real);ref=f.first_passage(.05*yref,.05*yref,.012,'tonic_excess',n=1200)['firing_probability']
            for name,update in [('full',cyc),('population_HQ',dephase@cyc@dephase),('scalar',s.RESET@cyc@s.RESET)]:
                x=np.linalg.matrix_power(update,20)@s.X0;y=float((s.trvec(9)@p@x).real)
                spike=f.first_passage(.05*y,.05*yref,.012,'tonic_excess',n=1200)['firing_probability']
                minimum=float(eigh(s.unvec(x,9),eigvals_only=True)[0]);assert min(minimum,cp)>-1e-7 and max(flow,mass)<1e-7
                rows.append(dict(condition=d['label'],gamma_oxygen_s=gamma,model=name,yield_P=y,yield_reference=yref,delta_yield=y-yref,delta_spike_probability=spike-ref,mass_error=mass,flow_error=flow,choi_min=cp,state_min=minimum,native_prediction=False))
                if d['label']=='upcj2N__full' and gamma==0 and name=='full':
                    fine=f.first_passage(.05*y,.05*yref,.012,'tonic_excess',n=2400)['firing_probability']
                    pde=f.pde_probability(.05*y,.05*yref,.012,'tonic_excess',dx=.1,dt=1e-5)
                    checks.append(dict(Fortet1200=spike,Fortet2400=fine,PDE=pde,absolute_error=abs(fine-pde['firing_probability'])))
        print(d['label'],'done',flush=True)
    pd.DataFrame(rows).to_csv(br.OUT/'conditional_products_spikes.csv',index=False)
    pd.DataFrame(retention).to_csv(br.OUT/'conditional_E_retention.csv',index=False)
    e.save(br.OUT/'propagation_checks.json',checks)
    e.save(br.OUT/'propagation_scope.json',dict(variable='Only SQ static QM EFG and HFC changed',
      fixed_RH='previous r12 embedded tensors, def2-svpd; no new R/H parameter identification',
      dynamics='same oxidized seed2026091521 dense6ps/2fs,1ps spectral window,whole protein orientation clamped',
      electrostatics='SQ induced density frozen and locally rotated; full external trajectory EFG added once; no instantaneous polarization',
      kinetics='triplet PT/3,kS1e7,escape4e6,supply/birth/Preturn10,Ereturn4 per second;20 cycles;J/dipolar0',
      neuron='G100pA,excess lifetime12ms,C20pF,I0=4pA,threshold10mV,sigma20mV/sqrt(s),no leak,50ms',
      native_T1=None,native_T2=None,native_gain=None,quantum_comparison='population_HQ shares single-RP quantum chemistry; not all-quantum versus all-classical'))
if __name__=='__main__':main()
