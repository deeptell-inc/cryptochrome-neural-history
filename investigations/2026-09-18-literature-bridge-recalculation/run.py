from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import literature_bridge_recalculation as r
from impl.chemical_carrier_mapping import cofactor_generator
OUT=ROOT/'data/literature-bridge-recalculation';OUT.mkdir(exist_ok=True)

def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def freeze():
    path=OUT/'preserved_inputs.json'
    if not path.exists():
        old=json.loads((ROOT/'data/chemical-carrier-mapping/preserved_inputs.json').read_text())['records']
        old+=json.loads((ROOT/'audit/chemical-carrier-mapping-manifest.json').read_text())
        old={x['path']:x for x in old}
        save('preserved_inputs.json',dict(records=list(old.values())))
    for x in json.loads(path.read_text())['records']:
        assert hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']

def main():
    freeze()
    fit=json.loads((ROOT/'data/hk-redox-update/fits.json').read_text())['fits']['dynamic']['parameters']
    k=fit['k_rest_per_min']/60
    save('spec.json',dict(interpretation_of_latest_values='a0=10/s primary, .1 and 100/s sensitivity; NADP off=.01,1,100/s; NADPH binding=100/s; oxidized rebinding=0',
        input='a(t)=a0 exp(-t/tau); same exponential shape as old task, NOT the rectangular pulse from latest table',
        tau_s=[.012,.05,.2],old_k_per_s=k,initial_Hk=[1,0,0],baseline_readout='zero oxidized occupancy; source is externally prescribed writing rate',
        native_parameters_identified=False,carbonyl_concentration_waveform_equivalence='only in unsaturated mass-action limit; kcat/KM and local dose unknown',
        amplitude_matching='time profiles matched; physical chemical dose not matched to old dimensionless u',
        contrast='independently specified +1% in a0, not derived from full-reset yield',new_experiments=0))
    rows=[];traces=[];retain=[];interventions=[];err=0.;mass=0.;minp=1.;tangent_error=0.
    for a0 in [.1,10.,100.]:
      for off in [.01,1.,100.]:
       for tau in [.012,.05,.2]:
        sol=r.solve_pulse(a0,off,100,tau);verify=r.solve_pulse(a0,off,100,tau,method='Radau')
        t=np.unique(np.r_[0,.04,.11,.2,.4,2.,np.geomspace(1e-6,10,400)])
        p=sol.sol(t);err=max(err,float(abs(p-verify.sol(t)).max()));mass=max(mass,float(abs(p[:3].sum(0)-1).max()));minp=min(minp,float(p[:3].min()))
        tp,yp=r.peak(sol);ss=r.held_steady_occupancy(a0,off,100)
        oldtp=np.log((1/tau)/k)/(1/tau-k);oldpeak=float(r.old_pulse(oldtp,k,tau))
        rows.append(dict(a0_s=a0,off_s=off,binding_s=100,tau_s=tau,new_peak_time_s=tp,new_peak_oxidized_fraction=yp,
            new_held_steady_oxidized_fraction=ss,new_peak_over_own_steady=yp/ss,
            old_peak_time_s=oldtp,old_peak_per_unit_steady_shift=oldpeak,
            own_steady_normalized_ratio_not_native_amplification=yp/ss/oldpeak,
            oxidized_at_200ms=sol.sol(.2)[1],oxidized_at_2s=sol.sol(2)[1],
            d_oxidized_d_log_a_at_200ms=sol.sol(.2)[4],d_oxidized_d_log_a_at_2s=sol.sol(2)[4]))
        for j,tt in enumerate(t):traces.append(dict(a0_s=a0,off_s=off,tau_s=tau,time_s=tt,reduced=p[0,j],oxidized=p[1,j],apo=p[2,j],
            sensitivity_d_oxidized_d_log_a=p[4,j],normalized_own_steady=p[1,j]/ss,old_normalized=float(r.old_pulse(tt,k,tau))))
        if a0==10:
            eps=1e-4;plus=r.solve_pulse(a0*np.exp(eps),off,100,tau);minus=r.solve_pulse(a0*np.exp(-eps),off,100,tau)
            tangent_error=max(tangent_error,float(abs((plus.sol(t)[:3]-minus.sol(t)[:3])/(2*eps)-p[3:]).max()))
            source=r.solve_pulse(a0*1.01,off,100,tau)
            for tt in [0,.04,.11,.2,.4,2.]:
                base=float(sol.sol(tt)[1]);full=float(source.sol(tt)[1]);delta=full-base
                for name,source_delta,readout_gain in [('intact',.01,1),('selective_readout_block',.01,0),('rescue',.01,1),('source_contrast_block',0,1)]:
                    interventions.append(dict(off_s=off,tau_s=tau,time_s=tt,intervention=name,
                      a0_reference_s=a0,a0_source_s=a0*(1+source_delta),
                      reference_oxidized=base,source_oxidized=full if source_delta else base,
                      raw_oxidized_contrast=delta if source_delta else 0.,
                      observed_occupancy_contrast=readout_gain*delta if source_delta else 0.,
                      source_contrast_preserved=bool(source_delta),readout_gain_assumed=readout_gain,
                      native_release_gain=None,native_stop_probability_difference=None))
    for off in [.01,1,100]:
      for tt in [.2,.4,2.]:
        p=expm(cofactor_generator(0,off,100,0)*tt)@np.array([0.,1.,0.]);exact=np.exp(-off*tt)
        assert abs(p[1]-exact)<1e-12
        retain.append(dict(off_s=off,time_s=tt,preloaded_oxidized_remaining_fraction=p[1],apo_fraction=p[2],reduced_fraction=p[0],
            old_same_pole_assumed_retention=np.exp(-k*tt)))
    pd.DataFrame(rows).to_csv(OUT/'peak_comparison.csv',index=False)
    pd.DataFrame(traces).to_csv(OUT/'traces.csv',index=False)
    pd.DataFrame(retain).to_csv(OUT/'preloaded_retention.csv',index=False)
    df=pd.DataFrame(interventions);df.to_csv(OUT/'virtual_block_rescue.csv',index=False)
    intact=df.query('intervention=="intact"').observed_occupancy_contrast.to_numpy()
    np.testing.assert_array_equal(intact,df.query('intervention=="rescue"').observed_occupancy_contrast.to_numpy())
    assert (df.query('intervention=="selective_readout_block" or intervention=="source_contrast_block"').observed_occupancy_contrast==0).all()
    print('validation diagnostics',err,mass,minp,tangent_error,flush=True)
    assert err<2e-8 and mass<1e-9 and minp>-1e-10 and tangent_error<2e-8
    # Frozen old analytic output is independently reproduced, not silently replaced.
    old=pd.read_csv(ROOT/'data/literature-virtual-bridge/normalized_bridge.csv');old=old.query('rate_case=="point_fit"')
    olderr=max(abs(r.old_pulse(row.time_s,k,row.chemical_pulse_tau_s)-row.new_write_per_unit_equilibrium_shift) for row in old.itertuples())
    assert olderr<1e-14
    save('checks.json',dict(scenarios=len(rows),independent_DOP853_Radau_error=err,mass_error=mass,min_population=minp,
        tangent_finite_difference_error=tangent_error,old_frozen_reproduction_error=float(olderr),
        preloaded_matrix_vs_analytic=True,virtual_rescue_identity=True,native_stop_probability_difference=None))
    print(pd.DataFrame(rows).query('a0_s==10')[['off_s','tau_s','new_peak_time_s','new_peak_oxidized_fraction','new_peak_over_own_steady','old_peak_per_unit_steady_shift']].to_string(index=False))
    print('checks',err,tangent_error,flush=True)

if __name__=='__main__':main()
