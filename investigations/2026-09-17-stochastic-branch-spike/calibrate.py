"""Synthetic independent assays, no history firing used in any fit."""
from pathlib import Path
import sys,json
from dataclasses import replace,asdict
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import stochastic_branch_spike as s
from impl import branch_current_gain as b


def step_g(times,y0,y,p,tau_channel):
    h0=b.baseline(y0,p);hinf=b.gate(p.A*y)
    def rhs(t,z):
        h=hinf+(h0-hinf)*np.exp(-t/p.tau_gate_s)
        return [(s.target_conductance(h,y0,p)-z[0])/tau_channel]
    return solve_ivp(rhs,(0,times[-1]),[p.gK_reference_nS],t_eval=times,rtol=1e-9,atol=1e-12).y[0]


def calibrate(seed,y0,folder):
    folder.mkdir(exist_ok=True,parents=True);rng=np.random.default_rng(seed)
    true=s.Noise();p0=b.Scenario(tau_gate_s=.005)
    mol=s.count_traces(true.molecules,y0,.012,.0005,1200,24,rng)
    ch=s.count_traces(true.channels,.5,.002,.0001,2000,24,rng)
    mf=s.fit_count_noise(mol[:16],y0,.0005,12);mv=s.fit_count_noise(mol[16:],y0,.0005,12)
    cf=s.fit_count_noise(ch[:16],.5,.0001,10);cv=s.fit_count_noise(ch[16:],.5,.0001,10)
    # Independent membrane assay: fixed conductance, known current, no chemical/channel noise.
    dt=.0002;nt=1500;currents=np.repeat([0.,4.,8.],24);V=np.zeros((72,nt+1))
    decay=np.exp(-dt/.02);sd=20*np.sqrt((1-decay**2)/100)
    for j in range(nt):V[:,j+1]=currents+(V[:,j]-currents)*decay+sd*rng.normal(size=72)
    keep=np.r_[np.arange(16),24+np.arange(16)]
    X=np.column_stack([V[keep,:-1].ravel(),np.repeat(currents[keep],nt)])
    a,beta=np.linalg.lstsq(X,V[keep,1:].ravel(),rcond=None)[0]
    rate=-np.log(a)/dt;gt=(1-a)/beta;C=1000*gt/rate
    residual=V[keep,1:].ravel()-X@np.array([a,beta]);sigma=np.sqrt(np.mean(residual**2)*2*rate/(1-a*a))
    noise=replace(true,molecules=max(1,round(mf['effective_count'])),channels=max(1,round(cf['effective_count'])),
        channel_tau_s=cf['tau_s'],background_sigma_mV_sqrt_s=float(sigma))
    rows=[];params={};arrays=dict(molecular_fraction=mol,channel_open_fraction=ch,membrane_voltage_mV=V,membrane_current_pA=currents)
    doses=y0+np.array([-.08,-.03,-.01,0,.01,.03,.08]);times=np.linspace(0,.04,81)
    for response,capacity in [('linear_gate',.2),('bounded_amplifier',3.125)]:
        p=replace(p0,channel_response=response,capacity_nS=capacity)
        h=b.gate(p.A*doses)
        expected=s.target_conductance(h,y0,p)
        obs=expected[None,:]+rng.normal(0,.001,(24,len(doses)))
        fit=least_squares(lambda l:(s.target_conductance(h,y0,replace(p,capacity_nS=float(np.exp(l[0]))))-obs[:16].mean(0))/.00025,[np.log(capacity*.8)])
        cap=float(np.exp(fit.x[0]));curve=step_g(times,y0,y0+.05,p,true.channel_tau_s)
        raw=curve[None,:]+rng.normal(0,.001,(24,len(times)))
        fit2=least_squares(lambda l:(step_g(times,y0,y0+.05,replace(p,capacity_nS=cap,tau_gate_s=float(np.exp(l[0]))),noise.channel_tau_s)-raw[:16].mean(0))/.00025,[np.log(.004)])
        tg=float(np.exp(fit2.x[0]))
        fitted=replace(p,capacity_nS=cap,tau_gate_s=tg,tau_P_s=mf['tau_s'],tau_E_s=mf['tau_s'],C_pF=float(C),gL_nS=float(gt-.2))
        params[response]=dict(scenario=asdict(fitted),noise=asdict(noise))
        rows.append(dict(seed=seed,response=response,molecules=noise.molecules,channels=noise.channels,
            molecular_tau_s=mf['tau_s'],channel_tau_s=noise.channel_tau_s,C_pF=C,total_g_nS=gt,sigma_mV_sqrt_s=sigma,
            capacity_nS=cap,tau_gate_s=tg,dose_validation_RMSE_nS=float(np.sqrt(np.mean((s.target_conductance(h,y0,fitted)-obs[16:].mean(0))**2))),
            step_validation_RMSE_nS=float(np.sqrt(np.mean((step_g(times,y0,y0+.05,fitted,noise.channel_tau_s)-raw[16:].mean(0))**2)))))
        arrays[response+'_dose_g_nS']=obs;arrays[response+'_step_g_nS']=raw
    arrays['dose_P_fraction']=doses;arrays['step_times_s']=times
    np.savez_compressed(folder/'calibration_raw.npz',**arrays)
    (folder/'fitted_parameters.json').write_text(json.dumps(params,indent=2)+'\n')
    (folder/'count_validation.json').write_text(json.dumps(dict(molecular_fit=mf,molecular_validation=mv,channel_fit=cf,channel_validation=cv),indent=2)+'\n')
    return params,rows
