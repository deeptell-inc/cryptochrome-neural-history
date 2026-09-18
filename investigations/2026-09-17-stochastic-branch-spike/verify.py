"""Independent deterministic-chemistry Fokker--Planck and saved-run audits."""
from pathlib import Path
import sys,json,hashlib
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import stochastic_branch_spike as s
from impl import branch_current_gain as b
OUT=ROOT/'data/stochastic-branch-spike';HERE=Path(__file__).resolve().parent


def pde(y0,dy,p,noise,dx=.05,dt=.00001):
    # Independent continuous gate/channel expectation; no stochastic chemical/channel counts.
    def chemistry(t,z):
        fraction=y0+dy*np.exp(-t/p.tau_P_s)
        heq=b.gate(p.A*(fraction+p.eta_E*(1-fraction)))
        return [(heq-z[0])/p.tau_gate_s,(s.target_conductance(z[0],y0,p)-z[1])/noise.channel_tau_s]
    chem=solve_ivp(chemistry,(0,noise.horizon_s),[b.baseline(y0,p),p.gK_reference_nS],rtol=1e-10,atol=1e-13,dense_output=True)
    low=-30.;high=noise.threshold_mV
    nx=round((high-low)/dx);dx=(high-low)/nx;x=low+np.arange(1,nx)*dx
    steps=round(noise.horizon_s/dt);dt=noise.horizon_s/steps
    rho=np.zeros(len(x));rho[round(-low/dx)-1]=1/dx
    D=noise.background_sigma_mV_sqrt_s**2/2
    upper=lower=0.;cdf=[];minimum=0.
    for j in range(steps):
        g=chem.sol((j+.5)*dt)[1];total=p.gL_nS+g
        drive=noise.current_pA+(p.gK_reference_nS-g)*(p.Vref_mV-p.EK_mV)
        muplus=1000/p.C_pF*(drive-total*(x+dx/2));muminus=1000/p.C_pF*(drive-total*(x-dx/2))
        lo=D/dx**2+muminus/(2*dx);hi=D/dx**2-muplus/(2*dx)
        diag=-2*D/dx**2-(muplus-muminus)/(2*dx)
        assert min(lo.min(),hi.min())>0
        theta=1. if j<2 else .5
        rhs=(1+(1-theta)*dt*diag)*rho
        rhs[1:]+=(1-theta)*dt*lo[1:]*rho[:-1];rhs[:-1]+=(1-theta)*dt*hi[:-1]*rho[1:]
        ab=np.zeros((3,len(x)));ab[1]=1-theta*dt*diag;ab[0,1:]=-theta*dt*hi[:-1];ab[2,:-1]=-theta*dt*lo[1:]
        new=solve_banded((1,1),ab,rhs,check_finite=False)
        upper+=dt*dx*(D/dx**2+muplus[-1]/(2*dx))*((1-theta)*rho[-1]+theta*new[-1])
        lower+=dt*dx*(D/dx**2-muminus[0]/(2*dx))*((1-theta)*rho[0]+theta*new[0])
        minimum=min(minimum,float(new.min()));rho=new;cdf.append(upper)
    cdf=np.array(cdf);rmst=noise.horizon_s-np.trapezoid(np.r_[0,cdf],dx=dt)
    return dict(probability=upper,RMST_ms=rmst*1000,mass_residual=abs(rho.sum()*dx+upper+lower-1),
        lower_loss=lower,minimum_density=minimum,times=np.arange(1,steps+1)*dt,cdf=cdf)


if __name__=='__main__':
    config=json.loads((OUT/'configuration.json').read_text());y0=config['reference_yield'];dy=config['history_delta_yield']
    noise=s.Noise(**config['baseline_noise']);results=[]
    pooled=pd.read_csv(OUT/'pooled_summary.csv')
    for response,cap in [('linear_gate',.2),('bounded_amplifier',3.125)]:
        p=replace(b.Scenario(),tau_gate_s=.005,channel_response=response,capacity_nS=cap)
        trials=np.concatenate([np.load(OUT/f'{response}__background_only__seed{seed}.npz')['first_s'] for seed in config['seeds']],axis=1)
        for arm,delta in enumerate([0.,dy]):
            coarse=pde(y0,delta,p,noise,dx=.1,dt=.00002);fine=pde(y0,delta,p,noise)
            d=s.distribution(trials[arm]);cdf=np.mean(trials[arm,:,None]<=np.linspace(0,.05,101),axis=0)
            theory=np.interp(np.linspace(0,.05,101),np.r_[0,fine['times']],np.r_[0,fine['cdf']])
            error=abs(d['probability']-fine['probability'])
            assert error<5*d['probability_MC_SE']+.001
            assert fine['mass_residual']<1e-8 and fine['lower_loss']<1e-8 and fine['minimum_density']>-1e-9
            assert abs(fine['probability']-coarse['probability'])<.0002
            results.append(dict(response=response,arm=arm,MC_probability=d['probability'],PDE_probability=fine['probability'],
              MC_SE=d['probability_MC_SE'],probability_gap=d['probability']-fine['probability'],
              CDF_max_error=float(np.max(abs(cdf-theory))),PDE_refinement_gap=fine['probability']-coarse['probability'],
              MC_RMST_ms=d['RMST_ms'],PDE_RMST_ms=fine['RMST_ms'],mass_residual=fine['mass_residual']))
            np.savez_compressed(OUT/f'pde_{response}_arm{arm}.npz',time_s=fine['times'],cdf=fine['cdf'])
    pd.DataFrame(results).to_csv(OUT/'independent_PDE_checks.csv',index=False)
    refinement=[]
    for response in ['linear_gate','bounded_amplifier']:
        d=pooled[pooled.response==response].set_index('case')
        for tag in ['dt_0.0002','dt_5e-05']:
            gap=d.loc[tag,'delta_probability']-d.loc['both','delta_probability']
            se=np.hypot(d.loc[tag,'paired_MC_SE'],d.loc['both','paired_MC_SE'])
            assert abs(gap)<5*se+.001
            refinement.append(dict(response=response,case=tag,delta_probability_gap=gap,combined_MC_SE=se))
    pd.DataFrame(refinement).to_csv(OUT/'timestep_checks.csv',index=False)
    # Exact equal-pole covariance degeneracy: many variance splits, same observable.
    lag=np.linspace(0,.05,101)
    cov1=(1.+2.)*np.exp(-lag/.002);cov2=(.5+2.5)*np.exp(-lag/.002)
    assert np.array_equal(cov1,cov2)
    for record in json.loads((OUT/'preserved_inputs.json').read_text())['records']:
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256'],record['path']
    report=dict(independent_PDE_cases=len(results),max_PDE_mass_error=max(x['mass_residual'] for x in results),
       max_PDE_refinement_probability_gap=max(abs(x['PDE_refinement_gap']) for x in results),
       full_noise_timestep_conditions=len(refinement),equal_pole_covariance_gap=float(np.max(abs(cov1-cov2))),
       tests_log='investigations/2026-09-17-stochastic-branch-spike/pytest.txt',native_noise_calibrated=False)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
