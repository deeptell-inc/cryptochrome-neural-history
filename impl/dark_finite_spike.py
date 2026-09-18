"""Time-dependent first passage for three explicit messenger-background models.

Fortet's first-kind renewal equation is discretized using bin masses at temporal
midpoints. Validate refinement and an independent absorbing Fokker-Planck solver.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.linalg import solve_banded
from . import dark_threshold as t
from . import dark_threshold_source as s

OUT=s.ROOT/'data/dark-finite-spike'
HALF=.02


def integrated_gate(M,times,tau):
    times=np.asarray(times)
    if np.isinf(tau):return times*M/(HALF+M)
    # log1p form avoids subtracting two nearly equal logarithms.
    decay=np.exp(-times/tau)
    return tau*np.log1p(M*(-np.expm1(-times/tau))/(HALF+M*decay))


def mean_voltage(times,M,Mref,tau,background,G=100.,I0=4.,C=20.):
    times=np.asarray(times)
    if background=='tonic_excess':
        delta=M-Mref;k=HALF+Mref
        if np.isinf(tau):addition=times*(M/(HALF+M)-Mref/(HALF+Mref))
        else:addition=HALF/k*tau*np.log1p(delta*(-np.expm1(-times/tau))/(k+delta*np.exp(-times/tau)))
    elif background=='compensated_pool':
        addition=integrated_gate(M,times,tau)-integrated_gate(Mref,times,tau)
    elif background=='uncompensated_pool':
        addition=integrated_gate(M,times,tau)-times*Mref/(HALF+Mref)
    else:raise ValueError(background)
    return 1000/C*(I0*times+G*addition)


def drift(times,M,Mref,tau,background,G=100.):
    times=np.asarray(times);decay=np.ones_like(times) if np.isinf(tau) else np.exp(-times/tau)
    gate=lambda x:x/(HALF+x)
    if background=='tonic_excess':diff=gate(Mref+(M-Mref)*decay)-gate(Mref)
    elif background=='compensated_pool':diff=gate(M*decay)-gate(Mref*decay)
    elif background=='uncompensated_pool':diff=gate(M*decay)-gate(Mref)
    else:raise ValueError(background)
    return 50*(4+G*diff)


def first_passage(M,Mref,tau,background,n=2400,horizon=.05,boundary=10.,sigma=20.,G=100.):
    dt=horizon/n;ends=np.arange(1,n+1)*dt;mids=ends-dt/2
    me=mean_voltage(ends,M,Mref,tau,background,G);mm=mean_voltage(mids,M,Mref,tau,background,G)
    rhs=ndtr((me-boundary)/(sigma*np.sqrt(ends)))
    masses=np.zeros(n)
    for i in range(n):
        k=ndtr((me[i]-mm[:i+1])/(sigma*np.sqrt(ends[i]-mids[:i+1])))
        masses[i]=(rhs[i]-np.dot(k[:i],masses[:i]))/k[i]
    # Preserve raw tiny negative quadrature errors in diagnostics; no clipping.
    cdf=np.r_[0,np.cumsum(masses)]
    rmst=horizon-np.dot(masses,horizon-mids)
    assert masses.min()>-1e-10 and cdf[-1]>=-1e-10 and cdf[-1]<=1+1e-10
    return dict(firing_probability=float(cdf[-1]),first_spike_RMST_s=float(rmst),minimum_bin_mass=float(masses.min()),n=n)


def pde_probability(M,Mref,tau,background,dx=.1,dt=1e-5,lower=-100.,horizon=.05,G=100.):
    """Independent centered finite-volume diffusion with absorbing endpoints.

    CN after two fully implicit startup steps. Integrate outgoing upper flux,
    not total lost mass, to distinguish the artificial lower boundary.
    """
    D=200.;upper=10.;count=round((upper-lower)/dx);dx=(upper-lower)/count
    x=lower+dx*np.arange(1,count);rho=np.zeros(len(x));rho[round(-lower/dx)-1]=1/dx
    steps=round(horizon/dt);dt=horizon/steps
    mus=drift((np.arange(steps)+.5)*dt,M,Mref,tau,background,G)
    upper_flux=0.;lower_flux=0.;min_density=0.
    diag=-2*D/dx**2
    for j,mu in enumerate(mus):
        lo=D/dx**2+mu/(2*dx);hi=D/dx**2-mu/(2*dx)
        assert min(lo,hi)>0
        theta=1. if j<2 else .5
        rhs=(1+(1-theta)*dt*diag)*rho
        rhs[1:]+=(1-theta)*dt*lo*rho[:-1];rhs[:-1]+=(1-theta)*dt*hi*rho[1:]
        ab=np.zeros((3,len(x)));ab[1]=1-theta*dt*diag;ab[0,1:]=-theta*dt*hi;ab[2,:-1]=-theta*dt*lo
        new=solve_banded((1,1),ab,rhs,check_finite=False)
        # Loss through the discretized absorbing neighbor is lo at the top,
        # hi at the bottom, consistent with the conservative finite-volume sum.
        upper_flux+=dt*dx*lo*((1-theta)*rho[-1]+theta*new[-1])
        lower_flux+=dt*dx*hi*((1-theta)*rho[0]+theta*new[0])
        min_density=min(min_density,float(new.min()));rho=new
    return dict(firing_probability=float(upper_flux),lower_boundary_loss=float(lower_flux),
        mass_residual=float(abs(rho.sum()*dx+upper_flux+lower_flux-1)),minimum_density=min_density,dx_mV=dx,dt_s=dt)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    source=pd.read_csv(s.OUT/'molecular.csv').query('retention==1 and cycles==20 and gamma_nuclear_s==10')
    rows=[]
    for basis,ge in [('pcJ2N',0.),('svpd',0.),('pcJ2N',1e8),('svpd',1e8)]:
        part=source[(source.basis==basis)&(source.gamma_oxygen_s==ge)].set_index('model')
        ref=float(part.loc['full','reference_product_fraction'])
        taus=[.012,.02,.05,.1,1.,np.inf] if basis=='pcJ2N' and ge==0 else [.012]
        for tau in taus:
            for background in ('tonic_excess','compensated_pool','uncompensated_pool'):
                models=('full','population_HQ') if tau==.012 and basis=='pcJ2N' and ge==0 else ('full',)
                base=first_passage(ref,ref,tau,background)
                for model in models:
                    M=float(part.loc[model,'product_fraction']);v=first_passage(M,ref,tau,background)
                    rows.append(dict(basis=basis,gamma_oxygen_s=ge,gamma_nuclear_s=10,model=model,background=background,
                        tau_messenger_s=tau if np.isfinite(tau) else None,M_initial=M,M_reference=ref,
                        baseline_firing_probability=base['firing_probability'],**v,
                        delta_firing_probability=v['firing_probability']-base['firing_probability'],
                        delta_RMST_s=v['first_spike_RMST_s']-base['first_spike_RMST_s']))
            print(basis,ge,'tau',tau,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'finite_spike.csv',index=False)
    s.save(OUT/'configuration.json',dict(date='2026-09-16',window_s=.05,neuron_C_pF=20,baseline_current_pA=4,threshold_margin_mV=10,
        noise_mV_sqrt_s=20,Gmax_pA=100,half_M=.02,n_time_bins=2400,
        lifetimes_s=[.012,.02,.05,.1,1,None],infinite_lifetime_encoding=None,
        tonic_excess='Mref maintained; only the excess M-Mref decays. Current offset fixed at initial reference.',
        compensated_pool='Both pools decay; an imposed common time-dependent background cancels reference current decay. Theoretical control, not inferred homeostasis.',
        uncompensated_pool='Both pools decay with the original fixed current offset; baseline current may become inhibitory.',
        biological_replicates=0,native_M_lifetime=None,CRY_lifetime_equals_M_lifetime=None,
        no_leak=True,comparison='nuclear-history full versus reset baseline, not all quantum versus classical; HQ population checked at .012 s'))


if __name__=='__main__':run()
