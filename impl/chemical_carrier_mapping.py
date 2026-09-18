"""Chemical identity and dimensions for a CONDITIONAL Hk cofactor carrier.

No native kinetic or synaptic default. P/E yields are reaction branches, not
4-ONE concentrations. A three-state binding model separates oxidation from
cofactor release and replacement. Concentrations are uM; times are seconds.
"""
import numpy as np
from scipy.linalg import expm
from scipy.special import expit,logit

MOLECULES_PER_uM_um3 = 602.214076  # exact SI Avogadro conversion

def nonnegative(*args):
    if not np.all(np.isfinite(args)) or min(args)<0:raise ValueError('finite nonnegative values required')

def molecule_count(concentration_uM,volume_um3):
    nonnegative(concentration_uM,volume_um3)
    return MOLECULES_PER_uM_um3*concentration_uM*volume_um3

def effective_sites(sites,correlation):
    if not isinstance(sites,(int,np.integer)) or sites<1 or not 0<=correlation<=1:raise ValueError('sites/correlation')
    return sites/(1+(sites-1)*correlation)

def peroxide_path(q,t,escape_oxidation_s,clearance_s,*,direct_delivery,early_escape_delivery,late_escape_delivery):
    """Mean peroxide equivalents per single completed HQ cycle.

    Full escaped SQ reoxidation; two superoxide equivalents dismutate to one
    peroxide. Instantaneous dismutation is a theoretical limiting case, not a
    measured target rate. Delivery fractions explicit; no 4-ONE conversion.
    States: delayed peroxide potential, peroxide, cumulative production.
    """
    nonnegative(t,escape_oxidation_s,clearance_s)
    if not 0<=q<=1:raise ValueError('branch probability')
    delivery=[direct_delivery,early_escape_delivery,late_escape_delivery]
    if not np.all(np.isfinite(delivery)) or any(x<0 or x>1 for x in delivery):raise ValueError('delivery fractions')
    immediate=direct_delivery*q+.5*early_escape_delivery*(1-q)
    delayed=.5*late_escape_delivery*(1-q)
    A=np.array([[-escape_oxidation_s,0,0],[escape_oxidation_s,-clearance_s,0],[escape_oxidation_s,0,0]])
    return expm(A*t)@np.array([delayed,immediate,immediate])

def cofactor_generator(write_s,off_s,bind_reduced_s,bind_oxidized_s):
    """Column generator for R=Hk:NADPH, O=Hk:NADP+, U=apo.

    R->O consumes a carbonyl substrate; O->U releases NADP+; U->R binds
    NADPH. U->O is oxidized-cofactor rebinding. This is not direct NADP+
    reduction inside Hk. Reduced cofactor dissociation is omitted explicitly.
    """
    nonnegative(write_s,off_s,bind_reduced_s,bind_oxidized_s)
    a,b,h,o=write_s,off_s,bind_reduced_s,bind_oxidized_s
    return np.array([[-a,0,h],[a,-b,o],[0,b,-h-o]])

def cofactor_step(initial,duration_s,*,write_s,off_s,bind_reduced_s,bind_oxidized_s):
    p=np.asarray(initial,float)
    if p.shape!=(3,) or np.any(p<0) or not np.isclose(p.sum(),1):raise ValueError('cofactor population')
    nonnegative(duration_s)
    return expm(cofactor_generator(write_s,off_s,bind_reduced_s,bind_oxidized_s)*duration_s)@p

def write_rate(concentration_uM,*,kcat_s,KM_uM):
    nonnegative(concentration_uM,kcat_s,KM_uM)
    if KM_uM==0:raise ValueError('positive KM required')
    return kcat_s*concentration_uM/(KM_uM+concentration_uM)

def recovery_mfpt(off_s,bind_reduced_s,bind_oxidized_s):
    """First passage O->R, including retries through apo and NADP+ rebinding."""
    nonnegative(off_s,bind_reduced_s,bind_oxidized_s)
    if off_s==0 or bind_reduced_s==0:return np.inf
    return (bind_reduced_s+bind_oxidized_s)/(bind_reduced_s*off_s)+1/bind_reduced_s

def release_probability(z,z0,*,p0,beta):
    if not 0<p0<1 or not np.isfinite(beta):raise ValueError('release calibration')
    z=np.asarray(z,float)
    if not np.isfinite(z0) or not 0<=z0<=1 or not np.all(np.isfinite(z)) or np.any((z<0)|(z>1)):raise ValueError('occupancy')
    return expit(logit(p0)+beta*(z-z0))

def required_occupancy_delta(delta_p,*,p0,beta):
    if not 0<p0<1 or not 0<p0+delta_p<1 or not np.isfinite(beta) or beta==0:raise ValueError('target/gain')
    return (logit(p0+delta_p)-logit(p0))/beta

def cofactor_release(populations,reference,*,p0,beta_oxidized,beta_apo):
    """Conditional-on-arriving-AP release; apo effect must be explicit too."""
    p=np.asarray(populations,float);r=np.asarray(reference,float)
    if p.shape[-1]!=3 or r.shape!=(3,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
        raise ValueError('cofactor shape/finite')
    if np.any(p<0) or np.any(r<0) or not np.allclose(p.sum(-1),1) or not np.isclose(r.sum(),1):raise ValueError('cofactor populations')
    if not 0<p0<1 or not np.all(np.isfinite([beta_oxidized,beta_apo])):raise ValueError('explicit gain required')
    return expit(logit(p0)+beta_oxidized*(p[...,1]-r[1])+beta_apo*(p[...,2]-r[2]))

def require_identified(record,keys):
    missing=[k for k in keys if record.get(k) is None]
    if missing:raise ValueError('unidentified native parameters: '+','.join(missing))
    return {k:record[k] for k in keys}

def finite_carbonyl_trials(*,sites,carbonyls,volume_um3,k2_uM_s,off_s,bind_reduced_s,clearance_s,n,seed,times_s):
    """Exact Gillespie paths for a finite carbonyl bolus and Hk cofactor pool.

    C+R->C_red+O; O->U+free NADP+; U+NADPH->R; C->cleared.
    Buffered NADPH/H+ bath, no NADP+ rebinding in this finite-pool test.
    All rate constants and molecule counts must be specified. No chemical
    molecule can write twice; no replenishment or regenerative carbonyl source.
    """
    nonnegative(volume_um3,k2_uM_s,off_s,bind_reduced_s,clearance_s)
    if volume_um3<=0 or any(not isinstance(x,(int,np.integer)) for x in [sites,carbonyls,n]) or sites<1 or carbonyls<0 or n<2:raise ValueError('pool sizes')
    times=np.asarray(times_s,float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(np.r_[0,times])<=0):raise ValueError('observation times')
    rng=np.random.default_rng(seed)
    # C,R,O,U,converted,cleared,freeNADP_produced,NADPH_bound
    x=np.zeros((n,8),dtype=np.int64);x[:,0]=carbonyls;x[:,1]=sites
    changes=np.array([[-1,-1,1,0,1,0,0,0],[0,0,-1,1,0,0,1,0],
                      [0,1,0,-1,0,0,0,1],[-1,0,0,0,0,1,0,0]])
    result=[];start=0.
    for endpoint in times:
        clock=np.full(n,start)
        while True:
            ids=np.flatnonzero(clock<endpoint)
            if len(ids)==0:break
            y=x[ids]
            rates=np.stack([k2_uM_s*y[:,0]*y[:,1]/molecule_count(1,volume_um3),
                off_s*y[:,2],bind_reduced_s*y[:,3],clearance_s*y[:,0]],axis=1)
            total=rates.sum(1);wait=np.full(len(ids),np.inf)
            active=total>0;wait[active]=rng.exponential(1/total[active])
            event=clock[ids]+wait<endpoint
            clock[ids[~event]]=endpoint
            use=ids[event]
            if len(use):
                draw=rng.random(len(use))*total[event]
                typ=(draw[:,None]>=rates[event].cumsum(1)).sum(1)
                x[use]+=changes[typ];clock[use]+=wait[event]
        assert np.all(x>=0)
        assert np.all(x[:,1:4].sum(1)==sites)
        assert np.all(x[:,[0,4,5]].sum(1)==carbonyls)
        assert np.all(x[:,2]==x[:,4]-x[:,6])
        assert np.all(x[:,3]==x[:,6]-x[:,7])
        result.append(x.copy());start=float(endpoint)
    return np.array(result)
