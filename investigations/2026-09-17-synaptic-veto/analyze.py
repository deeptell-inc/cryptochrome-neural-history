from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.stats import binom
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import synaptic_veto as v
OUT=ROOT/'data/synaptic-veto'

def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False)+'\n')

def bound(mean,var,n,m=1):
    # Empirical Bernstein, D in [-100,100] pp. Nonzero even with no disagreements.
    # One-sided empirical Bernstein uses log(2/delta). Split alpha across
    # both tails AND m outputs because magnitude/sign is read after simulation.
    log=np.log(4*m/.05)
    return np.sqrt(2*var*log/n)+7*200*log/(3*(n-1))


def summarize():
    seeds=pd.read_csv(OUT/'seed_summary.csv')
    rows=[]; time_rows=[]; cdf=[]
    for (cid,mode,alloc),d in seeds.groupby(['condition_id','integration','allocation'],sort=False):
        r=d.iloc[0].to_dict();r.pop('seed');n=int(d.n.sum());r['n']=n;r['seeds']=len(d)
        for label in ['stop','continue','unresolved']:
            mean=float(np.average(d[f'delta_{label}_pp'],weights=d.n))
            disagreements=int(d[f'{label}_paired_disagreements'].sum())
            variance=max(0,(10000*disagreements-n*mean**2)/(n-1))
            se=np.sqrt(variance/n)
            r[f'delta_{label}_pp']=mean;r[f'{label}_MC_SE_pp']=se
            r[f'{label}_MC95_low_pp']=mean-1.96*se;r[f'{label}_MC95_high_pp']=mean+1.96*se
            rad=bound(mean,variance,n)
            r[f'{label}_conservative95_radius_pp']=rad
            r[f'{label}_variance_pp2']=variance
            r[f'{label}_ref']=float(np.average(d[f'{label}_ref'],weights=d.n))
            r[f'{label}_source']=float(np.average(d[f'{label}_source'],weights=d.n))
            r[f'{label}_seed_SD_pp']=float(d[f'delta_{label}_pp'].std(ddof=1))
        for col in d.columns:
            if col.startswith(('expected_','soma_','trigger_failure_','trigger_late_','chemical_arrival_','posttrigger_','weighted_','chemical_input_','independent_')):
                r[col]=float(np.average(d[col],weights=d.n))
        # Re-read all trials, not mean conditional medians. Correct censor horizons.
        names=v.ALLOCS if r['group']!='grid' else ('top',)
        parts=[]
        for index,seed_row in d.iterrows():
            tt=np.load(OUT/'trials'/f'{cid}_{int(seed_row.seed)}.npz')['times'][v.MODES.index(mode),names.index(alloc)]
            parts.append(tt)
            cap=min(r['horizon'],r['cue']+.25)
            dd=(np.minimum(tt[1,:,1],cap)-np.minimum(tt[0,:,1],cap))*1000
            seeds.loc[index,'trigger_delta_RMST_ms']=dd.mean()
            seeds.loc[index,'trigger_RMST_MC_SE_ms']=dd.std(ddof=1)/np.sqrt(len(dd))
        all_t=np.concatenate(parts,axis=1)
        for idx,label in enumerate(v.ROUTES):
            cap=min(r['horizon'],r['cue']+.25) if label=='trigger' else r['horizon']
            dt=(np.minimum(all_t[1,:,idx],cap)-np.minimum(all_t[0,:,idx],cap))*1000
            r[f'{label}_delta_RMST_ms']=float(dt.mean());r[f'{label}_RMST_MC_SE_ms']=float(dt.std(ddof=1)/np.sqrt(n))
            for arm,an in enumerate(['ref','source']):
                t=all_t[arm,:,idx];hit=np.isfinite(t)
                if label=='trigger':lat=t-r['cue']
                elif label=='stop':
                    lat=np.full(n,np.inf);valid=hit&np.isfinite(all_t[arm,:,1]);lat[valid]=t[valid]-all_t[arm,valid,1]
                else:lat=t
                finite=np.isfinite(lat)
                rr=dict(condition_id=cid,integration=mode,allocation=alloc,arm=an,stage=label,n=n,
                    observed=int(hit.sum()),censored=int((~hit).sum()),RMST_absolute_ms=float(np.minimum(t,cap).mean()*1000),censor_time_s=cap,
                    latency_q10_s=float(np.quantile(lat[finite],.1)) if finite.any() else None,
                    latency_q50_s=float(np.quantile(lat[finite],.5)) if finite.any() else None,
                    latency_q90_s=float(np.quantile(lat[finite],.9)) if finite.any() else None)
                time_rows.append(rr)
                r[f'{label}_{an}_median_s']=float(np.median(t[hit])) if hit.any() else None
                r[f'{label}_{an}_censor']=float((~hit).mean())
                if r['group']=='confirmation':
                    for tt in np.arange(0,1.0001,.01):
                        cdf.append(dict(condition_id=cid,integration=mode,allocation=alloc,arm=an,stage=label,
                            time_s=tt,absolute_CDF=float((t<=tt+1e-12).mean()),latency_CDF=float((lat<=tt+1e-12).mean()),n=n))
        rows.append(r)
    seeds.to_csv(OUT/'seed_summary.csv',index=False)
    for cid,seed_rows in seeds.groupby('condition_id',sort=False):
        seed_rows.to_csv(OUT/f'{cid}.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(time_rows).to_csv(OUT/'time_distributions.csv',index=False)
    pd.DataFrame(cdf).to_csv(OUT/'confirmation_CDF.csv',index=False)
    grid=df[df.group=='grid'].copy();m=len(grid)
    grid['simultaneous_radius_pp']=[bound(r.delta_stop_pp,r.stop_variance_pp2,r.n,m) for r in grid.itertuples()]
    grid['simultaneous_abs_lower_pp']=np.maximum(0,abs(grid.delta_stop_pp)-grid.simultaneous_radius_pp)
    grid.to_csv(OUT/'grid_intervals.csv',index=False)
    minima=[]
    for axis,others in [('gain',['route','integration','k','tau']),('k',['route','integration','gain','tau']),('tau',['route','integration','gain','k'])]:
        for key,d in grid.groupby(others):
            for target in [.5,1.,5.]:
                point=d[abs(d.delta_stop_pp)>=target];confirmed=d[d.simultaneous_abs_lower_pp>=target]
                minima.append(dict(axis=axis,**dict(zip(others,key)),target_pp=target,
                    point_min=float(point[axis].min()) if len(point) else None,
                    simultaneous_min=float(confirmed[axis].min()) if len(confirmed) else None,
                    tested_min=float(d[axis].min()),tested_max=float(d[axis].max()),
                    interpretation='lowest tested qualifying value; not a continuous minimum; direction in grid_intervals.csv'))
    pd.DataFrame(minima).to_csv(OUT/'conditional_minima.csv',index=False)
    return df


def packet_audit():
    """Independent binomial quadrature: mean/variance BEFORE arrival delays.

    Includes same-uniform release correlation exactly (mixture rho) and common
    molecular noise using total variance, not the independent-release formula.
    """
    q0,ys=v.sources();records=[]
    for N in [100,10000,1000000]:
      for rho in [0.,.3]:
       c=v.Config(gain=1024,pool=N,rho=rho,tau=.2)
       w,delay,a,_=v.architecture(c);ww=w/(np.linalg.norm(w)*np.sqrt(c.p0*(1-c.p0)))
       for age in [.01,.05,.15]:
        for source in ['reset','full']:
         q=q0+(ys['full']-q0)*np.exp(-age/c.tau)*(source=='full')
         lo=max(0,int(binom.ppf(1e-13,N,q)));hi=min(N,int(binom.ppf(1-1e-13,N,q))+1)
         count=np.arange(lo,hi+1);mass=binom.pmf(count,N,q)
         for ai,name in enumerate(v.ALLOCS):
          p=v.probability(count/N,c,a[ai],q0)
          mean=(p-c.p0)@ww
          independent=(p*(1-p)*ww**2).sum(-1)
          selected=a[ai]>0;ps=p[:,selected][:,0]
          U=ww[selected].sum();V=ww[~selected].sum()
          shared=U**2*ps*(1-ps)+V**2*c.p0*(1-c.p0)+2*U*V*(np.minimum(ps,c.p0)-ps*c.p0)
          m=float(mass@mean);varchem=float(mass@mean**2-m**2)
          release=float(mass@((1-rho)*independent+rho*shared))
          records.append(dict(pool=N,rho=rho,age_s=age,source=source,allocation=name,
              total_probability_shift=float(mass@(p-c.p0).sum(-1)),
              weighted_mean=m,chemical_variance=varchem,conditional_release_variance=release,
              total_variance=varchem+release,omitted_probability_mass=float(1-mass.sum())))
    pd.DataFrame(records).to_csv(OUT/'exact_packet_moments.csv',index=False)


def checks(df):
    # Trial-level cross-check against summary; exact replay; source controls.
    plans=json.loads((OUT/'planned_conditions.json').read_text());seeds=[26091711,26091712,26091713]
    eq=[];partitions=True;causal=True;maxerr=0.
    for ci,p in enumerate(plans):
        cid=f'{ci:03d}_{p["group"]}'
        for seed in seeds:
            z=np.load(OUT/'trials'/f'{cid}_{seed}.npz');t=z['times'];o=z['outcomes']
            partitions &= bool(np.all(o.sum(-1)==1))
            # Independent logical recomputation: finite stop earlier than go wins.
            independent=np.isfinite(t[...,2])&(t[...,0]>t[...,2])
            assert np.array_equal(independent,o[...,0])
            valid=np.isfinite(t[...,2]);causal &= bool(np.all(t[...,2][valid]>=t[...,1][valid]+.07*p['config']['latency_scale']-1e-12))
            if p['config']['source']=='population':
                full=np.load(OUT/'trials'/f'{ci-1:03d}_main_{seed}.npz')['times']
                eq.append(bool(np.array_equal(t,full)))
            if p['config']['source']=='reset' or p['config']['gain']==0:
                assert np.array_equal(t[:,:,0],t[:,:,1])
            if p['group']=='confirmation':
                for m,mode in enumerate(v.MODES):
                    for ai,al in enumerate(v.ALLOCS):
                        raw=100*(o[m,ai,1,:,0].astype(float)-o[m,ai,0,:,0]).mean()
                        seed_df=pd.read_csv(OUT/f'{cid}.csv')
                        stored=seed_df.query('seed==@seed and integration==@mode and allocation==@al').delta_stop_pp.iloc[0]
                        maxerr=max(maxerr,abs(raw-stored))
    checks=dict(partitions=partitions,stop_causal=causal,all_full_population_trial_times_identical=all(eq),
        full_population_pairs=len(eq),max_raw_vs_seed_summary_pp=maxerr,max_budget_error=float(df.budget_error.max()),
        p_min=float(df.p_min.min()),p_max=float(df.p_max.max()),
        replay_checks=[x for s in seeds for x in json.loads((OUT/f'replay_check_{s}.json').read_text())],
        grid_conditions=int((df.group=='grid').sum()),all_biological_coefficients_calibrated=False,
        physical_experiments_performed=0)
    assert partitions and causal and maxerr<1e-10 and checks['max_budget_error']<1e-10
    save('verification.json',checks)


def plot(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    confirm=df[df.group=='confirmation']
    colors=['#999999','#d6a02d','#177e89','#7d6a9c','#c44747']
    for ax,mode in zip(axs.flat,v.MODES):
        for ai,al in enumerate(v.ALLOCS):
            y=[];e=[]
            for route in v.ROUTES:
                r=confirm.query('integration==@mode and allocation==@al and route==@route and tau==.2').iloc[0]
                y.append(r.delta_stop_pp);e.append(1.96*r.stop_MC_SE_pp)
            ax.errorbar(np.arange(3)+(ai-2)*.13,y,yerr=e,fmt='o',color=colors[ai],label=al,capsize=2)
        ax.axhline(0,color='k',lw=.7);ax.axhline(5,color='gray',ls=':',lw=.7);ax.axhline(-5,color='gray',ls=':',lw=.7)
        ax.set_xticks(range(3),v.ROUTES);ax.set_title(mode);ax.set_ylabel('Stop success change (pp)')
    axs[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('Conditional sparse modulation: 200 ms chemical lifetime\n24,576 paired trials; error bars = 95% Monte Carlo intervals; assumed gain 1024')
    fig.savefig(ROOT/'figures/synaptic-veto/route_allocation.png',dpi=160);plt.close(fig)
    grid=pd.read_csv(OUT/'grid_intervals.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,route in zip(axs,v.ROUTES):
        for tau in [.012,.05,.2]:
            d=grid.query('route==@route and k==8 and integration=="temporal" and tau==@tau').sort_values('gain')
            ax.errorbar(d.gain,d.delta_stop_pp,yerr=1.96*d.stop_MC_SE_pp,marker='o',label=f'{1000*tau:g} ms')
        ax.set_xscale('symlog',linthresh=50);ax.axhline(0,color='k',lw=.5);ax.set_title(route);ax.set_xlabel('Gain (probability budget / yield fraction)');ax.set_ylabel('Stop change (pp)');ax.legend()
    fig.suptitle('Conditional gain / lifetime curves: top 8, temporal integration')
    fig.savefig(ROOT/'figures/synaptic-veto/conditions.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    d=summarize();packet_audit();checks(d);plot(d);print('analysis complete',len(d))
