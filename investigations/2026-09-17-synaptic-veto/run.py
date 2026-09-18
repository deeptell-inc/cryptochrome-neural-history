from pathlib import Path
from dataclasses import replace,asdict
import sys,json,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import synaptic_veto as v
HERE=Path(__file__).resolve().parent;OUT=ROOT/'data/synaptic-veto'
SEEDS=[26091711,26091712,26091713]

def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def freeze():
    path=OUT/'preserved_inputs.json'
    if not path.exists():
        old=json.loads((ROOT/'data/hk-redox-update/preserved_inputs.json').read_text())['records']
        old+=json.loads((ROOT/'audit/hk-redox-update-manifest.json').read_text())
        old={r['path']:r for r in old}
        # Freeze priority inputs even if absent from inherited manifest.
        for f in ['SPARSE_SYNAPTIC_RESULTS.md','DARK_FINITE_SPIKE_RESULTS.md','POPULATION_MEMORY_RESULTS.md','CONDITIONAL_ACTION_RESULTS.md','LLNV_PATHWAYS_RESULTS.md','impl/sparse_synaptic.py','impl/synaptic_validation.py','impl/stochastic_branch_spike.py']:
            p=ROOT/f;old[f]=dict(path=f,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
        save(path,dict(records=list(old.values())))
    for r in json.loads(path.read_text())['records']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']


def configs():
    base=v.Config()
    out=[]
    for tau in [.012,.2]:
        for route in v.ROUTES:
            for source in ['full','population','reset','classical_mean']:
                out.append(('main',replace(base,tau=tau,route=route,source=source,gain=1024),1024,v.ALLOCS))
    for route in v.ROUTES:
        for tau in [.012,.2]:
            out.append(('confirmation',replace(base,tau=tau,route=route,gain=1024),8192,v.ALLOCS))
    for route in v.ROUTES:
        for change in [dict(rho=.3),dict(pool=100),dict(pool=1000000),dict(offset=-.4),dict(offset=.4),
                       dict(threshold=.75),dict(threshold=1.25),dict(architecture=26091699),
                       dict(homogeneous=True),dict(chemical_delay=.1),dict(chemical_delay=.5),
                       dict(cue=.1),dict(cue=.3),dict(latency_scale=.75),dict(latency_scale=1.25)]:
            out.append(('robustness',replace(base,route=route,tau=.2,gain=1024,**change),1024,v.ALLOCS))
    for route in v.ROUTES:
        for k in [1,4,8]:
            for tau in [.012,.05,.2]:
                for gain in [0,64,256,1024]:
                    out.append(('grid',replace(base,k=k,tau=tau,gain=gain,route=route),1024,('top',)))
    return out


def main():
    freeze();cs=configs();save(OUT/'planned_conditions.json',[dict(group=g,config=asdict(c),n=n,allocations=names) for g,c,n,names in cs])
    q0,ys=v.sources();save(OUT/'source_provenance.json',dict(reference=q0,yields=ys,
        source='data/dark-basis-resolution/conditional_products_spikes.csv',condition='upcj2N__full',gamma_oxygen_s=0,
        raw_population_full_difference=ys['full']-ys['population'],
        full_population_any_downstream_difference_bound_pp=100*(-np.expm1(10000*np.log1p(-abs(ys['full']-ys['population'])))),
        original_solver_discrepancy=2.7e-10,native_gain=None,native_noise=None,native_target_ids=None))
    records=[];start=time.time();(OUT/'trials').mkdir(exist_ok=True)
    for ci,(group,c,n,names) in enumerate(cs):
        key=f'{ci:03d}_{group}';csv=OUT/f'{key}.csv'
        if csv.exists():records.extend(pd.read_csv(csv).to_dict('records'));continue
        block=[]
        for seed in SEEDS:
            result=v.simulate(c,n=n,seed=seed,names=names)
            for row in v.summarize(result):row.update(group=group,condition_id=key);block.append(row)
            np.savez_compressed(OUT/'trials'/f'{key}_{seed}.npz',times=result['times'],outcomes=result['outcomes'],
                soma_means=result['soma_means'],weights=result['weights'],delays_s=result['delays_s'],alpha=result['alpha'])
            if ci==0:
                save(OUT/'fixed_network.json',dict(weights=result['weights'].tolist(),delays_s=result['delays_s'].tolist(),
                    alpha=result['alpha'].tolist(),scores=result['scores'].tolist(),allocation_names=result['names']))
            if group=='main' and c.source=='full' and c.route=='trigger' and c.tau==.2:
                np.savez_compressed(OUT/f'chemical_path_{seed}.npz',path=result['path'])
                check=[]
                for control in ['replay','classical_stochastic']:
                    other=v.simulate(replace(c,source=control),n=n,seed=seed,names=names,
                        replay=result['path'] if control=='replay' else None)
                    check.append(dict(control=control,seed=seed,
                        identical_times=bool(np.array_equal(result['times'],other['times'])),
                        identical_products=bool(np.array_equal(result['path'],other['path']))))
                save(OUT/f'replay_check_{seed}.json',check)
        pd.DataFrame(block).to_csv(csv,index=False);records.extend(block)
        print(f'{ci+1}/{len(cs)} {key} {c.route} tau={c.tau} gain={c.gain} k={c.k} ({time.time()-start:.1f}s)',flush=True)
    pd.DataFrame(records).to_csv(OUT/'seed_summary.csv',index=False)
    print('done',len(records),flush=True)

if __name__=='__main__':main()
