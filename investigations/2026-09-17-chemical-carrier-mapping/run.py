from pathlib import Path
import sys,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import chemical_carrier_mapping as m
from impl.synaptic_veto import sources
OUT=ROOT/'data/chemical-carrier-mapping';OUT.mkdir(exist_ok=True)

def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def freeze():
    p=OUT/'preserved_inputs.json'
    if not p.exists():
        old=json.loads((ROOT/'data/context-audited-bridge/preserved_inputs.json').read_text())['records']
        old+=json.loads((ROOT/'audit/context-audited-bridge-manifest.json').read_text())
        old={x['path']:x for x in old}
        save('preserved_inputs.json',dict(records=list(old.values())))
    for r in json.loads(p.read_text())['records']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']

def audit():
    path='evidence/2026-09-17-native-noise-literature/PMC12043502.xml'
    tree=ET.parse(ROOT/path).getroot();ex=[]
    for i,p in enumerate(tree.iter('p')):
        if i in [2,8,23,25,27,28,29,35,39,41,42]:
            ex.append(dict(source='Rorsman2025',doi='10.1038/s41586-025-08734-4',paragraph=i,text=' '.join(''.join(p.itertext()).split())))
    src=json.loads((ROOT/'data/dark-birth/source_excerpts.json').read_text())
    for r in src:
        if r['id']=='muller2011':
            for q in r['excerpts']:ex.append(dict(source='Muller2011',doi=r['doi'],**q))
    save('source_excerpts.json',ex)
    paths=[path,'data/dark-birth/evidence/muller2011.txt','data/context-audited-bridge/fogle2015_browser_excerpts.json',
        'DARK_ATOM_ASSIGNMENT_RESULTS.md','DARK_TURNOVER_RESULTS.md','data/dark-basis-resolution/conditional_products_spikes.csv']
    save('sources.json',[dict(path=s,sha256=hashlib.sha256((ROOT/s).read_bytes()).hexdigest(),
        evidence_type='primary full text/excerpt' if s in paths[:3] else 'conditional model specification/output',
        license='Rorsman CC BY 4.0; other primary texts retained for local verification, redistribution not established' if s in paths[:3] else 'project computational asset') for s in paths])
    candidates=[
      dict(id='HK_BOUND_NADP',priority='primary candidate; target-native link unverified',
        chemical_state='oxidized NADP+ retained at Hk/Kvbeta active site; reference Hk:NADPH',
        compartment='cytoplasmic face of membrane Shaker-Hk complex; presynaptic terminal localization must be demonstrated',
        input='4-oxo-2-nonenal (4-ONE) or a separately identified reducible lipid carbonyl; not P/FADox itself',
        writing='Hk:NADPH + carbonyl + H+ -> Hk:NADP+ + reduced carbonyl',
        writing_product_caveat='exact reduced 4-ONE product/regioselectivity not identified from screened neuronal data',
        recovery='NADP+ dissociation -> apo Hk -> NADPH binding; NADP+ rebinding competes',
        count='number of functional cofactor sites in electrically relevant membrane patch, with covariance',
        release_link='A-current kinetics -> presynaptic AP/Ca -> release conditional on AP arrival',
        source_support='dFB 4-ONE catalytic rescue and voltage-dependent reversal; no dark-CRY-to-terminal calibration'),
      dict(id='LOCAL_4ONE',priority='upstream writing substrate and possible reservoir; not equated to cofactor memory',
        chemical_state='local free/reactive 4-ONE',compartment='membrane-adjacent reactive volume and lipid source',
        input='lipid peroxidation/fragmentation under oxidative chemistry, not directly H2O2 -> 4-ONE',
        writing='unresolved oxidant activation, PUFA oxidation and carbonyl formation',
        recovery='enzymatic metabolism, adduct formation, diffusion/export, Hk turnover; separately parameterized',
        count='local concentration times accessible volume; membrane partition adds a separately measured reservoir',
        release_link='through Hk or separately tested off-target actions',
        source_support='Rorsman confirms supplied 4-ONE current response; endogenous CRY production unmeasured'),
      dict(id='4HNE_CONTROL',priority='negative substrate comparator in specific dFB assay',
        chemical_state='4-hydroxy-2-nonenal (4-HNE)',compartment='200uM pipette experiment, local concentration unknown',
        input='not interchangeable with 4-ONE',writing='no supported Hk writing rate in target',
        recovery='not calibrated',count='not calibrated',release_link='not calibrated',
        source_support='no corresponding A-current inactivation change in reported dFB assay; not universal inertness'),
    ]
    save('candidate_registry.json',candidates)
    native={k:None for k in ['CRY_HQ_turnover_per_s','P_peroxide_delivery_fraction','E_early_peroxide_delivery_fraction',
      'E_late_peroxide_delivery_fraction','escaped_SQ_reoxidation_s','superoxide_dismutation_parameters',
      'peroxide_clearance_s','oxidant_to_4ONE_formation_law','local_4ONE_uM','4ONE_volume_um3','4ONE_clearance_s',
      'Hk_kcat_s','Hk_KM_uM','NADP_off_s_by_voltage','NADPH_on_uM_s','local_NADPH_uM','NADP_on_uM_s','local_NADP_uM',
      'membrane_patch_area_um2','Shaker_complex_density_per_um2','Hk_sites_per_complex','functional_site_fraction',
      'site_covariance','presynaptic_AP_transfer','calcium_transfer','release_logit_beta','release_apo_logit_beta','baseline_release_conditional_on_AP']}
    save('native_parameters.json',dict(target='dark CRY -> membrane-proximal carbonyl -> Hk -> terminal release',values=native,
      status='chemical candidate specified, native coefficients unidentified; no implicit defaults',
      source_only_constraint='dFB reduced/oxidized current kinetics and qualitative post-miniSOG persistence, not target-native kinetics'))
    fields=[
      ('source','CRY_HQ_turnover_per_s','completed reactions/s','converts branch probability to reaction flux','CRY-bound flavin state and calibrated turnover'),
      ('source','P/E delivery fractions','dimensionless','compartment arrival differs by branch and time','local oxygen products alongside CRY state'),
      ('precursor','oxidant_to_4ONE_formation_law','explicit flux law, not fixed scalar','oxidant activation, lipid substrate, fragmentation, competing sinks','species-resolved carbonyl and oxidant time courses'),
      ('write','Hk_kcat_s','1/s','a(C)=kcat*C/(KM+C)','local concentration series and absolute Hk cofactor occupancy'),
      ('write','Hk_KM_uM','uM','distinct from pipette dose','multiple local concentrations including saturation if accessible'),
      ('erase','NADP_off_s_by_voltage','1/s','O->apo plus free NADP+','washout and voltage-resolved cofactor tracking'),
      ('erase','NADPH_on_uM_s * local_NADPH_uM','1/s','apo->reduced Hk','binding/donor concentration dependence'),
      ('erase','NADP_on_uM_s * local_NADP_uM','1/s','apo->oxidized Hk rebinding','oxidized-cofactor competition'),
      ('resource','4ONE_clearance_s','1/s if first-order approximation','clearance distinct from Hk recovery','local carbonyl washout with and without Hk turnover'),
      ('count','local_C_uM * local_volume_um3','molecules via 602.214076','free reactive carbonyl, separate membrane partition pool','absolute amount and reaction-accessible volume'),
      ('count','area * complex_density * Hk_sites * functional_fraction','sites','membrane-bound storage capacity','target-terminal localization and quantitative complex counting'),
      ('noise','site_covariance','fraction squared','determines effective rather than actual site count','cofactor state covariance within relevant window'),
      ('readout','release_logit_beta / release_apo_logit_beta','log-odds per occupancy fraction','AP-arrival-conditioned release','occupancy, terminal AP/Ca, release outcomes with controls'),
    ]
    pd.DataFrame([dict(stage=a,parameter=b,unit=c,interpretation=d,required_observation=e,native_value=None,
        status='unidentified in target',source_context='Rorsman2025 dFB for Hk mechanism; conditional dark CRY upstream; no native numeric transfer')
        for a,b,c,d,e in fields]).to_csv(OUT/'coefficient_mapping.csv',index=False)

def calculations():
    q0,ys=sources();q1=ys['full'];rows=[]
    for ke in [.1,1.,10.]:
      for clear in [0.,1.,5.]:
       for name,d in [('equal_access',(1,1,1)),('escape_lost',(1,0,0)),('direct_lost',(0,1,1))]:
        for t in [0,.012,.2,2.,20.,200.]:
            kw=dict(direct_delivery=d[0],early_escape_delivery=d[1],late_escape_delivery=d[2])
            a=m.peroxide_path(q0,t,ke,clear,**kw);b=m.peroxide_path(q1,t,ke,clear,**kw)
            rows.append(dict(escape_oxidation_s=ke,clearance_s=clear,delivery_scenario=name,time_s=t,
                available_peroxide_reference_per_cycle=a[1],available_peroxide_full_per_cycle=b[1],
                delta_available_per_cycle=b[1]-a[1],delta_cumulative_formed_per_cycle=b[2]-a[2],
                eventual_delta_delivered_per_cycle=(q1-q0)*(d[0]-.5*(d[1]+d[2])),
                status='theoretical branch mapping; instant dismutation; no native rates'))
    pd.DataFrame(rows).to_csv(OUT/'branch_to_peroxide.csv',index=False)
    rows=[]
    for a in [.1,10,100]:
      for off in [.01,1,100]:
       for pulse in [.012,.2]:
        p=m.cofactor_step([1,0,0],pulse,write_s=a,off_s=off,bind_reduced_s=100,bind_oxidized_s=0)
        late=m.cofactor_step(p,2,write_s=0,off_s=off,bind_reduced_s=100,bind_oxidized_s=0)
        rows.append(dict(write_s=a,NADP_off_s=off,NADPH_binding_s=100,pulse_s=pulse,
          oxidized_at_pulse_end=p[1],oxidized_2s_after=late[1],apo_2s_after=late[2],
          recovery_MFPT_s=m.recovery_mfpt(off,100,0),status='maintained-carbonyl pulse scenario; not finite dose'))
    pd.DataFrame(rows).to_csv(OUT/'separate_write_recovery.csv',index=False)
    rows=[]
    for volume in [.001,.01,.1]:
      for conc in [.01,.1,1.,10.,50.]:
        number=m.molecule_count(conc,volume)
        rows.append(dict(volume_um3=volume,concentration_uM=conc,mean_mobile_carbonyl_molecules=number,
          maximum_expected_new_fraction_200_sites=min(1,number/200),
          poisson_CV_if_assumed=1/np.sqrt(number),status='unit conversion/scenario; no native volume or concentration'))
    pd.DataFrame(rows).to_csv(OUT/'compartment_counts.csv',index=False)
    rows=[]
    for sites in [200,400,10000]:
      for rho in [0,.001,.01,.1]:
        rows.append(dict(actual_sites=sites,equal_pair_correlation=rho,noise_equivalent_independent_sites=m.effective_sites(sites,rho),
            channel_complexes_if_four_functional_sites=sites/4,
            equivalent_uM_in_point01_um3=sites/m.molecule_count(1,.01),
            status='equivalent volumetric density is NOT soluble cofactor concentration'))
    pd.DataFrame(rows).to_csv(OUT/'site_counts_and_covariance.csv',index=False)
    rows=[]
    for beta in [10.,100.,128/(.45*.55)]:
      for dp in [.005,.01,.05]:
        dz=m.required_occupancy_delta(dp,p0=.45,beta=beta)
        rows.append(dict(p0=.45,beta_per_fraction=beta,local_dp_dz=.45*.55*beta,desired_release_delta=dp,
           required_delta_oxidized_fraction=dz,expected_extra_oxidized_200_sites=200*dz,
           status='release conditional on AP; assumed beta; old g mapping requires z=h'))
    pd.DataFrame(rows).to_csv(OUT/'required_release_gain.csv',index=False)
    save('reaction_contract.json',dict(
      fixed_model_species={'HQ':'CRY:FADH-','E':'CRY:FADH radical plus escaped O2 radical anion','P':'CRY:FADox plus peroxide-forming branch'},
      P_overall='FADH- + O2 + H+ -> FADox + H2O2',
      E_recovery='FADH radical + O2 -> FADox + O2 radical anion + H+',
      E_initial='FADH- + O2 -> FADH radical + O2 radical anion',
      dismutation='2 O2 radical anions + 2 H+ -> H2O2 + O2',
      caveat='Formal flavin redox-site notation omits phosphate charges. Full E oxidation, all superoxide to peroxide, no alternate scavenging in equal-yield case.',
      forbidden_identifications=['P=4-ONE','P=all ROS','H2O2 directly oxidizes bound NADPH with known gain',
         'off-rate=on-rate','cofactor sites=CRY molecules','synapse counts=cofactor counts','somatic firing gain=release gain'],
      unknown_upstream_chemistry='peroxide/other oxidants -> activated lipid oxidation -> 4-ONE requires catalysts, PUFA substrate, transport and competing sinks',
      chain_gain='dp/dq = p(1-p)*beta * dz/dC4ONE * dC4ONE/dJoxidant * dJoxidant/dq; dynamic convolution, not a constant generally'))

def finite_pool():
    rows=[];out=OUT/'trials';out.mkdir(exist_ok=True)
    for carbonyls in [6,20,200]:
      for off in [.01,1.]:
       for clear in [0.,.5,10.]:
        for seed in [260917201,260917202,260917203]:
          x=m.finite_carbonyl_trials(sites=200,carbonyls=carbonyls,volume_um3=.01,k2_uM_s=.1,
            off_s=off,bind_reduced_s=100,clearance_s=clear,n=1024,seed=seed,times_s=[.2,2.2])
          np.savez_compressed(out/f'C{carbonyls}_off{off}_clear{clear}_{seed}.npz',counts=x)
          for i,t in enumerate([.2,2.2]):
            z=x[i,:,2]/200
            rows.append(dict(initial_carbonyls=carbonyls,off_s=off,clearance_s=clear,seed=seed,n=1024,time_s=t,
                oxidized_fraction_mean=z.mean(),oxidized_fraction_variance=z.var(ddof=1),
                carbonyl_converted_mean=x[i,:,4].mean(),carbonyl_cleared_mean=x[i,:,5].mean(),
                apo_fraction_mean=x[i,:,3].mean(),NADPH_supplied_mean=x[i,:,7].mean(),
                maximum_oxidized_fraction=z.max()))
    df=pd.DataFrame(rows);df.to_csv(OUT/'finite_pool_seed_summary.csv',index=False)
    combined=[]
    for key,g in df.groupby(['initial_carbonyls','off_s','clearance_s','time_s']):
        mu=g.oxidized_fraction_mean.mean();N=int(g.n.sum())
        var=(((g.n-1)*g.oxidized_fraction_variance).sum()+(g.n*(g.oxidized_fraction_mean-mu)**2).sum())/(N-1)
        combined.append(dict(zip(['initial_carbonyls','off_s','clearance_s','time_s'],key),n=N,
            oxidized_fraction_mean=mu,MC_SE=np.sqrt(var/N),between_seed_SD=g.oxidized_fraction_mean.std(ddof=1),
            fraction_of_carbonyl_converted=g.carbonyl_converted_mean.mean()/key[0],
            status='independent finite-bolus chemical scenarios, not full-reset or native effects'))
    pd.DataFrame(combined).to_csv(OUT/'finite_pool_summary.csv',index=False)

def independent_check():
    # Full small finite-resource CTMC, independently assembled.
    sites=3;nc=2;states=[(c,sites-o-u,o,u) for c in range(nc+1) for o in range(sites+1) for u in range(sites-o+1)]
    ix={x:i for i,x in enumerate(states)};Q=np.zeros((len(states),len(states)))
    for j,(c,r,o,u) in enumerate(states):
        events=[((c-1,r-1,o+1,u),5*c*r/6.02214076),((c,r,o-1,u+1),o),
            ((c,r+1,o,u-1),10*u),((c-1,r,o,u),.5*c)]
        for target,rate in events:
            if rate>0:Q[ix[target],j]+=rate;Q[j,j]-=rate
    p0=np.zeros(len(states));p0[ix[(2,3,0,0)]]=1
    prob=expm(Q)@p0;z=np.array([s[2] for s in states],float)
    mean=float(z@prob);var=float(z*z@prob-mean**2)
    x=m.finite_carbonyl_trials(sites=3,carbonyls=2,volume_um3=.01,k2_uM_s=5.,off_s=1,bind_reduced_s=10,
        clearance_s=.5,n=20000,seed=264911,times_s=[1.])
    actual=float(x[0,:,2].mean());se=np.sqrt(var/20000)
    assert abs(actual-mean)<5*se
    # Source contrast solution and three-state mean verified via independent ODE.
    q0,ys=sources();maxerr=0
    for q in [q0,ys['full']]:
        for t in [.2,2.]:
            y0=[.5*(1-q),.5*(1+q),.5*(1+q)]
            sol=solve_ivp(lambda t,y:[-y[0],y[0]-5*y[1],y[0]],(0,t),y0,rtol=1e-11,atol=1e-13)
            pred=m.peroxide_path(q,t,1,5,direct_delivery=1,early_escape_delivery=1,late_escape_delivery=1)
            maxerr=max(maxerr,float(np.max(abs(pred-sol.y[:,-1]))))
    save('checks.json',dict(finite_pool_CTMCMaster_mean=mean,Gillespie_mean=actual,MC_SE=se,within_5SE=True,
        independent_peroxide_ODE_error=maxerr,site_and_carbonyl_and_cofactor_flux_balances=True,
        native_parameters_identified=False,physical_experiments_performed=0))
    print('Independent CTMC/SSA',mean,actual,se,'ODE error',maxerr,flush=True)

def stochastic_release():
    rows=[]
    for carbonyls in [6,20,200]:
      for off in [.01,1.]:
       for clear in [0.,.5,10.]:
        x=np.concatenate([np.load(OUT/'trials'/f'C{carbonyls}_off{off}_clear{clear}_{seed}.npz')['counts']
            for seed in [260917201,260917202,260917203]],axis=1)
        for i,t in enumerate([.2,2.2]):
         for beta in [10.,100.,128/(.45*.55)]:
            states=x[i,:,1:4]/200
            p=m.cofactor_release(states,[1,0,0],p0=.45,beta_oxidized=beta,beta_apo=0)
            pmean=m.cofactor_release(states.mean(0),[1,0,0],p0=.45,beta_oxidized=beta,beta_apo=0)
            rows.append(dict(initial_carbonyls=carbonyls,off_s=off,clearance_s=clear,time_s=t,beta_oxidized=beta,beta_apo=0,
                delta_release_pp=100*(p.mean()-.45),chemical_MC_SE_pp=100*p.std(ddof=1)/np.sqrt(len(p)),
                mean_only_delta_release_pp=100*(pmean-.45),chemical_probability_variance=p.var(),
                conditional_Bernoulli_variance=np.mean(p*(1-p)),total_Bernoulli_variance=p.mean()*(1-p.mean()),
                status='added-carbonyl versus no-carbonyl ideal reference, not nuclear-history contrast; gain/sign assumed'))
            assert abs(p.var()+np.mean(p*(1-p))-p.mean()*(1-p.mean()))<1e-14
    pd.DataFrame(rows).to_csv(OUT/'finite_pool_release.csv',index=False)

if __name__=='__main__':
    freeze();audit();calculations();finite_pool();independent_check();stochastic_release()
    print('chemical mapping complete',flush=True)
