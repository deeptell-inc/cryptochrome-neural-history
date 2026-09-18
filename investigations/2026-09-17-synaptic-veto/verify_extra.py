"""Independent finite-sum checks, measurement precision, and replay audit."""
from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
from scipy.stats import binom,norm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import synaptic_veto as v
OUT=ROOT/'data/synaptic-veto'


def main():
 c=v.Config(gain=1024,tau=.2);q,ys=v.sources();w,d,a,score=v.architecture(c)
 probs=v.probability(np.array([ys['full']]),c,a,q)
 expected=[]
 for ps in probs:
  law=np.array([1.])
  for p in ps:law=np.convolve(law,[1-p,p])
  expected.append(float(law[32:].sum()))
 # Alternative distribution: sum of two binomials, selected vs unselected.
 selected=probs[2,a[2]>0][0]
 law=np.convolve(binom.pmf(np.arange(c.k+1),c.k,selected),binom.pmf(np.arange(65-c.k),64-c.k,c.p0))
 error=float(abs(law[32:].sum()-expected[2]))
 assert error<1e-12
 assert max(expected[1:])-min(expected[1:])<1e-12
 # Weighted, signed N=8 law by full event enumeration vs independent Monte Carlo.
 ww=w[:8];ww/=np.linalg.norm(ww);ps=np.linspace(.2,.7,8);cut=.15
 exact=0.
 for bits in itertools.product((0,1),repeat=8):
  b=np.array(bits);exact+=((b-ps)@ww>=cut)*np.prod(np.where(b,ps,1-ps))
 rng=np.random.default_rng(987);b=rng.random((200000,8))<ps
 empirical=float((((b-ps)@ww)>=cut).mean());se=np.sqrt(exact*(1-exact)/200000)
 assert abs(empirical-exact)<5*se
 # Signal survival and all-or-none downstream total-variation bound.
 delta=abs(ys['full']-ys['population']);bound=100*(-np.expm1(c.pool*np.log1p(-delta)))
 # Measurement design: independent Bernoulli-equivalent observations, NOT cells.
 design=[]
 for target in [.5,1.,5.]:
  for p0 in [.1,.5,.9]:
   for comparisons in [1,3]:
    d=target/100;z=norm.ppf(1-.05/(2*comparisons))+norm.ppf(.8)
    n=int(np.ceil(2*z*z*p0*(1-p0)/d**2))
    for icc,m in [(0.,1),(.05,20),(.2,20)]:
     design.append(dict(target_pp=target,baseline_probability=p0,comparisons=comparisons,
        independent_equivalent_n_per_arm=n,trials_per_cell_assumed=m,ICC_assumed=icc,
        design_effect=1+(m-1)*icc,illustrative_total_trials_per_arm=int(np.ceil(n*(1+(m-1)*icc))),
        caveat='normal approximation; assumed ICC; no native cell number identified'))
 pd.DataFrame(design).to_csv(OUT/'measurement_precision.csv',index=False)
 info=dict(equal_vote_probabilities=dict(zip(v.ALLOCS,expected)),majority_convolution_error=error,
    weighted_exact=exact,weighted_MC=empirical,weighted_MC_SE=se,
    full_population_downstream_TV_bound_pp=bound,
    survival_12ms_at40ms=float(np.exp(-.04/.012)),survival_12ms_at110ms=float(np.exp(-.11/.012)))
 (OUT/'independent_checks.json').write_text(json.dumps(info,indent=2)+'\n')
 print(info)
if __name__=='__main__':main()
