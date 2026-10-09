from pathlib import Path
import sys,json
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl.literature_bridge_recalculation import solve_pulse,peak
D=ROOT/'data/rotation-cases';rows=[]
with threadpool_limits(limits=1):
 for tau in [.012,.05,.2]:
  base=solve_pulse(10.,.01,100.,tau);pert=solve_pulse(10.1,.01,100.,tau)
  check=solve_pulse(10.,.01,100.,tau,method='Radau')
  times=np.r_[0,np.geomspace(1e-5,10,600)];error=float(np.max(abs(base.sol(times)-check.sol(times))))
  assert error<1e-8
  t,y=peak(base)
  row=dict(tau_s=tau,peak_time_s=t,peak_oxidized_fraction=y,oxidized_at_200ms=float(base.sol(.2)[1]),one_percent_input_delta_at_200ms_pp=float(100*(pert.sol(.2)[1]-base.sol(.2)[1])),independent_ode_error=error,CRY_calibrated=False)
  rows.append(row);print(json.dumps(row),flush=True)
 (D/'independent_chemical_storage.json').write_text(json.dumps(rows,indent=2)+'\n')
