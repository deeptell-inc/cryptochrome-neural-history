from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from impl import population_memory as m
from impl import dark_threshold_source as s
p=m.OUT/'latest_case.npz'
if not p.exists():np.savez_compressed(p,**m.build_case())
data=dict(np.load(p));geo=m.population_geometry(data['H_H'],data['D_H']);M,z=m.cycle(data)
rows,states=m.compare_cycle(M,data['P_0'],geo)
print('baseline',rows,flush=True)
print('modes',m.mode_table(geo,data['P_0'],states['full']),flush=True)
for st in ('R','H'):
 g=m.population_geometry(data['H_'+st],data['D_'+st]);print(st,'rates',g['rates'],'norm D',np.linalg.norm(data['D_'+st]),'PC',np.linalg.norm(g['C'].conj().T@data['D_'+st]@g['Q']),'gap',np.min(np.diff(g['energies'])),flush=True)
print('beforeH_coh',np.linalg.norm((np.eye(81)-geo['P'])@z['before_H']@s.X0),flush=True)
