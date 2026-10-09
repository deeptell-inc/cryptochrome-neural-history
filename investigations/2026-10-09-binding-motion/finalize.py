from pathlib import Path
import sys,json,hashlib,platform
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
D=ROOT/'data/binding-motion'
records=[json.loads(p.read_text()) for p in sorted(D.glob('L*_angle*.json'))]
assert len(records)==73,len(records)
aug={}
for p in D.glob('augment_angle*.json'):aug.update(json.loads(p.read_text()))
keys=lambda d:(d['rms_deg'],d['tau_w_s'],d['fraction'],d['koff_s'],d['bias'])
best={};checks={};convergence=[]
for d in records:
 assert np.isfinite(d['delta_yield']) and all(np.isfinite(list(d['yields'].values())))
 assert d['trace_error']<1e-8 and d['hermiticity_error']<1e-8 and d['minimum_density_eigenvalue']>-1e-9,d['label']
 if d['label'] in aug:d.update(aug[d['label']])
 k=keys(d)
 if k not in best or d['cutoff']>best[k]['cutoff']:best[k]=d
 if d['cutoff']==3:
  l2=next(x for x in records if keys(x)==k and x['cutoff']==2)
  err=abs(d['delta_yield']-l2['delta_yield']);qerr=max(abs(d['yields'][a]-l2['yields'][a]) for a in d['yields'])
  assert err<1e-6 and qerr<1e-5 and err/max(abs(d['delta_yield']),1e-10)<1e-3,(d['label'],err,qerr)
  convergence.append(dict(label=d['label'],delta_gap=err,yield_gap=qerr))
assert len(best)==58
rows=[]
for d in best.values():
 rows.append(dict(label=d['label'],angle_deg=d['rms_deg'],tau_w_ns=d['tau_w_s']*1e9,
  target_bound_fraction=d['fraction'],koff_s=d['koff_s'],mean_bound_dwell_s=1/d['koff_s'] if d['koff_s'] else np.inf,
  bias=d['bias'],cutoff=d['cutoff'],q_full=d['yields']['full'],q_population=d['yields']['population'],q_reset=d['yields']['reset'],
  delta_q=d['delta_yield'],full_population_gap=d['full_population_difference'],
  spin_contribution=d.get('instantaneous_spin_contribution'),motion_contribution=d.get('motion_distribution_contribution'),
  actual_bound_fraction_full=d.get('observed_bound_fractions',[np.nan]*3)[0]))
frame=pd.DataFrame(rows).sort_values(['tau_w_ns','angle_deg','target_bound_fraction','koff_s','bias']);frame.to_csv(D/'molecular_summary.csv',index=False)
get=lambda angle,f,off,bias=0.,tau=1e-11:best[(float(angle),tau,float(f),float(off),float(bias))]
# Pure-mobile control must recover the old matched result regardless of unused bound wobble.
old=json.loads((ROOT/'data/hq-rotation/L2_tau21.476596_B50.json').read_text())
free_errors=[]
for angle in (0.,1.,5.):
 d=get(angle,0,0);err=abs(d['yields']['full']-old['yields']['full']);assert err<1e-10
 free_errors.append(err);assert abs(d['delta_yield'])<1e-10
checks.update(unique_conditions=len(best),molecular_runs=len(records),convergence=convergence,
 max_delta_convergence=max(x['delta_gap'] for x in convergence),max_yield_convergence=max(x['yield_gap'] for x in convergence),
 free_control_errors=free_errors,max_trace_error=max(d['trace_error'] for d in records),
 max_hermiticity_error=max(d['hermiticity_error'] for d in records),minimum_density_eigenvalue=min(d['minimum_density_eigenvalue'] for d in records),
 max_full_population_gap=max(abs(d['full_population_difference']) for d in records))
quad=json.loads((D/'quadrature-check/checks.json').read_text());assert len(quad)==3
assert max(d['delta_difference_from_55_nodes'] for d in quad)<1e-7
checks['quadrature_55_to_171_nodes']=[dict(label=d['label'],delta_gap=d['delta_difference_from_55_nodes'],yield_gap=d['yield_difference_from_55_nodes']) for d in quad]
checks['molecular_runs_including_extra_quadrature']=len(records)+len(quad)
checks['independent_HQ']=json.loads((D/'independent_checks.json').read_text())
checks['independent_adjoint']=aug['independent_adjoint']
checks['independent_static_cycle']=aug['independent_static_cycle']
assert checks['independent_static_cycle']['difference']<1e-8
neural=pd.read_csv(D/'neural/summary.csv');small=neural.query('integration == "temporal"').copy();assert len(small)==18
small.to_csv(D/'neural_selected.csv',index=False)
checks['paired_neural_trials_per_readout']=int(small.paired_trials.sum());checks['status']='passed'
(D/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained',sharey=True)
for ax,angle in zip(axes,(0,1,5)):
 values=np.array([[get(angle,f,off)['delta_yield'] for off in (.01,1,100)] for f in (.01,.1,.5,.9)])
 im=ax.imshow(np.log10(np.maximum(values,1e-12)),vmin=-8,vmax=-2,aspect='auto',cmap='viridis',origin='lower')
 ax.set_xticks(range(3),['100','1','0.01']);ax.set_yticks(range(4),['0.01','0.10','0.50','0.90']);ax.set_xlabel('Mean bound dwell time (s)');ax.set_title(f'Bound RMS wobble: {angle} deg')
 for i in range(4):
  for j in range(3):ax.text(j,i,f'{values[i,j]:.5e}',ha='center',va='center',fontsize=8,color='white' if np.log10(values[i,j])<-5 else 'black')
axes[0].set_ylabel('Target bound fraction');fig.colorbar(im,ax=axes,label='log10 full minus reset yield')
fig.suptitle('Reversible binding; random anchors; wobble correlation 0.01 ns')
fig.savefig(D/'binding_motion_cases.png',dpi=190);fig.savefig(D/'binding_motion_cases.pdf');plt.close(fig)
selected=[(0,.5,1),(1,.5,1),(5,.5,1),(1,.5,100),(1,.01,1),(0,0,0)]
fig,ax=plt.subplots(figsize=(8,4),layout='constrained');labs=[]
for i,(a,f,k) in enumerate(selected):
 label=f'angle{a:g}_tau1e-11_f{f:g}_off{k:g}_bias0';r=small[small.label==label].iloc[0]
 ax.errorbar(r.delta_stop_pp,i,xerr=[[r.delta_stop_pp-r.low_pp],[r.high_pp-r.delta_stop_pp]],fmt='o',capsize=3)
 labs.append(f'{a:g} deg; f={f:g}; dwell={1/k:g} s' if k else 'Free control')
ax.set_yticks(range(len(labs)),labs);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Stopping probability difference (percentage points)');ax.set_title('Assumed gain 1024, pool 10000, chemical lifetime 200 ms\nConservative 95% Monte Carlo intervals')
fig.savefig(D/'conditional_neural_cases.png',dpi=190);fig.savefig(D/'conditional_neural_cases.pdf');plt.close(fig)

lines=['---','asset: binding-motion-results','version: 1.0.0','生成元入力: frozen HQ tensors; matched rotational cycle; user case-study request','依存工程: P4 / S15 / S5 / S6 / S16 / S17','生成日: 2026-10-09','---','','# 可逆的結合・拘束下揺動・配向分布を含む再計算','',
'**この模型では、結合割合だけでなく、結合が続く時間と結合中の角度揺動が核履歴の残存を決める。ランダムな配向でも履歴差は残り得るが、短い結合や大きな揺動では小さくなる。**',
'',f'異なる分子条件58件、角度次数を上げた15件、配向積分を細かくした3件、計76実行。神経回路は18条件×24576ペア＝{checks["paired_neural_trials_per_readout"]}ペア（各読出し方式で同じ試行を利用）。すべて条件付き模型計算であり、脳内の結合率・揺動・停止率の測定ではない。',
'','## 1. 計算した条件','',
'- 可動成分：既存の有限磁場・共通剛体回転モデル。rank-2回転相関時間21.47660 ns。',
'- 結合成分：固定された結合先の周囲で、3軸の小さな角度揺動。全回転ベクトルのRMS角0 / 0.10000 / 1.00000 / 5.00000°。円錐半角ではない。',
'- 主走査の揺動相関時間0.01000 ns。1°について0.10000 nsも比較。',
'- 運動単独の目標結合割合f=0.01000 / 0.10000 / 0.50000 / 0.90000。解離率0.01000 / 1.00000 / 100.00000 s⁻¹（平均結合時間100.00000 / 1.00000 / 0.01000 s）。',
'- 主走査は0、1、5°の各12条件＋配向と対照。0.1°と相関時間変更はf=0.5の5条件に限定。',
'- 結合・解離はHQ/R/E待機だけでなくラジカル対反応中にも作用させ、分子のスピン状態を引き継いだ。',
'- 磁場50 µT、現行HQ/R/Eテンソル、反応・回復・還元率、局所MD散逸を維持。20完了サイクル後の次反応収率を計算。追加酸素電子緩和・交換・双極子を0とする旧仮定も維持。',
'',
'これは20サイクルの反復準備後の比較であり、入力を止めた後の受動的な秒単位保持時間を新たに測った計算ではない。結合率konは擬一次率であり、結合先の濃度や飽和は明示的に解いていない。',
'',
'拘束下揺動は、dθₐ=−θₐdt/τw+√(2v/τw)dWₐ、v=θrms²/3 という線形化した拘束拡散の微小角・高速揺動近似である。有限の角度拡散係数v/τwと復元項を持ち、角度相関はv exp(−|t|/τw)となる。コードはこの極限の有効生成子を解き、OU軌道を直接生成してはいない。Vₐ=−i[Jₐ,Hbody] とすると追加散逸は 2vτw ΣₐD[Vₐ]。平均Hamiltonianと局所散逸を6方向の正の重みで回転平均する。自由回転と同じ無制限の回転拡散係数を結合成分へ与える方法ではない。遅い揺動・大角度拘束の厳密なSO(3)拡散解ではなく、この範囲の予備的な物理モデルである。',
'',
'fullは密度行列を保持、populationはHQ境界で平均HQ Hamiltonianの固有基底に射影、resetは核状態をI/9へ置換する。resetも結合状態と配向の周辺分布は保持する。full−resetには、その過去の反応差から生じた運動分布差も入り得るため、以下では別途分解した。',
'','## 2. 結合割合50%で、滞在時間と揺動を変える','',
'目標f=0.50000、結合先の配向はランダム、τw=0.01000 ns。Δqはfull−resetの反応収率差で、神経停止率ではない。',
'','| RMS揺動角 ° | 平均結合100 s：Δq | 1 s：Δq | 0.01 s：Δq |','|---:|---:|---:|---:|']
for a in (0,.1,1,5):lines.append('| '+f'{a:.5f}'+' | '+' | '.join(f"{get(a,.5,k)['delta_yield']:.5e}" for k in (.01,1,100))+' |')
lines += ['',
'同じ結合割合でも、解離と再結合を速めると履歴差は大きく減る。今回の可動成分では自由回転中に速い緩和が起こり、結合し直しても以前の核状態が自動的に復元されるわけではない。ただし新しい反応サイクルによる再準備は起こるため、完全消失と断定しない。',
'',
'揺動に対する変化は必ずしも単調ではない。1 sの結合では0.1°の弱い揺動が0°から収率差をわずかに増やす。角度揺動は記憶の緩和だけでなく、準備・反応・読出しも変える。元の固定基準も、収率差の数学的上限ではない。',
'',f"非交換のランダム固定／自由混合（f=0.5、揺動0°）ではΔq={get(0,.5,0)['delta_yield']:.5e}。以前の『全固定分子を元と同じ向きに置く』混合と異なり、配向平均を実際に計算した。自由成分だけでは差は10⁻¹⁰の報告分解能で未解像だった。",'',
'## 3. 結合割合を増やせば救えるか','',
'揺動1°、τw=0.01000 ns、ランダムな結合先。','',
'| 目標結合割合 | 平均結合1 s：Δq | 平均結合0.01 s：Δq |','|---:|---:|---:|']
for f in (.01,.1,.5,.9):lines.append(f"| {f:.5f} | {get(1,f,1)['delta_yield']:.5e} | {get(1,f,100)['delta_yield']:.5e} |")
lines += ['',
'今回のf≤0.9の範囲では、割合が大きくても結合が短ければ大きな履歴差を維持できない。fと解離率を独立に扱う必要がある。fは運動単独の平衡割合であり、化学サイクルの境界で観測される割合とは完全には一致しない。',
'','## 4. 配向分布と揺動相関時間','',
'捕捉率を kon(n)=kon[1+aP₂(nz)] とした。a=−0.9は横向きに偏る分布、0はランダム、1.8は軸方向に偏る分布。構造的な向きの偏りを仮定したもので、磁場が脳内分子を整列させるという主張ではない。',
'','| 分布 | 揺動0°：Δq | 揺動1°：Δq | 揺動5°：Δq |','|---|---:|---:|---:|']
for b,name in [(-.9,'横方向に偏る'),(0,'ランダム'),(1.8,'軸方向に偏る')]:lines.append('| '+name+' | '+' | '.join(f"{get(a,.5,1,b)['delta_yield']:.5e}" for a in (0,1,5))+' |')
lines += ['',
'今回の分布では配向平均後にも正の差が残った。これは任意の配向・磁場・テンソルへ一般化できない。',
'',f"f=0.5、平均結合1 s、揺動1°でτwを0.01000 nsから0.10000 nsへ増やすと、Δqは{get(1,.5,1)['delta_yield']:.5e}から{get(1,.5,1,tau=1e-10)['delta_yield']:.5e}へ変わった。この高速揺動領域では、振幅だけでなく相関時間も緩和を左右する。",'',
'## 5. 核状態の差と運動分布の差','',
'fullの最終状態に対して、結合・配向分布を保ったまま核状態だけをresetする追加読出しを行った。これにより「その時点の核状態による寄与」と「過去の反応によって変わった運動分布の寄与」を加法的に分離した。',
'','| 1 s結合、f=0.5 | 核状態寄与 | 運動分布寄与 | 合計Δq |','|---|---:|---:|---:|']
for a in (0,1,5):
 d=get(a,.5,1);lines.append(f"| 揺動{a:.5f}° | {d['instantaneous_spin_contribution']:.5e} | {d['motion_distribution_contribution']:.5e} | {d['delta_yield']:.5e} |")
lines += ['',
'運動分布の寄与はここでは小さいが、純粋な核状態差と同じものとして扱わない。',
'',f"全条件のfull−population差の最大絶対値は{checks['max_full_population_gap']:.5e}。今回の遅い化学待機と高速微小揺動の範囲では、核populationを保持した比較でもfullに近い結果だった。前回の極端な高速化ケースまで含めた一般則ではない。",'',
'## 6. 既存停止回路へ入力した条件付き結果','',
'ゲイン1024、分子プール10000、top8、stop経路、化学寿命200 ms、時間積分読出し。24576ペア／条件、3乱数種。区間はペア差に対する保守的95% Monte Carlo区間で、生物学的な係数不確実性や分子近似の誤差を含まない。',
'','| 条件 | 停止確率差 pp | 95%区間 pp |','|---|---:|---:|']
for a,f,k in selected:
 r=small[small.label==f'angle{a:g}_tau1e-11_f{f:g}_off{k:g}_bias0'].iloc[0]
 name=f'揺動{a:.5f}°、f={f:.5f}、結合{1/k:.5f} s' if k else '自由回転のみ'
 lines.append(f'| {name} | {r.delta_stop_pp:+.5f} | [{r.low_pp:+.5f}, {r.high_pp:+.5f}] |')
lines += ['',
'これらは実神経・実行動の停止効果の予測値ではない。分子から化学状態への書込みと神経利得は旧模型の仮定であり、nativeな接続・部位・分子数は未同定。ゼロを含む区間を無効果の証明として解釈しない。区間が重なる小さな差を点推定だけで順位づけしない。小数点以下5桁は表示規則であり、生物学的な推定精度を意味しない。',
'','## 7. 検証と近似の限界','',
f'- 角度次数2→3の15比較：収率差の最大変化{checks["max_delta_convergence"]:.5e}、個々の収率の最大変化{checks["max_yield_convergence"]:.5e}。全58条件を高次数で再実行したわけではない。',
f'- 最大trace誤差{checks["max_trace_error"]:.5e}、最大Hermiticity誤差{checks["max_hermiticity_error"]:.5e}。評価した角度節点で密度行列の正値性を確認。連続角度の厳密な正値性証明とは異なる。',
f'- 強い配向偏り等の3条件で、角度次数3のまま配向積分を55→171節点へ増加。Δqの最大変化{max(d["delta_difference_from_55_nodes"] for d in quad):.5e}。主表・神経入力には同じ標準計算を用い、この追加差は精度診断として分離した。',
'- 核traceを明示的に除いた独立な結合HQの直接線形解と、生成計算のSchur反復解を比較。反応収率は独立な随伴GMRES読出し、非交換ランダム配向対照は局所核サイクル行列の20乗でも照合。',
'- 揺動近似は、同じ分散・相関時間を持つ8状態の有限相関時間テレグラフ過程とHQ待機写像を比較。0.01000 nsでは相対Frobenius差が0.1°で約1.12115e-08、1°で7.16336e-05、5°で5.98643e-03。これは別の有限状態過程との比較であり、全サイクルやOU過程に対する厳密誤差上限ではない。',
'- Hamiltonianの最大帯域×τwは主走査0.00370、追加相関時間0.03701。反応・交換より十分短い揺動を仮定した。遅い揺動への外挿は不可。',
'- 局所MD雑音と追加揺動は独立と仮定。重複して同じ分子運動を数えていないかはnativeスペクトル同定で確認が必要。',
'- 分子の結合割合、結合寿命、拘束角、相関時間、HQ占有率のいずれも脳内実測値として採用していない。哺乳類CRYへの移植にはFAD/HQ状態と書込み経路の同定も必要。',
'','## 8. 主張の確度と論文への含意','',
'| 主張 | ラベル | 根拠・適用範囲 |','|---|---|---|',
'| 同じ結合割合でも結合寿命によって履歴差が大きく変わる | Strongly Supported（この近似模型内） | 3寿命×4割合、独立待機解、角度収束 |',
'| ランダムな結合先配向でも履歴差が残る場合がある | Strongly Supported（この近似模型内） | 配向積分と独立非交換サイクル解 |',
'| 結合していれば元の大きな履歴差が必ず維持される | Contradicted（この模型での反例） | 5°揺動・短い結合のケース |',
'| 揺動の効果は常に単調な抑制である | Contradicted（この模型での反例） | 0→0.1°で僅かに増える条件 |',
'| この揺動模型が脳内CRYの実際の運動を表す | Not Found in Uploaded Sources | 運動パラメータ・束縛位置・native揺動スペクトル未同定 |',
'| 表の停止率差がヒト脳・自由否定に生じる | Not Found in Uploaded Sources | 哺乳類への分子対応・化学書込み・神経利得未較正 |',
'',
'論文では、**結合寿命と拘束下揺動を含む条件付き保持・読出し領域**を示す模型計算として位置づけ、脳内で秒単位の核履歴が存在すると結論しない。現在の計算は、その領域が空ではないことと、結合割合だけでは判定できないことを示す。どの条件が実脳に近いかは、今回の計算だけでは選べない。',
'','## 9. 再現と成果物','',
'- 実装：`impl/hq_binding.py`。仕様と再現：`investigations/2026-10-09-binding-motion/`。',
'- 全分子表：`data/binding-motion/molecular_summary.csv`。個別JSON/NPZは全精度を保持。',
'- 独立検証：`independent_checks.json`、`augment_angle0.json`、`augment_angle1.json`、`checks.json`。',
'- 神経結果：`neural_selected.csv`、`neural/summary.csv`、保存試行。元のNumPy 2.5.0環境で実行。',
'- テスト4件。図はPNG/PDFの両方を保存。原稿・投稿PDF・GitHub配布物は本工程では更新していない。',
'',
'### S16/S17監査記録','',
'固定割合と平衡割合、RMS角と円錐角、拘束下揺動と自由拡散、核状態差と運動分布差、数値誤差と近似誤差を区別した。単調性・固定基準を上限とする見方を反例で検査した。主要数値は保存出力・角度収束・独立待機／静的サイクル／随伴読出しと照合し、全条件の完全独立再計算をしたとは表現しない。生体妥当性の主張は採用しない。',
'',f'![結合と揺動の比較](<{D}/binding_motion_cases.png>)',f'![条件付き神経結果](<{D}/conditional_neural_cases.png>)','']
(ROOT/'BINDING_MOTION_RESULTS.md').write_text('\n'.join(lines))
files=[ROOT/'impl/hq_binding.py',ROOT/'data/population-memory/latest_case.npz',ROOT/'impl/hq_rotation.py',ROOT/'impl/synaptic_veto.py']+list((ROOT/'investigations/2026-10-09-binding-motion').glob('*.py'))
manifest=dict(date='2026-10-09',python=sys.version,numpy=np.__version__,model='mobile continuous rotation + reversible capture + small-angle fast bound wobble',
 frozen_inputs_and_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},molecular_conditions=58,molecular_runs=76,neural_conditions=18,neural_trials_per_readout=checks['paired_neural_trials_per_readout'],physiological_calibration=False)
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(checks,indent=2))
