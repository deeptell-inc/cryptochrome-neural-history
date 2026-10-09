from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/membrane-field'
kappas=[0.,.10910941626120247,1.0910941626120247,10.910941626120245]
def read(kind,k,angle=1,off=1):
 L=3 if k==kappas[-1] else 2
 suffix='' if kind=='free' else f'_angle{angle:g}_off{off:g}'
 return json.loads((D/f'{kind}_L{L}_k{k:.8g}_axis2{suffix}.json').read_text())
spc=json.loads((D/'HQ_spectrum.json').read_text());checks=json.loads((D/'checks.json').read_text());limit=json.loads((D/'axial_limit.json').read_text())
line=[]
for k in kappas:
 c=0 if not k else 1/np.tanh(k)-1/k;p2=0 if not k else 1-3*c/k
 rr=read('free',k);tt=next(r['inverse_slowest_us'] for r in spc if r['axis']==2 and r['kappa']==k and r['cutoff']==4)
 line.append(f"| {k:.5f} | {c:.5f} | {p2:.5f} | {rr['yields']['reset']:.5f} | < 1.00000e-10 | {tt:.5f} |")
bound=[]
for k in kappas:
 a,b=read('bound',k,1),read('bound',k,5)
 bound.append(f"| {k:.5f} | {a['delta_yield']:.5e} | {b['delta_yield']:.5e} |")
a=read('bound',0);b=read('bound',kappas[-1]);pct=(b['delta_yield']/a['delta_yield']-1)*100
short0=read('bound',0,off=100);short1=read('bound',kappas[-1],off=100)
y0=read('free',0)['yields']['reset'];y1=read('free',kappas[-1])['yields']['reset']
conv=checks['cutoff_comparisons'];dy=max(abs(c['delta_difference']) for c in conv);yerr=max(c['max_yield_difference'] for c in conv)
text=f'''---
asset: membrane-field-results
version: 1.0.0
生成元入力: 膜電位によるスピン配向の可能性を場合わけして検証
依存工程: S15/S3/S4/S5/S6/S16/S17
生成日: 2026-10-09
---

# 膜電場による分子配向・HQ核履歴の条件別検証

**電場は分子の配向分布と反応収率を変えるが、今回の単一双極子・自由回転模型では、長時間のHQ核履歴を回復しなかった。履歴が残るのは引き続き強い結合拘束を与えた条件であり、電場による配向を加えると、代表条件の履歴収率差は {abs(pct):.5f}% 減少した。** これは指定した模型の計算結果であり、生体CRYの測定結果ではない。

## 何を場合わけしたか

- 電場なし＋自由回転、電場のみ＋自由回転、結合拘束のみ、電場で偏った配向分布＋結合拘束を比較。
- 有限電場は4強度。結合率目標 f=0.50000、平均結合時間1 s、結合中の三軸回転ベクトルRMS 1°/5°。両端の電場条件では平均結合時間0.01 sも計算。
- さらに、分子双極子軸が完全に整列しても、その軸まわりの回転が残る極限を、凍結テンソルのx/y/zの3軸について計算。
- 有限電場の準備→収率は14種類、次数比較4実行、完全整列極限3種類。HQスペクトルは8組の強度・分子軸について次数2/3/4の24実行。
- 有限電場の準備→収率は**電場と磁場が平行、双極子が凍結分子座標のz軸**。HQスペクトルでは分子内双極子方向x/yも確認した。分子内軸を変えることは電場と磁場の相対角を変えることとは異なる。

## 対応づけと仮定

温度310 K、膜電位差の大きさ70 mV、膜厚5 nmの例示的な平板模型では、膜内部の電場は14 MV/m。

\\[
U(\\Omega)/k_BT=-\\kappa\\,\\widehat{{p}}\\cdot n,
\\qquad \\kappa=p_{{\\rm eff}}E_{{\\rm local}}/(k_BT).
\\]

κ=0/0.10911/1.09109/10.91094は、この電場が一様にかかるなら有効双極子0/10/100/1000 Dに相当する。あるいは、双極子固定で局所電場の遮蔽を変えた感度解析として読める。**CRYの双極子や局所電場を実測・同定した値ではない。** 膜内部の電場を細胞質全体へ適用していない。

原テンソル、局所散逸、B=50 µT、triplet準備、反応率kS=10⁷/s・escape=4×10⁶/s、H/R待機率10/s・E待機率4/sは既存計算と同じ。自由回転のrank-2相関時間は21.47660 ns。20回の準備サイクル後、次反応の収率を測る。化学状態ごとの双極子変化は仮定していない。

## 電場だけの場合

ρ(n)=π(n)Σφ_a(n)x_a、π∝exp(κ p̂·n)と表現し、πで重みづけした球面調和基底をQRで直交化した。分子とスピン座標の回転をともに含め、回転生成子は共変微分K=L+ad Jの弱形式 −D_R〈Kφ_a,Kφ_b〉π。単に配向を偏らせて回転散逸をゼロにしていない。双極子軸まわりの回転も残る。

Δq=q_full−q_resetは**同一電場条件内**で核履歴を保持する場合とリセットする場合の収率差。〈cosθ〉とP₂は分子配向の指標であり、核スピンの分極ではない。

| κ | 分子〈cosθ〉 | 分子P₂ | reset収率 | 履歴収率差の報告上限 | HQ最遅モード逆減衰率 / µs |
|---|---:|---:|---:|---:|---:|
{chr(10).join(line)}

強い電場では分子が明確に配向している。それでも履歴差は未検出であり、最遅減衰時間は短くなった。上表の上限は報告用の数値分解能で、統計的信頼上限や実験的検出限界ではない。モデルで消えた微小差の符号は解釈しない。

reset収率は {y0:.5f}→{y1:.5f}、変化は {(y1-y0):.5f}（収率の百分率表示で {(y1-y0)*100:.5f}ポイント）。これは電場による反応条件・配向の変化で、過去のスピン状態の記憶とは異なる。

HQの数値は全結合スピン–配向生成子の最も遅い非定常モードの逆減衰率であり、実測の単一核T₁ではない。分子内双極子をx/y/zへ向けた強電場条件の値は、それぞれ {next(r['inverse_slowest_us'] for r in spc if r['axis']==0 and r['kappa']==kappas[-1] and r['cutoff']==4):.5f}/{next(r['inverse_slowest_us'] for r in spc if r['axis']==1 and r['kappa']==kappas[-1] and r['cutoff']==4):.5f}/{next(r['inverse_slowest_us'] for r in spc if r['axis']==2 and r['kappa']==kappas[-1] and r['cutoff']==4):.5f} µs。向きによって数値は変わるが、今回の範囲でms保持にはならない。

## 完全に配向しても残る回転

単一双極子の電場エネルギーは、その軸まわりの回転角に依存しない。完全に整列した極限でも、この回転を −D_R[ J_p,[ J_p,ρ ]] として保持した。

| 完全に整列させる分子軸 | HQ最遅モード逆減衰率 / µs | 準備履歴の収率差 |
|---|---:|---:|
{chr(10).join(f"| {'xyz'[r['axis']]} | {r['inverse_slowest_us']:.5f} | < 1.00000e-10 |" for r in limit)}

したがって、**一つの軸をそろえるだけで今回の履歴消失が解消する、という予測は、このテンソルと回転速度では支持されない。** 異方的分極率や複数点結合で残った回転も拘束する模型は、この単一双極子模型の範囲外。

## 結合拘束と併用した場合

電場は自由集団のドリフトと、自由・結合集団の配向重みπに入る。捕捉・解離は配向とスピンを引き継ぎ、反応中にも働く。結合中の高速微小揺動は前回と同じ近似（τw=0.01 ns）を使用。

**この併用模型では、電場に伴う結合井戸そのものの変形や解離率変化は再計算していない。** 「既定の拘束下で電場が配向分布を変える条件」の検証である。

| κ | RMS 1°のΔq | RMS 5°のΔq |
|---|---:|---:|
{chr(10).join(bound)}

平均結合1 s、1°の比較ではΔqが {a['delta_yield']:.5e}→{b['delta_yield']:.5e}、{abs(pct):.5f}%減少。電場なしですでに履歴が残る拘束条件に、配向の偏りを加えても増強されなかった。平均結合時間0.01 s、1°では {short0['delta_yield']:.5e}→{short1['delta_yield']:.5e}とさらに小さい。配向よりも、結合中の揺動と結合時間への依存が大きい。

等方調和井戸の剛性を元のRMSからc=3/θrms²と置いた診断では、κ=10.91094による局所角度分散の相対変化の上界は1°で0.00111、5°で0.02849。平衡角シフトのノルム上界は0.06355°/1.63214°。小角展開での診断であり、収率の誤差棒ではない。特に5°条件の電場による井戸変形まで精密化した予測とはしない。

## スピンの整列とpopulation

今回の生成子はスピン単位行列を保つ。したがって、初期I/9から静電場だけで核スピン分極を作る機構を入れた計算ではない。分子配向によって、既存の四極子・超微細相互作用の軸と磁場の相対関係が変わることを計算した。

¹⁴Nの四極子と電場勾配の結合自体は否定していない。凍結HQハミルトニアンの310 K Gibbs分布を独立に計算すると、I/9からのtrace distanceは {checks['HQ_thermal_energy_scale']['trace_distance_identity']:.5e}。これはエネルギースケールの参考値で、膜電場による分極の計算値ではない。化学反応による非平衡準備は別の機構。

fullは密度行列を保持し、populationはHQ境界でHQ固有基底の非対角成分を除く。最大full−population差は {checks['diagnostics']['max_full_population_difference']:.5e}。この計算範囲でも境界coherenceの寄与は報告分解能以下。populationは占有確率を意味し、波動関数の振幅そのものではない。

## 数値検証

- 9 tests通過：Boltzmannモーメント、電場ゼロの元生成子との一致、独立な節点積分からの共変Dirichlet行列、trace/identity、軸回転、Smoluchowski方程式のドリフト符号。
- 独立に一括構築したHQ自由/結合連立解と、本計算の分離反復解を照合。状態差最大 {max(checks[k]['state_error'] for k in ('direct_H_off1','direct_H_off100')):.5e}。
- 強電場の全準備サイクルを次数2→3で比較。最大収率差 {yerr:.5e}、履歴収率差の最大差 {dy:.5e}。強電場の表は次数3、それ以外は次数2を表示。5桁表示は5桁の物理精度を保証しない。
- HQスペクトルは次数2/3/4、代表2条件は密行列固有値と独立shift-invertで照合。HQ最遅モードの表は次数4。
- 正の角度積分を増やした生成子比較、解析的Langevinモーメントと独立300点積分も一致。
- 最大trace誤差 {checks['diagnostics']['max_trace']:.5e}、Hermiticity誤差 {checks['diagnostics']['max_hermiticity']:.5e}、節点での最小密度固有値 {checks['diagnostics']['min_sampled_eigenvalue']:.5e}。節点検査は連続全角度に対する正値性の証明ではない。
- 電場ゼロの結合条件を過去の保存値と比較。独立の角度積分による微小差はchecks.jsonへ記録。

## 確度と論文への含意

| 主張 | ラベル | 論拠・範囲 |
|---|---|---|
| 指定した双極子模型では電場が分子配向を偏らせる | Strongly Supported | 解析的Langevin式と独立積分。一般的な膜内ペプチド機構はBishtらのモデル研究、Abstract/pp.584–597 |
| 電場による単一軸整列だけでHQ履歴がmsスケールへ回復する | Contradicted | 今回の有限電場・完全整列極限に限る。現行テンソル・回転係数を固定した数値反例 |
| 既定の結合拘束下で、強電場の配向偏りは履歴差を約13%減らす | Partially Supported | 全周期と次数比較で数値的に確認。拘束下揺動近似・未較正パラメータ・井戸変形未計算に限定 |
| 膜電位で脳内CRYの核スピンが実際に強く整列する | Not Found in Uploaded Sources | 局在、双極子、局所場、電場勾配変化、スピン分極の同時測定なし |
| この効果で神経の停止・続行が変わる | Not Found in Uploaded Sources | 今回は分子計算。神経利得・回路対応の生体較正を追加していない |

論文の現状の主張を膜電位整列によって救済する根拠は、今回の結果からは得られない。電場依存の反応入力と、核履歴を保護する拘束条件を別々に記述する必要がある。電場が無作用という結論でもない。

未検証：EとBが非平行のSO(3)計算、膜法線の分布、電位波形、状態依存双極子・分極率、電場によるEFG/HFC/g/反応率の変化、電場依存の結合井戸と交換率、native局所電場および神経応答。過去の気相分子断片の電場応答係数を現行テンソルへ無検証で流用していない。

## 出典・再現性・レビュー

- Bisht K, Lomholt MA, Khandelia H. *Sensing membrane voltage by reorientation of dipolar transmembrane peptides*. Biophys J 123:584–597 (2024). [doi:10.1016/j.bpj.2024.01.037](https://doi.org/10.1016/j.bpj.2024.01.037)。前ターンで原著Abstractを確認。モデルペプチドのMD/理論でありCRY実測ではない。
- Veinberg SL et al. *Practical considerations for the acquisition of ultra-wideline 14N NMR spectra*. Solid State Nucl Magn Reson 84:45–58 (2017). [doi:10.1016/j.ssnmr.2016.12.008](https://doi.org/10.1016/j.ssnmr.2016.12.008)。Introductionで¹⁴N四極子・EFG依存を確認。膜電位によるCRY分極を示す文献ではない。
- 既存入力：`data/population-memory/latest_case.npz`、`BINDING_MOTION_RESULTS.md`、`DARK_ELECTRONIC_RESPONSE_RESULTS.md`。
- 実装：`impl/hq_electric.py`、`impl/hq_binding.py`の後方互換な生成子・角度基底指定。
- 再現：`sh investigations/2026-10-09-membrane-field/reproduce.sh`。環境・ソースhash・入力hashは`data/membrane-field/manifest.json`。数値表CSV、状態NPZ、独立検証JSON、PNG/PDFを同ディレクトリに保存。
- S16修正：高配向時のGram行列Choleskyを重みつきQRへ変更して精度劣化を回避。分子配向とスピン分極、電場収率差と履歴差、1軸固定と3軸拘束を分離。束縛井戸変形を計算したとの過大な表現を除去。
- S17：主要数値と保存JSONを照合、ゼロ電場回帰・解析式・独立HQ解・次数収束・反例の完全整列極限で確認。生体の未同定事項は未検証と表示。原稿/PDF/GitHub配布物は今回更新していない。
'''
(ROOT/'MEMBRANE_FIELD_RESULTS.md').write_text(text)
