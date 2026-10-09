"""Generate the bounded interpretation from completed calculation artifacts."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/field-couplings'
read=lambda name:json.loads((D/name).read_text())
f=lambda x:f'{x:.5f}'
e=lambda x:f'{x:.5e}'
checks=read('checks.json');spectrum=read('HQ_spectrum.json');free=read('free_electronic_L2.json')
rows=[read(f'{m}_L2_angle{a}.json') for a in (1,5) for m in ('baseline','electronic','well','combined')]
texts=['''---
asset: field-couplings-results
version: 1.0.0
生成元入力: 電場によるEFG・超微細相互作用や結合井戸の変化計算
依存工程: S3/S5/S6/S16/S17
生成日: 2026-10-09
---

# 電場による電子応答・結合井戸変形の条件別計算

現行スピン模型と同じ電子状態・幾何構造・静的点電荷環境から、電場に対するEFGと超微細結合の一次応答を計算した。さらに、仮定した結合井戸の変形を加え、状態準備20周期後の次反応収率まで計算した。**以下は指定した固定構造と条件付き拘束模型の計算結果であり、脳内の測定値ではない。** 表示は小数点以下5桁、微小量は科学表記とし、保存データは丸めていない。この表示規則は物理的精度の保証ではない。

## 1. 対応する電子状態と電場

- Hは閉殻のanionic HQ、Rは閉殻の酸化状態、Eは開殻のneutral SQ。HQでも¹⁴Nの四極子–EFG結合は存在する。閉殻H/Rには今回の非相対論模型で不対電子の超微細結合を設定しない。
- H/Rは半径12 Åの固定MM点電荷＋def2-SVPD、Eはfull minimum-image MM＋N上の非収縮pcJ-2。B3LYP、grid4、density fittingと保存チェックポイントを元計算に合わせた。
- 一様外部電場の直接の空間勾配はゼロだが、電子密度の再分布が核位置のEFGを変える。EFGの応答にはこの電子再分布を含めた。超微細応答にはFermi contactとspin-dipolar項を含めた。
- 電場は14.00000 MV/m、磁場は50.00000 µT、両者は平行。配向用κ=10.91094は前計算と同じ仮定した有効双極子1000 D・310 Kの例であり、CRYの実測値ではない。局所場と膜内の巨視的な場の一致も未同定である。

電子応答は χᵛₐᵢⱼ=∂Vᵢⱼ/∂Eₐ、χᴬₐᵢⱼ=∂Aᵢⱼ/∂Eₐ として3方向すべてを求め、元の局所MD回転を3階テンソルとして適用してから平均した。現行の静的外部EFGは二重加算しない。全体回転と拘束揺動には更新したH(E)を使用する。

|状態|核|最大EFG変化 / 元のQM EFG (%)|最大等方超微細変化 (MHz)|元の等方超微細 (MHz)|
|---|---|---:|---:|---:|''']
for st in ('H','R','E'):
 for r in read(f'{st}/response.json')['nuclei']:
  hfc='該当なし（閉殻）' if st!='E' else f(r['isotropic_HFC_max_change_MHz_at_14MVm'])
  base='該当なし' if st!='E' else f(r['baseline_isotropic_HFC_MHz'])
  texts.append(f"|{st}|{r['atom']}|{f(100*r['EFG_max_relative_change_at_14MVm'])}|{hfc}|{base}|")
texts.append(r'''
ここで「最大」は、固定電場強度で方向を動かしたときの線形応答の最大値である。EFGはFrobeniusノルムを用い、分母は凍結QM断片のEFGである。超微細の欄は等方部分の最大変化であり、全テンソルは保存NPZと収率計算に含まれる。各欄の最大方向は必ずしも同一ではない。

## 2. 結合井戸は何を計算したか

結合相手が特定されていないため、nativeの結合自由エネルギー面を量子化学で決定したわけではない。従来の調和的回転拘束へ電気双極子の項を加え、条件付きに井戸を変形した：

\[U(R)/(k_BT)=\tfrac12 c|\log R|^2-\kappa\,\mathbf n\cdot R\mathbf p,
\qquad c=3/\theta_{\rm rms,0}^2.\]

最小位置を非線形方程式で求め、SO(3)上の接空間Hessian Kから共分散K⁻¹を得た。回転移動度を電場なしの値に固定し、固有方向ごとの揺動相関時間をτᵢ=cτ₀/λᵢとした。基準τ₀=10.00000 ps。6点の正の角度求積で平均演算子を作り、微小角・高速雑音近似で散逸を構成した。

|元のRMS角 (°)|κ|配向重み付きRMS角 (°)|求積点での最大平均角変位 (°)|局所Laplace積分の正規化誤差|
|---:|---:|---:|---:|---:|''')
for r in read('well_averages.json'):
 texts.append('|'+ '|'.join([f(r['rms0_deg']),f(r['kappa']),f(r['weighted_rms_deg']),f(r['max_shift_deg']),e(r['laplace_partition_normalization_error'])])+'|')
texts.append('''
強電場の代表的なアンカー方向では次のようになる。F_resは既に配向分布に含めた−κ n·pを差し引いた局所調和自由エネルギー補正である。

|元のRMS角 (°)|アンカーと電場の角度 (°)|平均角変位 (°)|変形後RMS角 (°)|τ最小–最大 (ps)|F_res / kBT|
|---:|---:|---:|---:|---:|---:|''')
for raw in csv.DictReader((D/'well_cases.csv').open()):
 r={key:float(value) for key,value in raw.items()}
 if r['kappa']>10 and r['anchor_angle_deg'] in (0,90,180):
  texts.append(f"|{f(r['rms0_deg'])}|{f(r['anchor_angle_deg'])}|{f(r['mean_shift_deg'])}|{f(r['rms_deg'])}|{f(r['tau_min_s']*1e12)}–{f(r['tau_max_s']*1e12)}|{f(r['F_res_kBT'])}|")
texts.append('''
配向重み付きRMSは従来の配向分布πでの平均という診断量であり、反応後の実現分布を用いた観測値ではない。正方向と逆方向の電場、κ=0、弱電場、アンカー角0/45/90/135/180°の計50条件は`well_cases.csv`に保存した。電場と平行な井戸は主に硬くなり、逆向きの井戸は軟らかくなる。一般角では平均位置もずれる。

等方的なアンカー分布と同じ双極子を持つ自由／結合状態では、厳密な全配向分配関数が因数分解され、**この回転井戸の変形だけでは総平衡結合割合は変わらない**。表の微小な正規化誤差はLaplace近似の誤差として除去した。角度ごとの捕捉分布は変えるが、基準結合割合0.50000、解離率1.00000 s⁻¹を維持した。反応周期中の結合割合は平衡値と異なり得る。解離障壁、結合相手、遷移経路がないため、nativeの結合増加・解離率低下は算出していない。

## 3. 電子応答と井戸変形を分離した収率

全4条件で同じ強い分子配向ポテンシャルκ=10.91094を使用した。「配向のみ」は電場ゼロを意味しない。fullは核密度行列を次周期へ引き継ぎ、populationはHQ境界で当該Hamiltonian（結合状態では井戸内平均）の固有基底における対角成分のみ引き継ぎ、resetは核履歴を消す対照である。ここでpopulationは状態の確率・対角相関を保持する操作であり、波動関数の「振幅のみ」と同義ではない。

自由回転相関時間21.47660 ns、結合RMS角1°/5°、平均結合時間1 s。triplet準備、singlet反応10⁷ s⁻¹、escape 4×10⁶ s⁻¹、H/R待機10 s⁻¹、E待機4 s⁻¹は従来通り。Δq=q(full)−q(reset)を履歴収率差と定義する。

|拘束RMS角 (°)|追加機構|q(full)|q(reset)|Δq|配向のみからのΔq変化 (%)|
|---:|---|---:|---:|---:|---:|''')
names={'baseline':'配向のみ','electronic':'電子応答','well':'井戸変形','combined':'電子応答＋井戸変形'}
for r in rows:
 base=next(x['delta_yield'] for x in rows if x['mode']=='baseline' and x['rms_deg']==r['rms_deg'])
 texts.append(f"|{f(r['rms_deg'])}|{names[r['mode']]}|{f(r['yields']['full'])}|{f(r['yields']['reset'])}|{e(r['delta_yield'])}|{f(100*(r['delta_yield']/base-1))}|")
texts.append('\n表は角度次数L=2、combinedはL=3でも検証した。Δqは丸める前の値から計算した。小さい差については独立求解・次数検証に加えて模型近似誤差との比較が必要である。')
for a in (1,5):
 base=next(r['delta_yield'] for r in rows if r['mode']=='baseline' and r['rms_deg']==a)
 el=next(r['delta_yield'] for r in rows if r['mode']=='electronic' and r['rms_deg']==a)
 both=next(r['delta_yield'] for r in rows if r['mode']=='combined' and r['rms_deg']==a)
 texts.append(f'\n{a}°拘束では、電子応答単独の相対変化は{f(100*(el/base-1))}%、両機構を加えた変化は{f(100*(both/base-1))}%だった。')
texts.append(f"\n全表のfull−population差の最大絶対値は{e(checks['cycle_diagnostics']['max_full_population'])}。この遅い周期条件での収率差は、引き継ぐpopulationでほぼ再現される。ただし周期内のコヒーレントなスピンダイナミクスは残り、完全な古典模型への同一視はしない。")
texts.append('\n![電子応答と収率の比較](data/field-couplings/field_coupling_cases.png)')
texts.append('''
## 4. 自由回転の履歴消失は解消するか

|電子応答用電場 (MV/m)|角度次数L|HQ最遅減衰率の逆数 (µs)|
|---:|---:|---:|''')
for r in spectrum:
 texts.append(f"|{f(r['field_MVm'])}|{r['cutoff']}|{f(r['inverse_slowest_us'])}|")
texts.append(f"\nこれはスピンの実測T₁ではなく、指定生成子の最遅非自明モードの逆減衰率である。ここでは配向κを正の値に固定して電子応答の符号だけを反転した因子分離対照を含む。膜電位全体の符号反転ではない。自由集団のみの20周期後のΔqは{e(free['delta_yield'])}、full−population差は{e(free['full_population_difference'])}。この条件では、静的電子応答を加えても自由回転による履歴消失を救済しない。")
texts.append('''
## 5. 数値検証と誤差の区別

|電子状態|直接SCF電場 (MV/m)|EFG一次微係数の相対差 (%)|HFC一次微係数の相対差 (%)|EFG偶成分 / 一次変化 (%)|
|---|---:|---:|---:|---:|''')
for st in ('H','R','E'):
 for r in checks[st]['checks']:
  texts.append(f"|{st}|{f(r['amplitude_V_m']/1e6)}|{f(100*r['EFG_derivative_relative_error'])}|{f(100*r['HFC_derivative_relative_error']) if 'HFC_derivative_relative_error' in r else '該当なし'}|{f(100*r['EFG_even_nonlinearity_relative_to_linear'])}|")
texts.append('''
直接SCFは混合方向(1,2,3)/√14の正負電場で実行した。偶成分は非線形効果とSCFの数値残差の両方を含み、微小な非線形係数を同定したとは主張しない。一次応答の方向依存全体を有限電場で独立検証したわけでもない。''')
diag=checks['cycle_diagnostics'];ind=checks['independent_cycle'];w=checks['well_quadrature']
texts.extend([
f"\n- 収率のL=2/3最大絶対差：{e(max(x['max_yield_difference'] for x in checks['cutoff_comparisons']))}。履歴差の最大絶対差：{e(max(abs(x['delta_difference']) for x in checks['cutoff_comparisons']))}。",
f"- HQ直接連立解と反復解の差：{e(ind['direct_H_state_error'])}。第三階テンソル変換の独立差：{e(ind['third_rank_transform_error'])}。",
f"- 井戸Hessianを独立差分で検証。Haar測度を含む3次元Gauss–Hermite積分と調和近似の共分散差は最大{f(100*w['max_harmonic_covariance_relative_error'])}%、求積次数7/11差は最大{e(w['max_7_11_quadrature_difference'])}。",
f"- サンプル密度行列のtrace誤差{e(diag['max_trace'])}、Hermiticity誤差{e(diag['max_hermiticity'])}、最小固有値{e(diag['min_eigenvalue'])}。",
'- 入力チェックポイントの零電場EFG/HFCを再現し、応答方程式残差と電子数保存を確認した。tests.logとacceptance.jsonが最終判定を記録する。'
])
texts.append(f"\n微小角差分の前にHamiltonianの厳密なHermiticityを明示的に保ち、丸め誤差による人工的な非unital成分を除いた。変更前後の収率の最大差は{e(max(x['max_yield_change'] for x in checks['hermitian_projection_regression']))}である。ランダム複素状態を含むHQ直接解と、スピン単位行列の保存を検証した。14テストが通過し、変更前の計算はbefore_hermitian_projectionに保存した。")
texts.append('''
断片分極率からの誘起双極子エネルギーの尺度（14 MV/m、310 K）は以下の通り。この断片の分極率はタンパク質全体の分極率ではなく、今回の井戸ポテンシャルには追加していない。

|状態|最大誘起エネルギー絶対値 / kBT|方向間エネルギー幅 / kBT|
|---|---:|---:|''')
for r in checks['fragment_polarizability']:
 texts.append(f"|{r['state']}|{e(r['max_induced_energy_kBT'])}|{e(r['orientation_energy_span_kBT'])}|")
texts.append('''
## 6. 結論の範囲と確度

|主張|ラベル|根拠と限定|
|---|---|---|
|現行固定構造で電場はEFGとSQ超微細結合を変える|Strongly Supported|解析的応答と正負電場SCFが独立に支持。指定電子構造計算法の内部での主張。|
|仮定した拘束井戸は電場で変形する|Strongly Supported|極小/Hessianの直接差分、独立の全非線形Boltzmann積分で検証。native結合井戸とは未同定。|
|井戸変形や電子応答は拘束された集団の履歴収率差を変える|Partially Supported|次数比較と独立HQ求解で支持。ただし微小角・高速揺動、固定局所浴、粗視化した捕捉、一定反応率に依存。|
|今回の静的電場電子応答により自由回転でもms履歴が回復する|Contradicted|今回の14 MV/m・現行HQテンソル・自由回転時間について、HQスペクトルと20周期収率の双方が反証。全電場機構の一般否定ではない。|
|電場はnative結合割合や解離寿命を増加させる|Not Found in Uploaded Sources|結合相手と解離障壁が未同定。等方アンカー・同一双極子という限定模型では総平衡割合は不変。|
|この結果が脳神経の停止／続行を決める|Not Found in Uploaded Sources|局在・電場遮蔽・実効双極子・結合井戸・実際の揺動・下流神経利得が未較正。今回の神経再計算はない。|

**中心的な結論は、自由回転の履歴消失と、十分に拘束された集団での条件付き履歴残存という従来の区別が維持されること。** 一方、拘束井戸を電場に依存しないとした前計算の収率差は修正される。小さな電子応答の寄与は模型誤差を超える生物学的効果とみなさない。

特に、元の局所MD揺らぎから作った浴係数Kは固定している。局所MDの温度は298 Kであり、κ換算に用いた310 Kで浴を再取得した計算ではない。H(E)の変化に応じた全体回転・拘束揺動の緩和は更新したが、電場が局所電荷雑音スペクトル、溶媒・側鎖運動、反応・プロトン移動速度を変える効果は未計算である。固定構造を超えた幾何最適化、MM分極、空間的電場勾配、SOC/g応答、配向と膜内深さの連動も含まない。したがって「すべての電場効果を精査した」とはしない。

## 7. 出典・再現

新しい数値の一次根拠は本計算の電子応答NPZ、有限電場SCF、cycle JSON、独立検証JSONである。過去の気相断片の感受率は流用していない。

1. Bisht, Lomholt & Khandelia (2024), *Sensing membrane voltage by reorientation of dipolar transmembrane peptides*, Biophysical Journal **123**, 584–597. [doi:10.1016/j.bpj.2024.01.037](https://doi.org/10.1016/j.bpj.2024.01.037). Abstract：膜内モデルペプチドの配向について支持する。CRYの脳内局在・双極子・井戸・結合速度を支持する文献としては使用しない。
2. Veinberg et al. (2017), *Practical considerations for the acquisition of ultra-wideline ¹⁴N NMR spectra*, Solid State Nuclear Magnetic Resonance **84**, 45–58. [doi:10.1016/j.ssnmr.2016.12.008](https://doi.org/10.1016/j.ssnmr.2016.12.008). Introduction：¹⁴Nの四極子–EFG相互作用の一般的根拠。膜電位による脳内核スピン分極の実証ではない。

- 仕様・入力対応：`investigations/2026-10-09-field-couplings/spec.md`、`source_scope.md`。
- 再現：`sh investigations/2026-10-09-field-couplings/reproduce.sh`。電子応答の再実行は長時間を要する。
- 数値一覧：`data/field-couplings/electronic_summary.csv`、`cycle_summary.csv`、`well_cases.csv`。
- 検証とhash：`data/field-couplings/checks.json`、`acceptance.json`、`manifest.json`。
- 現行凍結population-memory NPZを維持。論文本文・PDF・投稿物・GitHub公開物はこの作業では更新していない。
''')
(ROOT/'FIELD_COUPLINGS_RESULTS.md').write_text('\n'.join(texts))
