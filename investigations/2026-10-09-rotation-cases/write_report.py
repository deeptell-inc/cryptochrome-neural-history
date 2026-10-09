from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/rotation-cases'
neur=pd.read_csv(D/'neural_selected.csv');mol=pd.read_csv(D/'molecular_summary.csv').set_index('case')
get=lambda label,tau=.2,route='stop':neur[(neur.scenario==label)&(neur.tau_s==tau)&(neur.route==route)].iloc[0]
ci=lambda row:f"{row.delta_stop_pp:+.5f} [{row.conservative_low_pp:+.5f}, {row.conservative_high_pp:+.5f}]"
lines=[]
for label,title in [('clamped','全体固定・元の条件'),('free_null','全体が自由回転（等入力対照）'),('protect_waits','HQ/R/E待機を保護、RP中は回転'),('all_fast_10000','全待機率×10⁴'),('all_fast_1e+06','全待機率×10⁶'),('all_fast_1e+08','全待機率×10⁸'),('field_0_to_50uT','核履歴なし、単回磁場プローブ0対50 µT')]:
 row=get(label);delta='未解像（報告分解能10⁻¹⁰）' if label=='free_null' else f'{row.delta_q:.5e}'
 lines.append(f'| {title} | {delta} | {ci(row)} |')
main='\n'.join(lines)
lines=[]
for label in ['protect_HQ','HQ_fast_10000','HQ_fast_1e+08']:
 row=mol.loc[label];lines.append(f'| {label} | {row.delta:.5e} | 10⁻¹⁰分解能で未検出 |')
negative='\n'.join(lines)
lines=[]
for f in [0.,.001,.01,.1,.5,1.]:
 row=get(f'protected_fraction_{f:g}');lines.append(f'| {f:.5f} | {row.delta_q:.5e} | {ci(row)} |')
mix='\n'.join(lines)
lines=[]
for label in ['all_fast_10000','all_fast_1e+06','all_fast_1e+08']:
 row=mol.loc[label];gapfraction=row.full_population_gap/row.delta
 lines.append(f'| {label} | {row.mean_preparation_time_s*1e6:.5f} | {row.full_population_gap:.5e} | {gapfraction*100:.5f}% |')
speed='\n'.join(lines)
storage=json.loads((D/'protected_wait_storage.json').read_text());chem=json.loads((D/'independent_chemical_storage.json').read_text())
ctable='\n'.join(f"| {r['tau_s']*1000:.5f} | {100*r['peak_oxidized_fraction']:.5f}% | {r['one_percent_input_delta_at_200ms_pp']:.5f} |" for r in chem)
text=f'''---
asset: rotation-cases-results
version: 1.0.0
生成元入力: HQ_ROTATION_RESULTS.md; current frozen operators; archived neural model
依存工程: P4 continuation / S15 / S5 / S6 / S16 / S17
生成日: 2026-10-09
---

# 回転拡散を含むケース別再計算：何の結論が変わるか

**結論は変わる。自由回転下では、秒単位の核population履歴を神経回路の入力源とする元の大きな停止効果は維持されない。** ただし、すべてのスピン化学応答が消えるわけではない。分子の保護、高速な再準備、即時の磁場応答を化学状態へ写す条件は分けて扱う必要がある。

この報告の神経数値は、元と同じ人工回路・仮定したゲインによる条件付き再計算である。実神経や行動の効果量を測定・較正した結果ではない。

## 1. 計算条件と主結果

核履歴のケースは現行HQ/R/Eテンソル、50 µT、singlet反応率10⁷ s⁻¹、escape率4×10⁶ s⁻¹、局所散逸を維持した。追加酸素電子緩和・電子交換・双極子を0とする元の仮定も維持しており、その生理的妥当性を新たに確認したものではない。基準の自由回転相関時間は21.47660 ns。20完了サイクル後の次反応収率を再計算した。前回の9.72000 / 99.12000 nsの自由回転結果も履歴消失を支持しているが、今回の追加ケースの網羅的な生体パラメータ探索ではない。

各ケースのreset収率をq₀、full収率をq₁として、既存M4の入力を置き換えた。下表は**プール10000、ゲイン1024、top8、stop経路、化学寿命200 ms、時間積分読出し**。3乱数種×8192ペア＝24576ペア/条件。単位ppは停止確率のpercentage points。角括弧は保守的な95% Monte Carlo区間で、生物学的なパラメータ不確実性を含まない。

| ケース | 反応収率差 Δq | 停止確率差 pp [95%区間] |
|---|---:|---:|
{main}

自由回転行は、数値分解能以下の残差を信号化せずq₁=q₀に置いた等入力対照。これが数学的に「真の差は厳密ゼロ」と証明されたという意味ではない。仮に未解像差を|Δq|≤10⁻¹⁰に制限する感度評価なら、一回の準備に対するcoupling上限は

`100 × [1 − (1 − |Δq|)^10000] ≤ 0.00010 pp`

となる。この上限はゲインや読出し方式に依存しない。多数決・重み付き・時間積分・競合型の全てで、等入力なら二群の結果は同じだった。生物学的誤差や模型の欠落まで10⁻¹⁰以下と保証する上限ではない。

## 2. HQだけを保護・高速化しても回復しない

| ケース | 計算された生の収率差 | 解釈 |
|---|---:|---|
{negative}

HQ以外の回復・還元区間で履歴が失われるため、HQ単独の改善では足りない。`protect_HQ`はHQ区間だけ回転拡散係数を0にし、他区間は自由回転のままとした。`HQ_fast`はHQ待機率だけを変更した。

これに対しHQ/R/Eの**全待機状態**を不動化し、RP中だけ回転させる仮想条件ではΔq≈0.00017が残った。保護HQでの追加保存の読出し半減期は**{storage['half_decay_s']:.5f} s**。角度分布だけが担う収率差は約{storage['angular_trace_only_contribution']:.5e}で、初期の主信号は核状態の差である。分子の結合/解離や固定化を実測した模型ではなく、状態依存の不動化を外から与えた対照である。

## 3. 再準備を速くすると、短い信号とcoherence寄与が現れる

RPの反応・escape率は固定し、HQ・還元・分岐回復の待機率を一括して変更した。

| ケース | 20サイクル準備の平均所要時間 µs | full−population収率差 | 全履歴差に対する割合 |
|---|---:|---:|---:|
{speed}

×10⁸ではHQ/還元/P回復が10⁹ s⁻¹、E酸化が4×10⁸ s⁻¹となる。これを生理的に可能な反応速度とは主張しない。平均準備時間は期待値で、全分子が同じ時刻に20回反応する模型ではない。

×10⁸で得た約0.00020の履歴差も、読み出し前に自由回転HQで100 µs追加保存すると約7.40398×10⁻¹⁰、1 msで約3.74942×10⁻⁴⁸へ減少した。後者の極小桁を物理的予測精度として採用しない。したがって、下流で効果が出た×10⁸行は、**極短時間で読み出した差を別の化学プールが200 ms保持する**というM4の仮定に依存する。

×10⁴の神経出力で差が観測されなかったことは無効果の証明ではない。この条件では分子差を用いた一回準備の解析的上限が0.00288 ppとなり、有限試行の区間より強い制約になる。

元の「fullとpopulationが一致する」は遅い待機条件に限定される。高速条件ではHQ境界を跨ぐcoherenceの除去が反応収率を変えた。ただし、これは特定の境界射影との比較であり、量子計算優位や生体でのcoherence利用の実証ではない。

神経出力では×10⁶のfullが+0.03662 pp、populationが+0.04476 ppだったが、この小さい差から優劣は判断できない。×10⁸ではfullが+0.50863 pp、populationが+0.37028 ppとなったが、両者のMonte Carlo区間も重なる。分子でのcoherence寄与と、回路でそれを識別できることを混同しない。

## 4. 保護された分子が一部だけ存在する場合

非交換の二集団として、固定配向部分fと自由回転部分1−fの収率を凸結合した。固定部分は元の基準と同じ配向であり、不動分子のランダム配向分布は今回計算していない。fは**不動分子の割合**であり、単一分子の角度拘束の強さや秩序パラメータではない。神経入力では分子種を独立に標本化する均質化したbinomialプールを仮定した。固定組成で分子ごとの感受性が異なる場合の分散まで同じとは限らない。

| 固定部分 f | 反応収率差 | 停止確率差 pp [95%区間] |
|---:|---:|---:|
{mix}

固定分子1%のケースでも、今回の高ゲイン条件では+0.14242 ppが得られた。0.1%の+0.01628 ppは保守的区間が0を含む。これらは必要な固定分子割合の実測値でも、どのシナプスがその分子を含むかの同定でもない。top8は仮定した回路重みと遅延で選んだ接続である。

## 5. 核履歴がなくても、瞬時の磁場応答は別に残る

自由回転・核resetでの収率は、B=0で0.34072、B=50 µTで0.34151だった。この差は約7.92616×10⁻⁴で、過去の核状態の保持を必要としない。

このケースは**単回の反応プローブで磁場条件を変えた二群**を比較する。M4ではcue時の一回のプール準備に反映し、その後は共通背景に回復する。自然の合図や意思が磁場を変えるという模型でも、磁場を変え続ける連続駆動模型でもない。

化学寿命200 msの仮定ではstop経路で+1.99382 pp、triggerで+2.90527 pp、goで−2.11995 ppとなった。同じ収率差でも作用先によって符号が変わる。化学寿命12 msではstop経路は+0.00814 ppで保守的区間が0を含み、triggerは+0.48828 ppだった。瞬時のスピン化学応答、化学記憶、時間依存の回路読出しを分ける必要がある。

## 6. 元の論文の各部分はどう変わるか

| 部分 | 再評価 |
|---|---|
| M1：秒単位の核population履歴 | 自由回転条件では成立しない。配向拘束・保護条件を明示する必要がある |
| M1：full≃population | 元の遅い条件・保護待機では維持。極端な高速再準備へ一般化できない |
| M2：反応枝の合流 | 式は維持。入力Δqを更新すると履歴依存の枝差も減る。最終共通生成物への合流は核履歴を救わない |
| M3：独立入力でのHk化学記憶 | 独立入力を据え置けば結果は維持。CRY由来の書込みが同定されたことにはならない |
| M4：大きな停止差 | 元の核履歴入力では自由回転と両立しない。保護・別の即時入力・化学保持があれば条件付きの差は得られる |
| M5：公開電流データの再解析 | 分子収率を入力にしていないため、回転追加ではフィット結果を変更しない。速いCRY→神経ゲイン未同定という結論も維持 |

M2の `ΔY_H2O2(t)=Δq exp(−4t)/2` を新しい分子差で再評価した表を保存した。これは完全な再酸化・不均化・同一区画への合流という元の仮定の下での式である。

独立Hk入力（a₀=10 s⁻¹、off=0.01000 s⁻¹、再結合100 s⁻¹）も再計算した。入力振幅を1%増加させる対照は、CRY収率差から較正したものではない。

| 入力寿命 ms | 酸化状態の最大割合 | 1%入力増加による200 ms時点の酸化割合差 pp |
|---:|---:|---:|
{ctable}

核状態が短命でも、**既に書き込まれた古典化学状態**が残ることはあり得る。しかし、どの短いCRY信号が、どれだけ、どの速度で書き込まれるかは依然未同定である。

## 7. 検証・確度・再現性

- 分子の角度次数2→3、trace/Hermiticity、最終状態の正値性を確認。追加ケースの収率差の次数間変化は最大約3.93147×10⁻⁸。主要反応差は独立Arnoldi反応解と約10⁻¹⁴以内で一致した。
- 保護HQの保存曲線を、角度ごとのpopulation生成子で独立に再計算し、収率差3.60227×10⁻¹²以内で一致した。半減期も小数点以下5桁で一致した。
- 待機のidentity/traceless分離を、別の射影部分空間の密な線形解で検証した。保護待機を含む最大差は2.19303×10⁻¹¹。
- 神経回路は元の実行環境（NumPy 2.5.0）に揃え、元の6条件×3乱数種の**18試行ファイルでイベント時刻・結果を完全再現**した。別環境NumPy 2.2.6の診断計算は最終表には混ぜていない。
- 保存イベントから停止・続行・未解決を再分類し、確率範囲と割当予算を確認した。両群が同じ入力の場合の一致も検証した。
- 95%区間は、ペアで増加/減少した試行数から各確率のClopper–Pearson区間を作り、Bonferroniで組み合わせた保守的区間。観測差0のケースを「誤差0」と表示しない。通常のMonte Carlo標準誤差と区間もCSVに保存した。
- Hk ODEはDOP853とRadauで照合し、差は10⁻⁸未満だった。

| 主張 | ラベル | 論拠・限定 |
|---|---|---|
| 自由回転では元の大きな核履歴由来停止差を支えない | Strongly Supported（模型内） | 分子再計算、同入力の全回路一致、一回準備coupling感度上限 |
| HQ単独の保護では不十分、全待機の保護で小さい差が残る | Strongly Supported（模型内） | 状態別介入・角度収束・独立反応解 |
| 高速再準備では境界coherenceの寄与が生じる | Strongly Supported（模型内） | full/population比較、角度次数と数値誤差との比較 |
| 自由回転が全ての瞬時磁場応答を消す | Contradicted（この模型） | B=0と50 µTの核reset収率が異なる |
| 脳内でも表の停止効果が生じる | Not Found in Uploaded Sources | 固定分子割合、書込み速度、局所ゲイン、対象接続が未同定 |
| 保護・極高速化が生理的に成立する | Not Found in Uploaded Sources | 仮想介入であり、その生体条件を裏付ける新しい測定はない |

S16では、瞬時応答と核履歴の混同、数値未解像差の厳密ゼロ扱い、非有意な小差の順位付け、固定分子割合と角度拘束の混同を修正した。S17では数値を保存出力・独立解法に照合し、条件外の生体因果主張を採用しなかった。既存文献が未同定の係数を埋めたとは主張していない。

## 8. 成果物

- `data/rotation-cases/molecular_summary.csv`：分子ケース別結果
- `data/rotation-cases/neural_summary.csv`：全読出し方式・全割当・区間
- `data/rotation-cases/neural_selected.csv`：本文で比較した時間積分/top8
- `data/rotation-cases/protected_wait_storage.json`：保護ケースの保存曲線
- `data/rotation-cases/routing_summary.csv`：反応枝合流の更新
- `data/rotation-cases/independent_chemical_storage.json`：独立化学入力の再計算
- `data/rotation-cases/checks.json`、`manifest.json`：検証・来歴
- `investigations/2026-10-09-rotation-cases/reproduce.sh`：再実行

原稿・PDF・GitHub配布物の改訂は別工程として残している。

![ケース比較]({str(D/'case_summary.png')})

図(c)の10⁻¹²未満の値は表示下端に置いた。下端を真の残留値とは解釈しない。
'''
(ROOT/'ROTATION_CASES_RESULTS.md').write_text(text)
