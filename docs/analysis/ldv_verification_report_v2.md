# Cosmos Policy 感覚運動結合動態検証 第2弾：言語の役割の再定位と真の生成行動によるD空間再構築

## 0. 本レポートについて

本レポートは `latent_dynamics_verification_report.md`（以下「前回レポート」）が実行した4フェーズ
検証（D空間構築・エネルギー場推定・動的Vector Field Steering・評価指標）の結果を受け、その§9
「今後の課題」および、それを踏まえて追加提案された「フェーズ5〜7」検証指示書の内容を実行した
結果をまとめる。

前回レポートの中心的結論は次の2点だった。(1) 感覚（潜在表現の進行多様体）と運動（エンドエフェクタ
・グリッパーの物理的効果）を結合した遅延座標埋め込み空間（D空間）は構築可能であり、元の表現では
見えなかった力学的構造（PnPCounterToCabにおける「成功=自己安定化・失敗=不安定化」パターン）を
顕在化させた。(2) しかし、言語プロンプトを完全に排除した状態でのベクトル場への動的介入
（dynamic vector field steering）は、単一の複雑なタスク（PnPCounterToCab、複数の操作可能物体を
含むキッチンシーン）ではタスク完遂を一切誘発できず、中心仮説「言語なしでも動的介入で自律的に
タスクへ引き込まれる」は反証された。ただし、動的介入下の行動は最後まで感覚フィードバックに強く
従属して変化しており（観測固定条件との経路長比4.20倍、`p=7.8×10^-5`）、感覚運動ループとしての
機能は保たれていた。

本レポートはこの結果を踏まえ、3つの独立した仮説を追加検証する。

1. **フェーズ5（単一アフォーダンスタスクでの再検証）**: PnPCounterToCabでの反証は、タスクが
   複数の操作可能物体・複数の起こりうる操作を含む複雑なシーンであり、「どの物体をどう扱うか」
   という選択情報を言語が担っていたことが原因ではないか、という前回レポート§6・§9の解釈を、
   操作対象がより一意な単一アフォーダンスタスク（CoffeePressButton・CloseDrawer）に同じ
   パイプラインを適用することで直接検定する。
2. **フェーズ6（言語のエントレインメント仮説）**: 言語プロンプトを「行動系列全体を持続的に
   記述・制御する表象」としてではなく、「力学系を特定のアトラクタ盆へ引き込む一過性のトリガー」
   として再定位できるか、エピソード内でのプロンプト動的切り替え（fadeout）により検定する。
3. **フェーズ7（真の生成行動によるD空間再構築）**: 前回レポート§2.6・§9で開示した限界——D空間の
   結合ベクトルが「行動の結果」（eef/gripperの環境物理量の差分）を代理指標として使い、モデルが
   実際に生成した行動チャンク（X_hat_0）そのものを使っていなかったこと——を解消し、真に生成された
   運動指令を結合した空間で前回レポート§3.3のDMD力学解析を再評価する。

前回レポートおよびその前提となった `verification_report_v3.md`・`attractor_verification_report.md`
（以下、先行検証）が確立した基盤（モデル・データ収集・プローブ層・統計手法・循環性回避の原則、
P1〜P8の不変原則）をそのまま踏襲する。実装中に発見したバグは隠さず該当節にまとめ、結果は都合の
良い部分だけを取り上げず正直に報告する——この姿勢は前回レポートから変えていない。

> **編集注記**: 本文中のスクリプトパスは全て
> `cosmos_policy/experiments/robot/robocasa/analysis/verification/latent_dynamics/` 以下の
> 現行パスである（詳細: `verification/README.md`）。

---

## 目次

- [1. 実験設定と検証基盤](#1-実験設定と検証基盤)
- [2. フェーズ5：単一アフォーダンスタスクにおける完全自律駆動の再検証](#2-フェーズ5単一アフォーダンスタスクにおける完全自律駆動の再検証)
- [3. フェーズ6：アトラクタへの「意図の引き込み」（Entrainment）検証](#3-フェーズ6アトラクタへの意図の引き込みentrainment検証)
- [4. フェーズ7：D空間の「真の生成行動」へのアップグレード](#4-フェーズ7d空間の真の生成行動へのアップグレード)
- [5. 統合結論](#5-統合結論)
- [6. 実装の詳細：発見・修正したバグ](#6-実装の詳細発見修正したバグ)
- [7. 使用スクリプトと再現性](#7-使用スクリプトと再現性)
- [8. 今後の課題](#8-今後の課題)

---

## 1. 実験設定と検証基盤

**実行環境**: Singularity不使用（`.venv`直接アクティベート、`uv`管理のPython 3.10）。GPU2基
（RTX 4090×2）を使い、フェーズ5・6・7のオンラインロールアウトを並列実行した
（`CUDA_VISIBLE_DEVICES=0`/`1`、24GB中1プロセスあたり実測8.7〜8.8GB、2プロセス同居も可能な
余裕があった）。シミュレータはRoboCasa（robosuiteベース、PandaMobileロボット）、GPUレンダリングは
EGLヘッドレス（`MUJOCO_GL=egl`）。

**モデル**: `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`（前回レポートと同一、28ブロックDiT、EDM
拡散フレームワーク、`num_denoising_steps_action=5`）。プローブ層はBlk-13（前回・先行検証と同一）。

**データ**: フェーズ5・6は前回レポートが構築済みの
`results/latent_dynamics_verification/dynamics_embedding_artifact_<task>.pkl`
（フェーズ1のD空間エンコーダ・進行多様体PCA・成功エピソードのフローライブラリを含む）を
そのまま再利用する——`dynamic_vector_field_steering.py`はタスク名を引数に取る設計で
最初から一般化されていたため、新規のオフライン再構築は不要だった。フェーズ7のみ、後述の
理由により新規のGPUロールアウト収集（`collect_action_chunks.py`、行動チャンク付き）を要した。

---

## 2. フェーズ5：単一アフォーダンスタスクにおける完全自律駆動の再検証

### 2.0 要旨

前回レポートの動的Vector Field Steeringパイプライン（`dynamic_vector_field_steering.py`）を
一切変更せず、`--task_name`のみを差し替えて単一アフォーダンスタスク（CoffeePressButton・
CloseDrawer）に適用した。**両タスクとも、C1〜C4（ダミープロンプト条件、動的フィールド・静的
steering・ランダム方向いずれを含む）は全て`success_rate=0.00`（各n=8）だった**——
PnPCounterToCabで見られた反証が、操作対象が一意な単純タスクでは解消されるという前回レポート§6の
仮説的解釈は、この2タスクでは**支持されなかった**。唯一の例外はCloseDrawerのC5（観測固定+動的
steering）で1/8が偶然成功したが、これは観測を更新する通常条件（C2）ではなく観測を固定した統制
条件での結果であり、「言語なしでのタスク遂行」の証拠にはならない。動的因果効力（DTW/Fréchet
距離）・多様体逸脱度でもC2は他のダミー条件と統計的に区別できず（全て`p>0.11`）、前回のPnP同様
の否定的パターンを再現した。感覚フィードバック従属性（C2 vs C5経路長比）はCoffeePressButtonでは
前回と同方向に有意だった（比2.45倍、`p=5.4×10^-4`）が、CloseDrawerでは方向自体が逆転し有意で
なかった（比0.90倍、`p=0.052`）——単一アフォーダンスタスクへの一般化はこの指標についても
一様ではなかった。

### 2.1 目的

前回レポート§6は、PnPCounterToCabでの動的フィールドsteeringの反証（§4.5）について、「タスクが
複数の操作可能物体・複数の起こりうる操作を含む複雑なシーンであり、そこでの『どの物体をどう
扱うか』という選択情報の大部分を言語プロンプトが担っていた」という解釈を提示した。この解釈が
正しければ、環境が提示するアフォーダンスが一意なタスク（対象操作が実質1種類しかないタスク）
では、言語なしでも視覚と動的フィールドsteeringだけでタスクを完遂できるはずである。本フェーズは
この予測を、`collect_v2/`の元データで成功率が既に非常に高い（=タスクの物理的難度自体は低い）
`CoffeePressButton`（コーヒーマシンのボタンを押す、押す以外の操作の余地が乏しい）と
`CloseDrawer`（引き出しを閉める、閉める以外の操作の余地が乏しい）の2タスクで直接検定する。

### 2.2 手法

前回レポート§4.2で確立した`dynamic_vector_field_steering.py`のパイプライン（オンライン因果
状態推定器`CausalStateEstimator`、kNNフローライブラリ`DynamicFlowField`、動的steeringフック
`DynamicFieldHook`）を一切変更せず、`--task_name`のみ`CoffeePressButton`・`CloseDrawer`に
差し替えて再実行した。既に`--task_name`を引数化する設計だったため新規コードは不要である。
条件設計・`α=40.0`・`k_range=(0,1)`・N=8 episode/条件も前回のPnPCounterToCabと完全に同一とし、
直接比較可能にした（タスクごとの`α`個別チューニングは実施していない、§2.6参照）。

条件は前回と同じ6条件（C0実プロンプト基準、C1ダミー無介入、C2動的フィールド、C3静的steering、
C4ランダム方向、C5観測固定）。静的steeringベクトル`v_steer`は`steering_intervention.py`の
`compute_steering_vectors()`を用い、`collect/`（フェーズラベル付きデータ、CloseDrawer・
CoffeePressButtonについても既に用意されている）から計算した。

続けて`phase4_evaluation_metrics.py`を各タスクに適用し、動的因果効力（教師軌跡とのDTW/Fréchet
距離）・多様体逸脱度・感覚フィードバック従属性（C2 vs C5経路長比）を前回と同一の定義で計算した。

T1（`α=0` no-opチェック）は各タスクで実行前に確認した（P7準拠）。

### 2.3 結果

**T1（`α=0` no-op）確認**: 両タスクとも`max|Δaction|=0.00e+00`（P7準拠）。

**success_rate**（`α=40.0`、`N=8`/条件。比較のため前回レポートのPnPCounterToCab結果を併記）:

| 条件 | PnPCounterToCab（前回） | CoffeePressButton | CloseDrawer |
|---|---|---|---|
| C0_real_prompt_no_steer | 0.50 | **1.00** | **1.00** |
| C1_dummy_no_steer | 0.00 | 0.00 | 0.00 |
| C2_dummy_dynamic_field | 0.00 | 0.00 | 0.00 |
| C3_dummy_static_v_steer | 0.00 | 0.00 | 0.00 |
| C4_dummy_random_field | 0.00 | 0.00 | 0.00 |
| C5_dummy_dynamic_field_frozen_obs | 0.00 | 0.00 | **0.125**（1/8） |

CoffeePressButton・CloseDrawerともC0（実プロンプト基準）は`success_rate=1.00`——`collect_v2/`
元データの成功率（95〜100%、前回レポート§2.3表）と整合し、本フェーズの実装が正しく動作している
ことを追加で確認した。C1〜C4はいずれもn_steps上限（CoffeePressButton: 300、CloseDrawer: 500）
まで走り、一度もタスクを完遂しなかった。CloseDrawerのC5でのみ1エピソードが成功した
（`n_steps=477`、他7エピソードは全て上限500まで到達）。

**動的因果効力・多様体逸脱度**（`phase4_evaluation_metrics.py`、教師軌跡は各タスク
`collect_v2/`の成功エピソードから最大15本）:

| 条件 | CoffeePressButton: 最小DTW | 多様体逸脱度 | CloseDrawer: 最小DTW | 多様体逸脱度 |
|---|---|---|---|---|
| C0 | 0.022 | 3.482 | 0.014 | 4.474 |
| C1 | 0.121 | 4.248 | 0.277 | 5.351 |
| C2 | 0.115 | 4.092 | 0.281 | 5.086 |
| C3 | 0.107 | 4.146 | 0.267 | 5.479 |
| C4 | 0.090 | 4.164 | 0.278 | 5.100 |
| C5 | 0.084 | 3.370 | 0.264 | 3.856 |

C2 vs 他ダミー条件のMann-Whitney検定（片側、C2が下回るか）: CoffeePressButton — DTW: vs C1
`p=0.520`、vs C3 `p=0.747`、vs C4 `p=0.903`；多様体逸脱度: vs C1 `p=0.221`、vs C3 `p=0.439`、
vs C4 `p=0.561`。CloseDrawer — DTW: vs C1 `p=0.520`、vs C3 `p=0.601`、vs C4 `p=0.561`；
多様体逸脱度: vs C1 `p=0.287`、vs C3 `p=0.117`、vs C4 `p=0.520`。**いずれのタスクも有意水準に
届かない**（前回のPnPCounterToCabと同じパターン）。

**感覚フィードバックへの従属性**（C2 vs C5、経路長）:

| タスク | 経路長比（C2/C5） | Mann-Whitney p（C2>C5） | action標準偏差（C2 / C5） |
|---|---|---|---|
| PnPCounterToCab（前回） | 4.197 | 7.77×10⁻⁵ | 0.128 / 0.011 |
| CoffeePressButton | 2.448 | 5.4×10⁻⁴ | 0.100 / 0.012 |
| CloseDrawer | **0.903** | **0.052（非有意）** | 0.063 / 0.012 |

CoffeePressButtonは前回のPnPCounterToCabと同方向（C2の方が有意に経路が長い）だが、比率は
やや小さい（2.45倍 vs 4.20倍）。**CloseDrawerのみ方向が逆転した**——C5（観測固定）の平均経路長
（1.68）がC2（通常観測、1.52）をわずかに上回り、有意水準にも届かない（`p=0.052`）。ただし
action標準偏差（行動そのものの時間的分散）はCloseDrawerでもC5が1桁小さく（0.012 vs 0.063）、
前回・CoffeePressButtonと同方向を保っている——経路長という指標だけがCloseDrawerで異なる挙動を
示した。C5の個別episode経路長を見ると、8episode中2episodeが極端に大きい値（4.81・5.99）を
示し、残り6episodeは前回・CoffeePressButtonのC5同様に小さい値（0.18〜0.42）にとどまっていた
——経路長の平均が少数の外れ値に支配されたことが、方向逆転の主因である可能性が高い（§2.4）。

### 2.4 解釈（都合の良い部分だけを取り上げない）

- **中心仮説「操作対象が一意な単純タスクなら言語なしでも動的フィールドsteeringでタスクが
  自律的に完遂できる」は、テストした2タスクいずれでも支持されなかった**。CoffeePressButton
  （ボタンを押すだけ）・CloseDrawer（引き出しを閉めるだけ）は`collect_v2/`元データで成功率
  95〜100%と、PnPCounterToCabよりも物理的難度が明らかに低く、かつ「押す」「閉める」以外の
  操作の余地がほぼない一意なアフォーダンスを持つ。それでもなお、ダミープロンプト下では
  steeringの種類（動的・静的・ランダム）によらず一貫して`success_rate=0.00`だった。これは
  前回レポート§6が提示した「複数の操作可能物体を含む複雑なシーンだから失敗した」という解釈
  だけでは、この否定的結果を十分に説明できないことを意味する——**言語プロンプトが担っている
  情報は、単に「どの物体を選ぶか」という選択情報だけではなく、より根本的にタスクの発火条件
  そのものである可能性が高い**。動的フィールドsteeringが提供する情報（成功エピソードの局所的な
  "次に何が起きたか"の統計）だけでは、たとえ選択の余地がない状況でも、行動系列を「タスク完遂」
  という目標へ向けて開始・維持するには不十分だったということである。
- **CloseDrawerのC5で観測された1/8の成功は、"言語なしでの視覚駆動タスク遂行"の反例にはならない**
  ——この成功はC2（通常observation、ダミープロンプト、動的steering）ではなく、C5（observationを
  call0に固定、ダミープロンプト、動的steering）という統制条件で生じた。フェーズ4評価
  （§2.3のCloseDrawer C5 action標準偏差=0.012、他条件の1/5〜1/7）が示す通り、C5の行動は
  ほぼ一定値に近い出力へ収束している。CloseDrawerが引き出しを閉めるタスクであることを踏まえると、
  ロボットが「引き出しへ向かって押す」に近い一定の運動を500ステップ持続すれば、たとえ感覚
  フィードバックが更新されなくても機械的に引き出しが閉まる、という偶発的な成功が原理的に
  あり得る（純粋にopen-loopな反復運動でも達成できるほど単純なタスクである可能性）。この解釈は
  本検証だけでは確証できないが、少なくとも「言語なしでの意図的なタスク遂行の成功」と解釈する
  根拠はない。
- **感覚フィードバック従属性の指標（C2 vs C5経路長比）がCloseDrawerでのみ方向逆転したことは、
  前回レポート・CoffeePressButtonの結果を安易に一般化できないことを示す具体例である**。
  action標準偏差（行動の時間的分散そのもの）は3タスク全てで一貫してC5が1桁小さく、「観測を
  固定すると行動の多様性が失われる」という中心的な主張自体は崩れていない。しかし経路長
  （エンドエフェクタが移動した総距離）という指標は、少数の外れ値episodeに支配されやすく
  （CloseDrawerのC5で8episode中2episodeが平均を押し上げた）、タスク・タスク難度依存で
  頑健性が変わりうる——「動的介入下の行動が感覚運動ループとして機能し続けている」という前回の
  結論を支持する指標として経路長を使う際は、この頑健性の限界を明記すべきである。

### 2.5 結論

**反証**。単一アフォーダンスタスク（CoffeePressButton・CloseDrawer）でも、ダミープロンプト下の
動的フィールドsteeringは静的steering・ランダム方向介入と区別できず、いずれもタスク完遂を
（統制条件での1例の偶発的成功を除き）誘発しなかった。前回レポートが提示した「PnPCounterToCabの
失敗は複雑なシーンでの選択情報の欠如が原因」という解釈は、この2タスクでは支持されない——言語が
担う情報は選択情報だけに還元できない可能性が高い。感覚フィードバック従属性の指標は
タスク依存で頑健性が変わり、単純な一般化はできない。

### 2.6 注意点・限界

1. `α=40.0`はPnPCounterToCabで予備的に選定された値をそのまま流用しており、CoffeePressButton・
   CloseDrawerに対して個別にdose-responseチューニングしていない。
2. `N=8`/条件は前回と同じく、二値指標としての検出力は小さい。
3. CoffeePressButton・CloseDrawerは元データで失敗episodeが極端に少ない（39/1・40/0、
   `collect_v2/`）ため、フェーズ1のD空間構築時点で「進行度逸脱検定」自体が実施できていない
   （前回レポート§2.3表内「検定不可」）——本フェーズのD空間エンコーダ・フローライブラリは
   その制約を継承したまま構築されたものである。

### 2.7 今後の検証案

- タスクごとの`α`個別dose-responseスイープ。
- 他の単一アフォーダンスタスク（OpenSingleDoor等）への拡張。

---

## 3. フェーズ6：アトラクタへの「意図の引き込み」（Entrainment）検証

### 3.0 要旨

PnPCounterToCabで、エピソード最初の5 callだけ実プロンプトを与え以降ダミープロンプトに切り替える
条件（C_fadeout_only）は、期待通りsuccess_rate=0.00（8episode全て500ステップの上限まで到達）
となり、言語の持続的供給を打ち切ると運動がタスク完遂に至らなくなることを確認した。**しかし、
同じ切替タイミングの直後から動的フィールドsteeringを起動した条件（C_fadeout_dynamic）も
success_rate=0.00で、C_fadeout_onlyと統計的に区別できなかった**（経路長: `p=0.117`、教師との
最小DTW距離: `p=0.287`、いずれも非有意）。中心仮説「言語は初期のアトラクタ選択のトリガーに
過ぎず、その後の動的フィールドが自律的に行動を引き継ぐ」は、本パラメータ設定
（`fadeout_calls=5`、`α=40.0`）では**反証された**。基準条件C_full（実プロンプト持続、
steeringなし）はsuccess_rate=0.625で、前回レポートのC0（0.50）と同オーダーであり、本フェーズの
実装が既存基盤と整合していることを追加確認した。

### 3.1 目的

前回レポート§4.4は、言語プロンプトが伝える情報が「どの物体をどうするか」という選択情報である
可能性を指摘したが、これは「言語は行動系列全体を通じて必要とされる持続的な表象なのか、それとも
エピソード冒頭の一過性のトリガーに過ぎないのか」という、より根本的な問いには答えていない。
本フェーズは、メルロ=ポンティ的な運動的志向性の枠組みに沿って後者の仮説——**言語は力学系を
特定のアトラクタ盆（初期状態）へ引き込む(entrainment)ためのトリガーであり、一度引き込まれれば
その後の行動生成は言語的表象に依らず感覚運動の結合動態が自律的に引き受ける**——を、エピソード
内でのプロンプト動的切り替え（fadeout）実験により検定する。これは前回のsuccess_rateレベルの
検定（§4）を「言語のON/OFF」の二値ではなく「言語がいつ・どれだけ必要か」という時間構造へ
分解する検証であり、本レポートで唯一の**因果**検証である（フェーズ5・7は前回同様の観察的・
構成的検証）。

### 3.2 手法

新規スクリプト`prompt_fadeout_entrainment.py`を実装した。`dynamic_vector_field_steering.py`の
オンライン因果推定器・フローライブラリ・steeringフック（`CausalStateEstimator`・
`DynamicFlowField`・`DynamicFieldHook`・`FinalFeatCapture`）をそのまま再利用し、「どのcallで
どちらのプロンプトを使うか」という制御ロジックのみを新規実装した。

タスクはPnPCounterToCab（前回§4と同一、言語の役割が最も顕著だったタスク）。3条件:

| 条件 | プロンプトスケジュール | steering |
|---|---|---|
| C_full | エピソード完了まで実プロンプト | なし（基準） |
| C_fadeout_only | 最初のfadeout_calls call は実プロンプト、以降ダミープロンプトに切替 | なし |
| C_fadeout_dynamic | C_fadeout_onlyと同じ切替タイミング | ダミープロンプトに切り替わった**同じcall**から動的フィールドsteering ON |

`fadeout_calls=5`（call粒度。1 call = `num_open_loop_steps=16` env-stepに相当する行動チャンクの
適用単位であり、design.mdの「最初の5〜10ステップ、対象物へ向かい始める初期フェーズ」における
「ステップ」はこの粒度と解釈した——個々のenv-stepでは物理的にほぼ何も進行しないため。設計書が
示す範囲5〜10の下限を採用し、網羅的なスイープは実施していない、§3.6参照）。

**因果ロジックの詳細**: `dynamic_vector_field_steering.py`と同じ1-call遅延構造を踏襲する。
call `t`でその時点のBlk-13特徴を捕捉し状態推定器を更新した直後、`(t+1) >= fadeout_calls`かつ
`C_fadeout_dynamic`であれば、次のcallで使うsteeringベクトルを`field.query()`でセットする。
プロンプト自体は各callの推論直前に`call_idx < fadeout_calls`かどうかで即座に切り替える（1-call
遅延はsteeringベクトル計算のみに存在し、プロンプト切替そのものには遅延がない）。

実プロンプトのT5埋め込みは`env.get_ep_meta().get("lang", task_name)`から得られるタスク記述文字列
をキーに、事前ロード済みの`robocasa_t5_embeddings.pkl`キャッシュから取得される（RoboCasaの言語
記述は収集・評価時に使われる標準文一式であり、既にキャッシュ済みのため、追加でT5-11bエンコーダを
ロードする必要はない——前回レポート§4.2.7のダミープロンプト埋め込みと同じキャッシュ機構を利用）。

T1（`α=0` no-opチェック）は実行前に確認した（P6/P7準拠）。

**検証の狙い**: C_fadeout_onlyで運動が崩壊する（success_rateが0または軌跡が破綻する）にも
かかわらず、C_fadeout_dynamicでタスク完遂（またはsuccess_rateの有意な上昇）が見られれば、
「言語は初期のアトラクタ選択にのみ必要で、その後の行動生成は言語的表象に依らない」という仮説の
具体的な支持になる。反対にC_fadeout_dynamicもC_fadeout_onlyと同程度にしか成功しなければ、この
仮説は本タスク・本パラメータ範囲では支持されない。

### 3.3 結果

**T1（`α=0` no-op）確認**: `max|Δaction|=0.00e+00`。

| 条件 | success_rate (n=8) | n_steps（8episode） | 経路長(mean) | action標準偏差(mean) | 教師との最小DTW(mean) | 教師との最小Fréchet(mean) | 多様体逸脱度(mean) |
|---|---|---|---|---|---|---|---|
| C_full | **0.625** | [248,500,223,247,493,500,500,217] | 2.374 | 0.168 | 0.074 | 0.291 | 4.812 |
| C_fadeout_only | 0.00 | 全て500 | 1.782 | 0.106 | 0.122 | 0.507 | 4.770 |
| C_fadeout_dynamic | 0.00 | 全て500 | 2.081 | 0.103 | 0.114 | 0.474 | 4.850 |

C_fadeout_dynamicのフックの実効注入量は最大23.9〜27.6（前回のC2と同オーダー、フックは正しく
起動していた）。C_full以外の全16episodeが最大ステップ数（500）まで到達し、一度もタスクを
完遂しなかった。

**C_fadeout_dynamic vs C_fadeout_only のMann-Whitney検定**（片側）: 経路長（dynamicが上回るか）
`p=0.117`、教師との最小DTW距離（dynamicが下回るか）`p=0.287`。**いずれも有意水準に届かない**。
経路長は数値上わずかにC_fadeout_dynamicの方が長い（2.081 vs 1.782、+17%）が、action標準偏差は
両条件でほぼ同じ（0.103 vs 0.106）。

### 3.4 解釈（都合の良い部分だけを取り上げない）

- **「言語は初期のアトラクタ選択のトリガーに過ぎない」という中心仮説は、本パラメータ設定では
  明確に反証された**。C_fadeout_only（言語打ち切り後は無介入）とC_fadeout_dynamic（言語打ち切り
  直後から動的フィールドsteering起動）は、success_rate（ともに0.00）でも、より連続的な指標
  （経路長・教師とのDTW距離・多様体逸脱度）でも統計的に区別できなかった。これはフェーズ3
  （前回レポート§4）の「ダミープロンプト単体が既に床効果に達しており動的介入の効果を検出できない」
  という限界とは異なる——本フェーズのC_fadeout_onlyは「言語なし」ではなく「言語で5 call分
  引き込まれた後に打ち切られた」状態であり、design.mdの仮説が正しければここには既に何らかの
  タスク方向への「引き込み」が生じているはずである。それにもかかわらず動的フィールドが
  それを検出可能な形で維持・発展させられなかったということは、動的フィールドsteering自体
  （kNNフローライブラリ+局所接平面射影+弱いエネルギー勾配）の情報量が、5 call分の言語露出で
  生じた初期状態を目標達成まで導くには不十分であることを示唆する。
- **経路長のわずかな増加（+17%、非有意）は、前回レポート§5.4の「感覚フィードバックへの
  従属性」という肯定的知見と整合する方向ではある**——C_fadeout_dynamicの行動はC_fadeout_only
  よりわずかに活発だった。しかしこれは統計的に有意な水準には遠く、「動的フィールドが行動を
  多少なりとも動かし続けている」以上の主張はできない。前回レポート§5で確認された「感覚運動
  ループとしては機能し続けている」という結論自体は本フェーズの範囲でも矛盾しないが、それが
  タスク完遂に近づく方向への機能かどうかは、この結果からは支持されない。
- **`fadeout_calls=5`という値そのものが「引き込みに十分な時間」ではなかった可能性は残る**——
  本フェーズは単一の切替タイミングのみを検証しており、より長い言語露出（例: 10〜20 call）で
  引き込みが起きるかどうかは未検定である（§3.7）。ただしC_fullの成功8episode中5episodeが
  n_steps=217〜248（`num_open_loop_steps=16`換算で約14〜16 call相当）という比較的早い段階で
  完了していたことを踏まえると、fadeout_calls=5はこれら比較的速い成功事例の「接近フェーズ」の
  途中で言語を打ち切っていた可能性が高く、design.mdが想定する「対象物へ向かい始める初期フェーズ」
  を過ぎる前に言語を除去してしまっていた懸念がある。

### 3.5 結論

**反証**。テストした範囲（PnPCounterToCab、`fadeout_calls=5`、`α=40.0`）では、言語プロンプトを
初期の5 callのみ与えその後動的フィールドsteeringに切り替える条件は、steeringなしで言語を
打ち切る条件と、success_rate・経路長・教師との軌跡類似度のいずれでも統計的に区別できなかった。
「言語は初期のアトラクタ選択のトリガーに過ぎず、その後は感覚運動の結合動態が自律的に行動を
引き継ぐ」という仮説は、本パラメータ設定では支持されない。ただし単一の切替タイミングのみの検定
であり、より長い言語露出での再検定は行っていない（§3.7）。

### 3.6 注意点・限界

1. `fadeout_calls=5`は単一値のみを検証しており、切替タイミングのスイープ（design.mdが例示する
   5〜10の範囲内での網羅比較）は実施していない。
2. 単一タスク（PnPCounterToCab）・単一`α`（40.0、前回と同一値を流用）でのみ検証した。
3. `N=8`/条件、前回と同じ検出力の限界を持つ。

### 3.7 今後の検証案

- `fadeout_calls`のスイープ（例: 3, 5, 8, 12）による「引き込みに必要な最小言語曝露時間」の推定。
- 他タスクへの一般化。
- C_fadeout_onlyの崩壊過程そのものの定量化（軌跡がどの時点でどう破綻するか）。

---

## 4. フェーズ7：D空間の「真の生成行動」へのアップグレード

### 4.0 要旨

新規収集した行動チャンク付きデータ（`collect_actions/`、4タスク×2 seed系列×20 episode、
`collect_v2/`と同一の収集方法論、成功率は各タスクとも±5pt以内で`collect_v2/`と一致——独立な
ロールアウトとして統計的性質が同等であることを確認済み）を使い、`Delta_eef`/`Delta_grip`を
実際の生成行動チャンクのPCA特徴に置き換えたD空間を再構築した。**交絡を避けるため、
「新しいサンプル」の効果と「新しい表現（結合ベクトルの定義）」の効果を分離する3群比較を行った**
——(1)前回レポートの`collect_v2/`上のeef/gripper代理指標版D空間、(2)本フェーズの新規
`collect_actions/`上で(1)と全く同じeef/gripper代理指標の定義を再現した「サンプルのみ新しい」
D空間、(3)`collect_actions/`上で行動チャンクPCAに置き換えた「サンプルも表現も新しい」D空間。

**主な結果**: PnPCounterToCabで、§3.3のDMD再評価（成功=自己安定化・失敗=不安定化パターン）は
(1)`p=0.0228`→(2)`p=0.0092`→(3)`p=1.33×10⁻⁵`と、サンプルを固定した(2)→(3)の比較でも
約690倍p値が縮小し、真に生成行動を用いた表現への置き換えがこの力学的シグナルを強めることが
確認された。TurnOnStoveでは(2)と(3)のDMD検定p値が偶然一致する（`p=6.12×10⁻⁸`、
Mann-Whitney検定がランクのみに依存するため、絶対値は異なるがepisode順位が偶然一致した）という
興味深い現象が観察されたが、いずれにせよ両空間とも極めて有意という結論自体は変わらない。
一方、「|λ|>1到達率」という二値化した指標でのFisher検定はPnPCounterToCabで(2)`p=0.0448`から
(3)`p=0.219`（非有意）へと**悪化**した——連続量（`max|λ|`そのもの）での分離は強まったが、
1.0という単一閾値での二値分離はむしろ弱まるという、指標選択に依存した非自明な結果である。
CloseDrawer・CoffeePressButtonは新規収集でも失敗episode不足（0件・1件）のため、いずれの
D空間でも検定不可のままだった。

### 4.1 目的

前回レポート§2.6・§9は、フェーズ1の結合ベクトル`c_t`が「行動が環境に及ぼした効果」
（eef_pos/gripper_qposのcall間差分）を代理指標として使っており、モデルが実際に生成した行動
チャンク`X_hat_0`（32 timesteps×7次元）そのものは未捕捉だったことを開示していた。これは
design.mdが要求する「感覚と運動の結合動態を一次的対象として扱う」という要求に対し、「運動」の
側を厳密には満たしていない——「行動の結果（環境からのフィードバック）」と「運動指令そのもの
（モデルの出力）」は現象学的に異なる対象である。本フェーズはこの代理指標を実際の生成行動チャンク
に置き換えたD空間を新規に構築し、前回レポート§3.3で発見した最も新規性の高い知見——D空間で
PnPCounterToCabの「成功=自己安定化・失敗=不安定化」というDMD力学シグナルが統計的に有意になった
こと（`X_p`空間: `p=0.38〜0.59`、非有意 → D空間(eef/gripper代理指標版): `p=0.023`）——が、
より設計書に忠実な結合表現でも再現される、あるいはさらに強まるかを検証する。

### 4.2 手法

**4.2.1 新規データ収集の必要性**。`collect/`・`collect_v2/`はいずれも生成行動チャンクを保存して
いない。新規スクリプト`collect_action_chunks.py`を実装し、`attractor/collect_multitask.py`と
完全に同一の収集方法論（4タスク×2 seed系列×20 episode、call毎seed規約P1、Blk特徴・物理量の
捕捉）を踏襲した上で、各callの`get_action()`が返す生成済み行動チャンク`result["actions"]`
（形状`(32, 7)`、モデルが実際に出力した7次元アーム+グリッパー行動——env適用時にのみ12次元へ
ゼロパディングされる、そのパディング前の生の出力）を追加保存した。それ以外（タスク集合・seed系列
・エピソード数・プローブ層・収集方法論）を変える意図はなく、新規収集を行った唯一の理由は
「actionsフィールドを含むnpzが存在しなかったから」である。

**4.2.2 新しい結合ベクトルの構成**。design.md（ldv_design_v2.md）§フェーズ7実装指示3の定義
に従い、

```
c_t = [ Xp_t (10dim, §5.6/§5.7と同一の進行多様体),
        V_t = Xp_t - Xp_{t-1} (10dim, backward difference),
        A_PCA_t (5dim, そのcallの生成行動チャンクをタスク内でPCA圧縮したもの) ]  ∈ R^25
```

とする。`A_PCA_t`は前回の`Delta_eef_t`・`Delta_grip_t`（call間差分、「行動の効果」）を完全に
置き換える——design.mdの指示通り両者を混在させることはしなかった（混在させると「効果」と
「指令」のどちらが検定対象か不明瞭になるため）。`A_PCA_t`はcall間差分ではなく、そのcall単体の
生成内容そのものである点が`Delta_eef`等と本質的に異なる。行動チャンク（32×7=224次元、call単位で
フラット化）は標準化後、タスク内（2 seed系列プール）でPCA(上位5成分)に圧縮した。

**4.2.3 それ以外の構成要素はフェーズ1と同一**。`Xp`（進行多様体、scene残差化後PCA(10)）・
遅延座標埋め込み（backward窓τ=3、線形PCA、D_dim≤8）・標準化してから埋め込みPCAへ入力する
（§2バグ#1の教訓を継承）という設計は全てフェーズ1（`dynamics_embedding_test.py`）から変更して
いない。本スクリプトはオフライン専用の事後解析であり、フェーズ3のようなオンライン再利用の予定が
ないため、`V_t`をforward differenceに戻す選択肢もあったが、フェーズ1との直接比較を優先し
backward差分のまま統一した。

**4.2.4 §3.3 DMD再評価**。`dmd_jacobian_stability_test.py`の`windowed_dmd`（窓幅8、§5.7・
前回§3.3と同一設定）をこの新しいD空間の軌跡に再適用する。これは軌跡上を窓幅8callでスライドさせ、
各窓内の局所的な線形力学系近似（Dynamic Mode Decomposition、`D_{t+1} ≈ A D_t`という線形写像
`A`を窓内の点対から最小二乗推定し、その固有値`λ`を求める）を行うものである——`|λ_max|`
（推定された`A`の最大絶対固有値）は、その窓の近傍でわずかな摂動がどれだけ急速に拡大・縮小
するかを表す局所的リアプノフ指数の近似であり、`|λ_max|<1`なら局所的に軌跡は自己収束的（外乱を
受けても元の軌道へ戻る）、`|λ_max|>1`なら局所的に発散的（外乱が拡大する）と解釈できる。
「成功エピソードは自己収束的な窓が多く、失敗エピソードは発散的な窓が多い」という仮説を、
エピソード内`max|λ|`（そのエピソード中で最も不安定だった瞬間）のMann-Whitney検定（成功<失敗、
分布形状を仮定しない順位検定）と、`|λ|>1`という閾値を超えたエピソードの割合のFisher正確検定
（2×2分割表の閾値超過有無を比較する、小標本でも正確なp値が得られる検定）の両方で評価する——
前者は連続量としての不安定性の"程度"を、後者は"閾値を超えるか否か"という二値化した情報を、
それぞれ独立に検定するものであり、両者が異なる結論を示す場合はその乖離自体が情報を持つ
（§4.4）。前回同様、成功エピソードのみを用いた進行度逸脱検定（§2.3相当）もこの新D空間で
再実行し、退化していないかを確認した。

**4.2.5 交絡分離のための3群比較設計**。本フェーズの新規収集データ（`collect_actions/`）は
`collect_v2/`とは独立なロールアウト（同一モデル・同一seedだが別実行）であるため、前回レポート
§3.3の数値（`collect_v2/`上のeef/gripper代理指標版）と単純に比較すると、「表現を変えたことに
よる差」と「たまたまサンプルが変わったことによる差」が交絡する。この交絡を分離するため、
`dynamics_embedding_test.py`・`energy_field_test.py`を（一切変更せず）`collect_actions/`に
そのまま適用し、「新しいサンプル・旧eef/gripper代理指標」というちょうど中間の比較群を追加した。
すなわち3群——(1)`collect_v2/`×eef/gripper代理指標（前回レポートの数値そのもの）、
(2)`collect_actions/`×eef/gripper代理指標（本フェーズが追加した中間比較）、(3)`collect_actions/`
×行動チャンクPCA（本フェーズの主対象）——を比較することで、(1)→(2)がサンプルの効果、
(2)→(3)が表現の効果にそれぞれ対応する。

### 4.3 結果

**新規収集の品質確認**（`collect_actions/`、`collect_v2/`との比較）:

| タスク×seed | collect_v2 success_rate | collect_actions success_rate |
|---|---|---|
| PnPCounterToCab/195 | 45.0% | 50.0% |
| PnPCounterToCab/196 | 45.0% | 45.0% |
| CloseDrawer/195 | 100.0% | 100.0% |
| CloseDrawer/196 | 100.0% | 100.0% |
| TurnOnStove/195 | 35.0% | 40.0% |
| TurnOnStove/196 | 40.0% | 40.0% |
| CoffeePressButton/195 | 100.0% | 100.0% |
| CoffeePressButton/196 | 95.0% | 95.0% |

8ファイル中6ファイルは`collect_v2/`とビット単位ではなく成功率レベルで一致し、残り2ファイル
（PnP/195, TurnOnStove/195）も±5ptに収まった——新規ロールアウトが既存データと統計的に
同等の性質を持つことを確認した。

**行動PCA・エンコーダの説明分散比**（タスク内2 seed系列プール）: 行動チャンクPCA（上位5成分）は
CloseDrawer 0.847、CoffeePressButton 0.835、TurnOnStove 0.720、PnPCounterToCab 0.651
の分散を説明した。D空間エンコーダ（tau=3窓の線形PCA、D_dim≤8）の説明分散比は0.51〜0.75。

**3群比較（PnPCounterToCab, TurnOnStove——CloseDrawer/CoffeePressButtonは以下いずれの
D空間でも失敗episode不足のため検定不可、n_fail=0/1で不変）**:

| 検定 | タスク | (1) collect_v2×eef/grip（前回） | (2) collect_actions×eef/grip（中間比較） | (3) collect_actions×行動チャンクPCA（本フェーズ） |
|---|---|---|---|---|
| 進行度逸脱 p(succ<fail) | PnP | 3.91×10⁻⁸ | 3.50×10⁻⁸ | 3.50×10⁻⁸ |
| 逸脱マージン(正規化) | PnP | 6.86 | 7.75 | **9.20** |
| DMD max\|λ\| MW p(succ<fail) | PnP | 0.0228 | 0.0092 | **1.33×10⁻⁵** |
| \|λ\|>1到達率(succ/fail) | PnP | 13/18, 22/22 | 12/19, 19/21 | 17/19, 21/21 |
| Fisher p(到達率succ<fail) | PnP | 0.0130 | 0.0448 | 0.219（非有意） |
| 進行度逸脱 p(succ<fail) | TurnOnStove | 8.63×10⁻⁸ | 1.87×10⁻⁶ | 1.87×10⁻⁶ |
| 逸脱マージン(正規化) | TurnOnStove | 5.82 | −2.42（非完全分離） | −2.30（非完全分離） |
| DMD max\|λ\| MW p(succ<fail) | TurnOnStove | 8.55×10⁻⁸ | 6.12×10⁻⁸ | 6.12×10⁻⁸（(2)と偶然一致、後述） |
| \|λ\|>1到達率(succ/fail) | TurnOnStove | 0/15, 25/25 | 0/16, 24/24 | 0/16, 22/24 |
| Fisher p(到達率succ<fail) | TurnOnStove | 2.49×10⁻¹¹ | 1.59×10⁻¹¹ | 2.43×10⁻⁹ |

（(1)はn=18succ/22fail、(2)(3)はn=19succ/21fail(PnP)・16succ/24fail(TurnOnStove)——新規
ロールアウトのため成功/失敗の内訳自体が(1)とわずかに異なる。窓幅8・階数上限4は全群で共通。）

**TurnOnStoveのDMD p値が(2)(3)間で完全一致した点について**: Mann-Whitney U検定は値の絶対量
ではなく順位のみに依存する統計量である。(2)(3)で\|λ\|>1到達率が24/24 vs 22/24とわずかに異なる
（絶対値は異なる計算がなされている）にもかかわらずp値が一致したのは、「各episodeのmax\|λ\|の
episode間順位付け」が両空間で偶然完全に一致したためである——c_tの共有成分（Xp,V）がD空間の
主要な変動を支配しており、追加ブロック（eef/grip代理指標 vs 行動PCA）の違いが順位を変えるほどの
影響を持たなかったことを示唆する（§4.4）。

### 4.4 解釈（都合の良い部分だけを取り上げない）

- **PnPCounterToCabにおいて、サンプルを固定した上での表現の置き換え（(2)→(3)）が、DMD力学
  シグナルを約690倍（p値換算）強めた**——これは前回レポート§3.4が最も新規性の高い知見として
  報告した「D空間でPnPCounterToCabの力学的不安定性シグナルが顕在化する」という結果が、
  代理指標（環境への効果）ではなく真の生成行動（運動指令そのもの）を使うことでさらに強化される
  ことを示す、design.mdの中心仮説——「観測と行動を分離すると本来の構造が見えなくなる」——を
  支持する具体的な追加証拠である。逸脱マージンも6.86→7.75→9.20と単調に改善しており、サンプル
  効果・表現効果の両方が同方向に寄与している。
- **ただし「行動チャンクPCAの方が常に優れている」と単純化はできない**——|λ|>1到達率という
  二値指標でのFisher検定は(2)`p=0.0448`（有意）から(3)`p=0.219`（非有意）へ悪化した。これは
  成功episodeの中でも|λ|>1へ到達するものが増えた（12/19→17/19）ためであり、「行動チャンクPCA
  表現では、成功episodeでもより頻繁に瞬間的な不安定性の兆候(|λ|>1)が生じるが、その不安定性の
  "程度"(連続量としてのmax|λ|)は失敗episodeほど極端ではない」という、より繊細な描像を示唆する
  ——**"不安定性が生じるかどうか"と"どれだけ不安定か"は異なる情報を運んでおり、指標選択
  （二値閾値 vs 連続量ランク検定）によって同じデータから逆方向の結論を引き出しうる**という
  一般的な教訓でもある。本レポートは連続量のMann-Whitney検定を主指標として採用しているため
  （§4.2.4）、全体としての結論は「強化された」側を取るが、この非一貫性自体を注意点として
  明記する。
- **TurnOnStoveでのDMD p値の完全一致（(2)(3)間）は、偶然の一致でありながら情報を持つ**——
  Mann-Whitney検定はランク（順序）のみに依存する統計量であり、値そのものの分布が変わっても
  episode間の相対順位が保たれれば同じp値が出る。この一致は、TurnOnStoveタスクにおいては
  「どのepisodeが力学的に不安定か」という順位付けそのものが、結合ベクトルに何を追加するか
  （eef/grip代理指標 vs 行動PCA）に対してロバストであることを意味する——おそらくTurnOnStove
  （ノブを回すという単純な回転運動）は、Xp・V（潜在表現の進行と速度）だけで力学的な安定性・
  不安定性のほぼ全てが決まっており、行動の付加情報がその上に何を足しても順位を変えないほど
  支配的な信号になっているためと考えられる。これは前回レポート§3.4の「PnPCounterToCabと
  TurnOnStoveでは結合表現の恩恵が非対称」という知見と整合する——TurnOnStoveは既に(1)の時点で
  両空間とも有意だった「恩恵を受けにくい」タスクである。
- **CloseDrawer・CoffeePressButtonの検定不可は表現を変えても解消されない**——これはこれらの
  タスクの物理的難度が本質的に低く失敗episodeがほとんど生じないという、データ側の制約であり、
  D空間の構築方法をどう変えても解決できない種類の限界である。

### 4.5 結論

**部分的に支持**。真の生成行動チャンクを用いたD空間は、サンプルを固定した厳密な比較において、
PnPCounterToCabの「成功=自己安定化・失敗=不安定化」というDMD力学シグナルを連続量の検定で
大幅に（約690倍のp値縮小）強化した——design.mdが期待した「真の運動指令を使うことでより明確な
構造が見える」という仮説を、この1タスク・この指標について支持する。ただし二値化した指標
（|λ|>1到達率のFisher検定）では逆に非有意化しており、指標選択に依存した非一貫性がある。
TurnOnStoveでは表現の変更による実質的な差は見られなかった（既に両表現で極めて有意であり、
「恩恵を受けにくい」タスクであることが前回レポートと整合する形で再確認された）。
CloseDrawer・CoffeePressButtonの検定不可は表現に依らずデータ側の制約として残った。

### 4.6 注意点・限界

1. `collect_action_chunks.py`は`collect_v2/`と独立な新規ロールアウトである。この交絡は§4.2.5の
   3群比較（`collect_actions/`上でeef/gripper代理指標版D空間も再構築）により分離を試みたが、
   それでも各群の成功/失敗episodeの内訳自体は完全には一致しない（例: PnP `n_fail`=22 vs 21）
   ——サンプルサイズが小さいため、この程度の内訳変化がp値に与える影響を完全には排除できない。
2. `A_PCA_t`の次元数(5)は設計書の「上位3〜5成分」の上限を採用したのみで、感度分析は未実施。
3. 視覚特徴の勾配・非線形自己符号化器など、前回レポート§2.6が開示した他の限界はフェーズ7でも
   未解消のままである。
4. |λ|>1到達率（Fisher検定）とmax|λ|そのもの（Mann-Whitney検定）とで結論の方向が食い違う
   （§4.4）——本レポートは連続量検定を主指標として採用したが、この選択自体が結論の一部を
   左右している点は注意点として明記する。

### 4.7 今後の検証案

- `A_PCA_t`の次元数を3〜5でスイープし、DMD有意性の頑健性を確認する。
- このD空間上でフェーズ2のエネルギー場・フェーズ3のオンラインsteeringを再構築する
  （本フェーズはフェーズ1・2のDMD再評価部分のみを再実行し、フェーズ3・4への展開は範囲外とした）。

---

## 5. 統合結論

本レポートが実行した3フェーズは、前回レポートの結論を出発点に、3つの独立した仮説を検定した。
結果を俯瞰すると、**「感覚運動の結合を表現として埋め込むこと」に関する仮説は一貫して支持され
強化された一方、「言語なしの因果的介入でタスクを駆動できる」という仮説は、条件を変えて
再検定するたびに繰り返し反証された**という、明確なパターンが浮かび上がる。

**支持・強化された側（表現の仮説）**: フェーズ7は、D空間の結合ベクトルを環境への効果の代理指標
（eef/gripper delta）から真の生成行動（行動チャンクのPCA特徴）に置き換えることで、
PnPCounterToCabの「成功=自己安定化・失敗=不安定化」というDMD力学シグナルが、サンプルを固定した
厳密な比較でも約690倍（p値換算）強化されることを示した。これは前回レポート§3.4が発見した
「観測と行動を分離すると本来の構造が見えなくなる」という知見の直接的な追加証拠であり、
design.mdが要求する「感覚運動の結合動態を一次的対象として扱う」というアプローチの妥当性を、
より設計書に忠実な実装のもとでも再確認するものである。ただし二値化した指標では逆の結論が
出ており（§4.4）、この強化は指標選択に依存する部分がある点は明記しておく。

**反証が繰り返された側（因果的介入の仮説）**: フェーズ5は、前回レポートが提示した「PnP
CounterToCabの失敗は複雑なシーンでの選択情報の欠如が原因」という解釈を、操作対象が一意な単純
タスク（CoffeePressButton・CloseDrawer）で直接検定したが、支持されなかった——どちらのタスクも
ダミープロンプト下では動的フィールドsteeringを含むいかなる介入でもタスクを完遂できなかった。
フェーズ6は、言語を「持続的な表象」ではなく「初期のアトラクタ選択トリガー」として再定位する
仮説を、プロンプト動的切り替え実験で検定したが、これも支持されなかった——5 call分の言語露出後に
動的フィールドsteeringへ切り替えても、無介入で言語を打ち切った場合と統計的に区別できなかった。
これら2つの反証は、前回レポート§4.5の反証（ダミープロンプト下での動的介入は静的介入・ランダム
介入と区別できない）を、(a)タスクの性質（単一アフォーダンス）、(b)言語提示のタイミング
（部分的露出+切替）という異なる2つの軸に沿って一般化したものであり、いずれの軸でも当初の
反証が覆らなかったことになる。

**両者を合わせた解釈**: 本検証群が繰り返し示しているのは、Cosmos Policyの潜在空間には
「感覚運動が結合した、低次元で力学的に意味のある構造」が確かに存在し、その構造は表現を
洗練させるほど（真の生成行動を使うほど）明確になる——しかし、**その構造への"アクセス"
（読み出し・分析）と、その構造を"操作"して所望の行動を引き出す介入は、全く異なる困難さを
持つ**ということである。前回レポートが最終的に至った「動的フィールドsteering下の行動は
タスクを完遂できないが、感覚運動ループとしては機能し続けている」という結論——受動的な構造の
観察は肯定的だが能動的な介入は否定的、という非対称な結果——は、本レポートの3フェーズを通じて
より強固なパターンとして確認された。言語プロンプトが果たしている役割（前回レポート§9・
本レポート§2.4で示唆された「単なる選択情報を超えた、タスク発火条件そのもの」としての役割）を
モデル内部の力学だけで代替する試みは、テストした全ての変種（タスク・タイミング・介入方式）に
おいて一貫して失敗した。

---

## 6. 実装の詳細：発見・修正したバグ

前回レポートと同じく、実装中に発見したバグは隠さず記録する。本レポートのフェーズ5・6・7は、
前回レポートで既にP6/P7原則に沿って検証済みのコンポーネント（`CausalStateEstimator`・
`DynamicFlowField`・`DynamicFieldHook`・`compute_steering_vectors`等）を再利用する設計を
徹底したため、新規に発見された計算上のバグは1件のみだった。

1. **`run_prompt_fadeout_entrainment.sh`の`common/env.sh`相対パス誤り**
   （実行環境設定、コードのバグではない）: `verification/README.md`が示す`env.sh`ソース
   テンプレート`source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/../common/env.sh"`は
   `verification/`直下のスクリプト用であり、`verification/<suite>/`のようにサブディレクトリに
   置かれたスクリプトからは、`common/`への相対パスが1階層分ずれる（`../common/env.sh`が正しく、
   `../../common/env.sh`相当の誤ったパスでは存在しないファイルを指す）。この誤りにより
   `.venv`が有効化されないまま`/usr/bin/python3`（システムPython、`wandb`等の依存関係が
   未インストール）でスクリプトが起動され、`ModuleNotFoundError: No module named 'wandb'`で
   即座にクラッシュした。GPUロールアウトを開始する前の最初の一行でクラッシュしたため実害はなく、
   `README.md`の記述通り「(adjust the relative `cd`/path count above to your actual suite
   depth)」という注記が実際に必要になった具体例である。パスを`.../..`から`..`へ修正して解消した
   （`run_phase5_*.sh`・`run_collect_action_chunks.sh`は元々`run_dynamic_vector_field_
   steering.sh`のEGL/HF環境変数を直接コピーする旧来パターンを踏襲しており、この問題を最初から
   踏んでいない）。

このほか、フェーズ7のDMD再評価で「TurnOnStoveの(2)(3)群間でp値が完全一致した」現象（§4.3末尾）
に遭遇したが、調査の結果Mann-Whitney検定の順位依存性による正当な振る舞いであり計算バグではない
ことを確認した——前回レポート§7バグ#3（天井効果の誤認）と同種の「一見バグに見えるが検定の
数学的性質として正しい」ケースであり、同じ轍を踏まないよう最初から順位依存性を疑って調査した。

---

## 7. 使用スクリプトと再現性

全成果物は`cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/`
以下に格納されている（前回レポートのフェーズ1〜4成果物と同じディレクトリを共有する）:

```
dynamic_vector_field_steering_CoffeePressButton.json   — フェーズ5結果（CoffeePressButton）
phase4_evaluation_metrics_CoffeePressButton.json       — フェーズ5評価指標
dynamic_vector_field_steering_CloseDrawer.json         — フェーズ5結果（CloseDrawer）
phase4_evaluation_metrics_CloseDrawer.json             — フェーズ5評価指標
prompt_fadeout_entrainment_PnPCounterToCab.json        — フェーズ6結果
collect_actions/<task>_seed{S}.npz                      — フェーズ7新規収集データ（action_chunk付き）
collect_actions/multitask_manifest.json                 — フェーズ7収集マニフェスト
action_dspace_test.json                                  — フェーズ7結果（行動チャンクPCA版D空間）
action_dspace_artifact_<task>.pkl                        — フェーズ7成果物
dynamics_embedding_on_collect_actions/                   — フェーズ7の中間比較群
  dynamics_embedding_test.json                            （collect_actions×eef/gripper代理指標）
  energy_field_test.json                                  （同、DMD再評価）
```

新規スクリプト（`cosmos_policy/experiments/robot/robocasa/analysis/verification/latent_dynamics/`）:

- `dynamic_vector_field_steering.py`（フェーズ5） — **新規コードなし**。前回レポートが実装した
  スクリプトを`--task_name CoffeePressButton`/`--task_name CloseDrawer`で再実行しただけであり、
  最初から task-genericに設計されていたことを確認する結果ともなった。
- `run_phase5_coffeepressbutton.sh`・`run_phase5_closedrawer.sh` — フェーズ5の実行ラッパー
  （GPU0/GPU1で並列実行、`common/env.sh`未使用の旧来パターン、既存の
  `run_dynamic_vector_field_steering.sh`をそのまま複製し`--task_name`のみ変更）。
- `prompt_fadeout_entrainment.py`（フェーズ6） — 新規実装。`dynamic_vector_field_steering.py`の
  `CausalStateEstimator`/`DynamicFlowField`/`DynamicFieldHook`/`FinalFeatCapture`/
  `get_action_with_dynamic_hook`/`xp_direction_to_raw`/`t1_noop_check`をインポートして再利用し、
  プロンプト切替ロジックのみを新規実装。
- `run_prompt_fadeout_entrainment.sh` — フェーズ6の実行ラッパー（`common/env.sh`使用）。
- `collect_action_chunks.py`（フェーズ7データ収集） — `attractor/collect_multitask.py`の
  `MultiTaskCapture`/`get_action_with_capture`/`sha256_file`/`git_commit_hash`を再利用し、
  `action_chunk`フィールドの追加保存のみを新規実装。
- `run_collect_action_chunks.sh` — フェーズ7データ収集の実行ラッパー。
- `action_dspace_test.py`（フェーズ7解析） — 新規実装。
  `dynamics_embedding_test.py`の`build_delay_embedding_input`/`deviation_mannwhitney`/
  `intrinsic_dim_quick`/`plot_embedding`、`dmd_jacobian_stability_test.py`の`windowed_dmd`を
  再利用し、行動チャンクPCAベースの結合ベクトル構築・DMD再評価ロジックのみを新規実装。

実行コマンド:

```bash
# フェーズ5（GPU必要、既存フェーズ1-2成果物に依存、CoffeePressButton/CloseDrawerで並列実行可）
bash cosmos_policy/experiments/robot/robocasa/analysis/verification/latent_dynamics/run_phase5_coffeepressbutton.sh
bash cosmos_policy/experiments/robot/robocasa/analysis/verification/latent_dynamics/run_phase5_closedrawer.sh

# フェーズ6（GPU必要、既存フェーズ1成果物に依存）
bash cosmos_policy/experiments/robot/robocasa/analysis/verification/latent_dynamics/run_prompt_fadeout_entrainment.sh

# フェーズ7データ収集（GPU必要、独立、~50分/8ファイル）
bash cosmos_policy/experiments/robot/robocasa/analysis/verification/latent_dynamics/run_collect_action_chunks.sh

# フェーズ7解析（GPU不要、オフライン、上記収集完了後）
python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.action_dspace_test \
    --collect_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/collect_actions \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification

# フェーズ7中間比較群（GPU不要、3群比較の(2)群、§4.2.5）
python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.dynamics_embedding_test \
    --collect_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/collect_actions \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/dynamics_embedding_on_collect_actions
python3 -m cosmos_policy.experiments.robot.robocasa.analysis.verification.latent_dynamics.energy_field_test \
    --embedding_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/dynamics_embedding_on_collect_actions \
    --out_dir cosmos_policy/experiments/robot/robocasa/analysis/results/latent_dynamics_verification/dynamics_embedding_on_collect_actions
```

---

## 8. 今後の課題

個別の検証に紐づく今後の検証案は各節末尾に記載した（§2.7・§3.7・§4.7）。以下は複数のフェーズに
またがる、または検証全体の設計に関わる横断的な課題である。

1. **因果的介入の仮説をさらに一般化する前に、介入手法そのものの見直しが必要** — フェーズ5
   （単一アフォーダンスタスク）・フェーズ6（プロンプトタイミング）という2つの軸での一般化が
   いずれも反証されたことは、動的フィールドsteering（kNNフローライブラリ+局所接平面射影+
   弱いエネルギー勾配）という介入手法自体の表現力・情報量が、言語プロンプトが供給する情報を
   代替するには構造的に不足している可能性を示唆する。今後の介入手法の改良（例: より高次の
   局所モデル、複数モードを持つフローライブラリ、視覚特徴を直接条件付けに使う等）を検討する
   前に、「介入の情報量」を定量的に見積もる予備検討が有益かもしれない。
2. **表現の仮説（フェーズ7）は他タスク・他次元設定への拡張余地が大きい** — 本レポートは
   PnPCounterToCab・TurnOnStoveの2タスクのみで検証し、行動PCA次元数も単一値(5)のみを試した。
   §4.7の今後の検証案（次元数スイープ、フェーズ2・3への展開）に加え、CloseDrawer・
   CoffeePressButton以外の「失敗episodeが十分得られる」タスクの追加収集も検討に値する。
3. **`fadeout_calls`のスイープが未実施** — フェーズ6は単一の切替タイミング(5)のみで反証したが、
   design.mdが示す範囲（5〜10）の上限や、それを超える値では異なる結果が出る可能性は排除
   できていない。
4. **プロンプトフェードアウトと単一アフォーダンスタスクの組み合わせは未検証** — フェーズ5・6は
   独立に検定されたが、「単一アフォーダンスタスク×プロンプトフェードアウト」という組み合わせ
   条件（両方の交絡要因を同時に除いた、最も条件の整った検証）はまだ実施していない。
