# jidohub-agents の設計

Agent をロードして実行する Python API の設計。`jidohub-core` の標準スキーマを入出力とし、
オンライン推論（実車・シミュレーション）とオフライン評価の双方を同じ実装から提供する。

関連文書
    - `docs/design/tasks/2d_tasks.md` — 2D タスクの入出力型、画素座標の規約
    - `docs/design/tasks/streaming_agents.md` — `reset` / `step` の契約
    - `docs/design/tasks/coordinate_transforms.md` — 座標変換の責務分担

---

## 1. 位置づけと依存

| リポジトリ | 役割 |
|---|---|
| jidohub-core | 標準スキーマ・config・直列化・Hub |
| **jidohub-agents（本リポジトリ）** | Agent のロードと実行 |
| jidohub-datasets | データセット → 標準スキーマ |
| jidohub-interfaces | 実車・シミュレーションとの接続 |
| jidohub-server | HTTP / ジョブ API（agents を包む） |

- **core にのみ依存する。** datasets / interfaces / server に依存しない
- **`torch` がここで初めて必須依存になる。** core と datasets が torch なしで動くことは
  プラットフォーム全体の前提であり、その境界がこのリポジトリ
- 評価（`evaluate`）は当面 agents 内のモジュールとして持つ。
  閉ループ評価が視野に入った時点で切り出す

---

## 2. Agent の契約

### 2.1 基本形

```python
class BaseAgent(ABC):
    config: AgentConfig

    @classmethod
    def from_pretrained(cls, reference, **kwargs) -> "BaseAgent": ...

    @abstractmethod
    def predict(self, input): ...
```

- **入力は単一引数**。`pack(input)` → RPC → `unpack` → `predict` という経路が
  引数の増加に耐えないため（`streaming_agents.md` 4 章）。
  複合入力が必要なタスクは `<TaskName>Input` 型を core に定義する
- タスクごとの抽象クラスが**出力型を固定**する

```python
class Detection3DAgent(BaseAgent):
    def predict(self, sample: Sample) -> Detection3DOutput: ...

class Detection2DAgent(BaseAgent):
    def predict(self, sample: ImageSample) -> Detection2DOutput: ...
```

### 2.2 ステートフル Agent（`StreamingMixin`）

tracking や時系列モデル（UniAD 等）は状態を持つ。**タスク横断の能力**として定義し、
`BaseTracker` のような tracking 専用クラスにはしない（`streaming_agents.md` 3 章）。

```python
class StreamingMixin:
    def reset(self) -> None: ...          # Agent 作者が実装
    def step(self, input): ...            # Agent 作者が実装
    def predict(self, inputs: list):      # 基底が提供。reset → step ループ → 集約
        ...
```

- **Agent 作者が実装するのは `reset()` と `step()` のみ**
- interfaces は `step()` を、評価ハーネスは `predict()` を呼ぶ
- 未 `reset()` での `step()` は明示的にエラー（前シーンの状態が漏れるのを防ぐ）
- **検証可能な不変条件**: `predict(inputs)` の結果は、`reset()` してから
  `step()` を手動ループした結果と一致する。共通テストで機械的に検証する

### 2.3 共通テストスイート

`BaseAgent` を実装する Agent が満たすべき性質を、**再利用可能なテストスイート**として提供する。
ネイティブ実装にも、外部の Agent 作者にも同じものを適用できる。

- 出力型が `TaskType` の宣言と一致する
- `config.prompt.required` が `true` なら、プロンプトなしの入力で明示的にエラー
- ステートフルなら `predict` と `reset`+`step` ループが一致する
- 2D なら出力座標が入力 `Image` の現サイズ基準である（4.3）

---

## 3. ロードと解決

### 3.1 `AutoAgent`

```python
agent = AutoAgent.from_pretrained("acme/UniAD-tiny@v1.2")
```

処理の流れ

1. `HubClient.snapshot()` でリポジトリをローカルに用意（core が担当）
2. `load_agent_config()` で `agent_config.json` を読む（core が担当）
3. `implementation.type` に応じてクラスを解決（3.2）
4. `runtime.isolation` と引数から Runner を決める（5 章）
5. 重みをロードし、インスタンスを返す

**core との責務境界**: 取得・検証・config 解釈は core。
クラス解決・重みロード・実行は agents。

### 3.2 二層構造

| `implementation.type` | 解決方法 | 実行 |
|---|---|---|
| `native` | agents 同梱のレジストリから `native_class` を引く | inprocess 可 |
| `remote_code` | `auto_map` のパスからモジュールを動的ロード | **隔離必須** |

`remote_code` は未審査コードの実行にあたる。config 側で
`runtime.isolation == "required"` が強制されている（core の検証）が、
**agents 側でも二重に検証**する。config を経由しない経路でも破られないようにするため。

### 3.3 ネイティブ実装のレジストリ

```python
NATIVE_AGENTS: dict[str, type[BaseAgent]] = {
    "CenterPointAgent": CenterPointAgent,
}
```

- 登録は明示的に行う（エントリポイント経由の暗黙登録にしない）。
  **どのクラスが審査済みかが一覧で読めること**を優先する
- ネイティブ実装は**純 PyTorch に保つ**。`mmcv` / `spconv` などの
  ビルドが不安定な依存を入れない。mm 系は `remote_code` + docker の側で扱う

---

## 4. Processor と前後処理

### 4.1 責務

`Sample` / `ImageSample` ↔ モデル固有テンソルの変換を担う。**Agent ごとに実装する。**

core の型はデータセット非依存だが、モデルの入力形式（解像度、正規化、
voxel 化のパラメータ）はモデル固有であり、core に置けない。

### 4.2 前処理と後処理は変換情報を共有する（重要）

モデル入力用に resize / crop / padding を行った場合、**後処理はその逆変換を知る必要がある**。
SAM の `postprocess_masks` がパディング情報を要求するのと同じ構造。

```python
@dataclass
class PreprocessResult:
    tensors: dict[str, torch.Tensor]
    transform: ImageSource | None   # 入力 Image → モデル入力 の変換
    input_size: tuple[int, int]     # 入力 Image の (height, width)
```

Processor は変換情報を**戻り値に含める**。Agent がグローバル状態として持ち回ると、
ストリーミング時やバッチ時に取り違える。

### 4.3 出力座標は入力 `Image` の現サイズ基準に戻す（Agent の責務）

`2d_tasks.md` 3.2 の規約により、出力座標は**入力 `Image` の現サイズ基準**でなければならない。
Agent がモデル入力用に resize している場合、戻すのは Agent の仕事である。

- **ボックス**は乗算で戻せる
- **マスクは再サンプリングが必要**。core が `to_source_image()` で scale を伴う
  マスク移動を `NotImplementedError` にしているのは、この処理を core に持ち込まないため

```
モデル出力（モデル解像度、二値化前）
  → F.interpolate で入力 Image 解像度へ         ← 二値化の前に行う
  → 二値化 → 外接矩形 → Instance2D
```

**二値化してから最近傍で拡大しないこと。** 境界が劣化する。

利用者側の `output.to_source_image(image)` は、Agent が上記を済ませている前提で
crop の平行移動だけを行う。

### 4.4 画像デコーダの登録

`frame.image.array` を呼ぶには、`register_image_decoder()` によるデコーダ登録が必要である。
core は画像コーデックに依存しないため、登録は利用側の責務になる。

**agents は datasets に依存しない**（星形依存）ので、agents 側にも登録の仕組みが要る。

- `jidohub.agents` の import 時に、利用可能な最速のデコーダを登録する
- 優先順位: 既に登録済みなら**上書きしない** → nvJPEG（利用可能なら）→ Pillow
- 判定は `importlib.util.find_spec` で行い、import 失敗時に
  `ImageDecodeError` ではなく `ImportError` が出る状態を作らない
  （datasets の `register_default_decoder` と同じ方針）

nvJPEG を使う利点は、デコード結果が最初から GPU 上に載ることである。
コンテナ内の推論では効果が大きい。

---

## 5. Runner

### 5.1 責務

「どのプロセスで動かすか」を決める。**Agent の性質ではなく実行環境の性質**なので、
`agent_config.json` には書かず `from_pretrained` の引数で指定する。
ただし config の `runtime.isolation` による**制約**は受ける。

| Runner | 用途 | 制約 |
|---|---|---|
| `inprocess` | native / 自作 Agent の開発・デバッグ | `isolation: required` の Agent では**禁止** |
| `docker` | 未審査コード、依存衝突するモデル | — |

`runner="auto"`（既定）は `runtime.isolation` から決める。
`required` なら docker、それ以外は inprocess。

### 5.2 転送プロファイル

docker runner は、`Sample` を直列化してコンテナへ渡す。

| `transport` | 形式 | 用途 |
|---|---|---|
| `stream`（既定） | HTTP / gRPC のバイト列 | ホストをまたげる。データセット駆動 |
| `shm` | 共有メモリ | 同一ホスト・ライブセンサ |

**`transport="auto"` は `shm` を選ばない。** 共有メモリの利得は
「画素が生で供給され、生産側が共有バッファへ直接書ける」ときにしか生じず、
Runner 側からは判定できないため。データセット駆動では JPEG のまま運ぶ方が速い。

**`shm` は IPC 名前空間の共有を要するため隔離が弱まる。**
`implementation.type == "remote_code"` の Agent では拒否する。

### 5.3 セッション管理（ステートフル Agent）

`StreamingMixin` を持つ Agent は、**同一ストリームの連続フレームを
同じインスタンスにルーティング**する必要がある。

- セッション ID の発行と対応付け
- アイドルタイムアウトによる破棄（状態がメモリを占有し続けるため）
- **1 インスタンスを複数ストリームで共有できない**。
  同時ストリーム数だけインスタンスが必要になり、GPU メモリの見積もりに直結する

server 側の実装が主だが、Runner の抽象がセッションを表現できる必要がある。

---

## 6. 評価（当面 agents 内）

### 6.1 メトリクスは自前実装しない

nuScenes の NDS / mAP は **公式 devkit の評価コードを使う**。
mmdetection3d も内部で devkit を呼んでいるため、同じコードを使えば
「メトリクス実装の差」という変数が消え、モデル出力の差だけを比較できる。

evaluate モジュールの役割は、`Detection3DOutput` → devkit の提出フォーマットへの
変換と、devkit 評価のラッパに留める。

### 6.2 ベンチマークハーネス

データセットを回して `predict()` を呼び、結果を集計する CLI。
datasets に依存するため **optional extra** とし、通常のインストールには含めない。

```
pip install jidohub-agents              # 推論のみ
pip install jidohub-agents[benchmark]   # datasets と devkit も入る
```

### 6.3 評価プロトコルの記録

タスクを統合した結果（プロンプトの有無で分けない）、**評価条件の記録が必須**になった。

- プロンプトの有無・種別・**供給源**（GT 由来 / モデル出力 / 人手）
- 上流タスクの出力を入力に取る場合（detection-based tracker 等）、その供給源
- ストリーミング Agent では reset のタイミング

**GT 由来のプロンプトを与えた SAM2 と Mask R-CNN の mask AP を同じ表に並べない。**
評価設定が一致するラン同士のみ比較を許す。

---

## 7. CenterPoint-Pillar（最初のネイティブ実装）

### 7.1 なぜ pillar 版か

voxel 版は sparse convolution（`spconv`）を必要とし、CUDA 固定の依存が増える。
**pillar 版は純 PyTorch で完結**する（scatter + 通常の Conv2D）。

ネイティブ実装の目的は、(a) スキーマ設計の検証、(b) 登録者が真似るお手本、
(c) デモがゼロ依存で動くこと、(d) inprocess で動く審査済み実装、である。
精度（NDS 60 前後）はこの目的に十分。

### 7.2 mmdetection3d は「重み工場」

学習パイプライン（データ拡張、CBGS、スケジュール）の再現は推論の数倍の労力がかかり、
失敗すると「実装が悪いのかスキーマが悪いのか」の切り分けができなくなる。

1. mmdet3d で学習（または公式チェックポイントを流用）
2. **重み変換スクリプト**（mmdet3d の state_dict → ネイティブ実装の state_dict）
3. 同一入力での出力一致テスト

mmdet3d は開発環境に留め、**agents のランタイム依存には入れない**。
この変換 + 一致検証は、将来ユーザーが mm 系で学習したモデルを持ち込む際のお手本にもなる。

### 7.3 三段階の検証

| 段階 | 内容 | CI |
|---|---|---|
| 1. モジュール単位 | pillar encoder / FPN / head の出力が mmdet3d と一致 | fixture 比較。毎 PR |
| 2. end-to-end | 単一サンプルの最終 Box 出力が一致 | fixture 比較。毎 PR |
| 3. データセット全体 | nuScenes val の NDS / mAP | 手動 / nightly |

**段階 1・2 の fixture は mmdet3d の入出力を一度だけ書き出してコミット**する。
以後の CI は mmdet3d も実データも要求しない。

不一致の原因は、モデル本体より**前処理・後処理**（voxel 化の丸め、NMS、座標変換）に
潜むことが圧倒的に多い。段階 1 が通っても段階 2 は別途必要である。

**許容基準**: 段階 3 で mmdet3d の報告値と NDS ±0.3 ポイント程度。
完全一致を追うより、段階 1・2 の厳密な一致で実装の正しさを担保する。

段階 3 の結果は `benchmark.json` に記録する。これは**自己申告値**であり、
将来 server が実行する platform-verified 値とは区別する。

---

## 8. スコープ（当面やらないこと）

- **Trainer / 学習ループの標準化**。自動運転モデルの学習は
  マルチタスク損失・サンプリング戦略・多段学習などモデルごとの差異が激しく、
  共通抽象の設計はスキーマ設計より難しい。推論の標準化で価値を証明してから
- **CenterPoint voxel 版**（spconv 依存）
- **UniAD のネイティブ実装**。`remote_code` + docker で扱う
- **`pack_into` / 共有メモリ転送**。実車・シミュレーション対応時
- **閉ループ評価**

---

## 9. 実装順序

1. `BaseAgent` / タスク別抽象クラス / `StreamingMixin` の契約
2. `AutoAgent` の解決機構（native のみ、inprocess のみ）
3. デコーダ登録
4. CenterPoint-Pillar のネイティブ実装 + 重み変換 + 段階 1・2 の一致テスト
5. ベンチマークハーネス + 段階 3
6. docker Runner
7. `remote_code` の動的ロード

**1 を最初に確定させる。** `StreamingMixin` を後から入れると全 Agent に影響する。
2〜5 が「CenterPoint が動く」までの最短路であり、6・7 は server 統合の前提。
