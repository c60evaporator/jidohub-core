# jidohubプラットフォーム

## jidohubとは

jidohubは、世界中のユーザーが開発した自動運転スタックを共有するためのプラットフォームです。

- リポジトリ: 自動運転スタックをアップロードして共有するためのデータストレージ
- API: 自動運転スタックをダウンロードしてシステム内で利用するための仕組み
    - Python API: Pythonライブラリで自動運転スタックの出力を得る
    - ROS API: ROSノードとして自動運転スタックを利用し、トピックで出力を得る

### PublicとPrivateリポジトリ

jidohubのリポジトリは、以下2種類に分けられます

- Public: 世界中の不特定多数のユーザーが自動運転スタックをアップロード・ダウンロードできる
- Private: 組織内・または認可を得たユーザのみが自動運転スタックをアップロード・ダウンロードできる

### アップロードできる自動運転スタックの種類

- Agent: 自動運転のPlanningをEnd-to-Endで実行、または自動運転の各種タスク（E2E planning、3D object detection等）を実行するコンポーネント。センサ等のデータを入力し、予測軌跡等を出力する
- Dataset: センサデータやアノテーションを含む自動運転用のデータセット。ベンチマークとして使用することも想定する
- Interface: Agentの入出力形式と、実車・シミュレーションとの入出力形式を変換するアダプタ

jidohub は自動運転向けの Agent / Dataset / Interface 共有プラットフォーム。
リポジトリ構成は以下の5つで、**core は他の4つすべてが依存する共通基盤**。

### jodohubのリポジトリ構成

jidohub本体は、以下のリポジトリで構成される

| リポジトリ | 役割 |
|---|---|
| jidohub-web | Agents / Datasets / Interfaces をホストするWebプラットフォーム |
| **jidohub-core（本リポジトリ）** | 標準スキーマ・Hubクライアント・configパーサ |
| jidohub-agents | Agentをロードして実行するPython API |
| jidohub-datasets | Datasetをロードして標準スキーマに変換するPython API |
| jidohub-interfaces | 実車・シミュレーションとの入出力変換 |

jidohubの利便性を上げるアドオンとして、以下のリポジトリも追加

| リポジトリ | 役割 |
|---|---|
| jidohub-server | ローカルで複数のモデルを管理し、リクエストに応じて推論やカタログ情報を提供するWebサーバー |
| jidohub-leaderboard | jidohubのAgentを使ってCARLA Leaderboard/Bench2Driveでクローズドループ評価。シナリオ作成等を実施するツールキット |

## Agent

Agentは、自動運転を構成する各種タスクを実行するコンポーネントで、AIモデルを核に置いています。

Agentが実行するタスクは、E2E planning（`sensing_to_planning`）や3D object detection（`object_detection_3d`）。タスクごとに決まった入力フォーマット（`Sample`等）と出力フォーマット（`E2EOutput`, `Detection3DOutput`等）を定めることで、異なるモデルを共通のAPI（後述の`AutoAgent`クラス）を用いた簡単な実装で動作させることができます。

また、Agentの実装タイプは大きく以下の2種類に分けられます。

| 実装タイプ (`agent_config.json`の"implementation.type"に記載) | 概要 | 動作環境（後述の`AutoAgent.from_pretrained`メソッドの`runner`引数） |
|---|---|---|
| native | jidohub-agentにネイティブ実装された公式Agent | Python APIと同プロセス（inprocess） |
| remote_code | ユーザーがjidobub-webにアップロードしたAgent | 独立したDockerコンテナ（docker） |

Agentのタスクや実装タイプ等の属性情報は、後述の`agent_config.json`に記載されます。

### Agentのリポジトリ構造

各Agentのリポジトリは、以下のフォルダ・ファイルで構成されます

```
*****/UniAD-tiny-nuScenes/   # agent_idがフォルダ名になる
├── agent_config.json        # エージェントの仕様を定義するメタデータ
├── README.md                # モデルカード（人間可読、frontmatterにタグ）
├── weights/
│   └── model.safetensors    # AIモデルの重みファイル
├── src/                     # エージェントの実装（動的ロード対象）
│   ├── modeling.py          # BaseAgent実装クラス
│   └── processing.py        # Processor実装クラス（前処理）
├── runtime/
│   ├── Dockerfile           # Python API用実行環境
│   └── launch/              # ROS用（将来）
└── benchmark.json           # ベンチマーク結果（NDS, mAP等）
```

### agent_config.json

Agentのタスクや実装方法等の属性情報を記述します。以下のフィールドを記載できます。

| フィールド | required | 型 | 記載内容 |
|---|---|---|---|
| schema_version | ✅ | str | スキーマのバージョン |
| agent_id | ✅ | str | Agentのタグ（jidohub_webでの検索IDとなる） |
| task | ✅ | str | タスク名 |
| sensors.cameras | | list[str] | カメラ名のリスト |
| sensors.requires_ego_state | | bool | 入力に自己位置が必須か |
| sensors.requires_command | | bool | 入力にコマンドが必須か |
| sensors.history_length | | int | 何系列前までの時系列入力を使用するか（temporalなモデルで使用） |
| implementation.type | ✅ | "native" / "remote_code" | Agentの実装タイプ |
| implementation.native_class | | str | ネイティブ実装Agentのクラス名（`implementation.type="native"`の場合のみ） |
| implementation.auto_map.AutoAgent | | str | リポジトリ内のユーザー実装Agentのファイルパスとクラス名（`implementation.type="remote_code"`の場合のみ） |
| implementation.auto_map.AutoProcessor | | str | リポジトリ内のユーザー実装Processorのファイルパスとクラス名（`implementation.type="remote_code"`の場合のみ） |
| runtime.isolation | ✅ | "required" / "not-required" | `runner="docker"`（[後述]()）が必須か |
| runtime.dockerfile | | str | リポジトリ内のDockerfileのパス（`runner="docker"`の場合のみ） |
| runtime.gpu_required | ✅ | bool | GPUが必須か |
| weights.path | | str | リポジトリ内の重みファイルのパス |
| weights.format | | "safetensors" | リポジトリ内の重みファイルのフォーマット |
| weights.sha256 | | str | safetensorsの整合性チェック用ハッシュ値 |
| platforms | ✅ | list["python" / "ros"] | 対応（Python API or ROS） |
| framework | ✅ | dict | 依存フレームワーク・ライブラリのバージョン制約 |
| license | ✅ | str | Agentのライセンス |
| training_datasets | | str | 学習に使用したデータセット名 |
| params | | dict | Agentに渡すデフォルトパラメータ |

UniADでの記載例を示します

```json
{
  "schema_version": "0.1",
  "agent_id": "acme/UniAD-tiny-nuScenes",
  "task": "sensing_to_planning",
  "sensors": {
    "cameras": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
    "requires_ego_state": true,
    "requires_command": true,
    "history_length": 3
  },
  "implementation": {
    "type": "remote_code",
    "auto_map": {
      "AutoAgent": "src/modeling.py:UniADAgent",
      "AutoProcessor": "src/processing.py:UniADProcessor"
    }
  },
  "runtime": {
    "isolation": "required",
    "dockerfile": "runtime/Dockerfile",
    "gpu_required": true
  },
  "weights": [
    {"path": "weights/model.safetensors", "format": "safetensors", "sha256": "1111111111111111111111111111111111111111111111111111111111111111"}
  ],
  "intermediate_outputs": ["detection", "tracking", "map", "motion_forecast", "occupancy"],
  "platforms": ["python", "ros"],
  "framework": {"torch": "2.1", "cuda": "12.1", "library": "mmdet3d", "library_version": "1.0.0rc6"},
  "license": "Apache-2.0",
  "training_datasets": ["nuscenes/v1.0-trainval"],
  "params": {"bev_h": 200, "bev_w": 200, "planning_steps": 6}
}
```

### Agentのタスク一覧

#### Planning系

| タスク名 | 入力 | 出力 | 出力dataclass | 役割 |
|---|---|---|---|---|
| sensing_to_planning | センサデータ（画像・LiDAR等） | 予測軌跡 | `E2EOutput` | いわゆる**End-to-end自動運転** |
| vision_language_action | センサデータ（画像）＋テキスト | 予測軌跡＋テキスト | `VLAOutput` | いわゆる**VLA自動運転** |
| planning | 周囲オブジェクト位置（`Detection3DOutput`）＋マップ情報（`MapOutput`）＋自己位置（`EgoState`） | 予測軌跡 | `PlanningOutput` | Perceptionなしのplanning用モデル |

#### Perception系

| タスク名 | 入力 | 出力 | 出力dataclass | 役割 |
|---|---|---|---|---|
| object_detection_3d | センサデータ（画像・LiDAR等） | 3D bounding box | `Detection3DOutput` | 3D物体検出 |
| object_tracking_3d | 複数フレームのセンサデータ（画像・LiDAR等） | 3D bounding box＋tracking_id | `Tracking3DOutput` | 3Dトラッキング |
| map_construction | センサデータ（画像・LiDAR等） | ベクトル化された地図表現 | `MapOutput` | オンラインHD地図構築（MapTR等） |
| object_detection_2d | 画像 （Grounding DINOのようにテキストプロンプトも入力する場合あり）| 2D bounding box | `Detection2DOutput` | 2D物体検出 |
| object_tracking_2d | 複数フレーム画像 | 2D bounding box＋tracking_id | `Tracking2DOutput` | 2Dトラッキング |
| instance_segmentation_2d | 画像（SAM3のようにポイントやテキストプロンプトも入力する場合あり） | インスタンスマスク | `InstanceSegmentation2DOutput` | インスタンスセグメンテーション |
| instance_segmentation_2d_tracking | 複数フレーム画像（SAM3のようにポイントやテキストプロンプトも入力する場合あり） | 各フレームのインスタンスマスク＋track_id | `InstanceSegmentation2DTrackingOutput` | インスタンスセグメンテーション＋トラッキング |
| semantic_segmentation_2d | 画像 | セグメンテーションマスク | `SemanticSegmentation2DOutput` | セマンティックセグメンテーション |
| panoptic_segmentation_2d | 画像 | セグメンテーションマスク＋インスタンスマスク | `PanopticSegmentation2DOutput` | パノプティックセグメンテーション |
| image_classification | 画像 | 各フレームのインスタンスマスクとtrack_id | `ClassificationOutput` | 画像分類 |
| depth_estimation | 画像または複数フレーム画像 | 深度画像 | `DepthOutput` | 単眼深度推定（Depth Anything v2のような相対深度モデル、Metric3Dのようなメトリックモデル、Video Depth Anythingのような複数フレーム入力モデルを内包） |
| multi_view_reconstruction | 多視点画像 | カメラパラメータ・深度画像・point map・point track等 | `ReconstructionOutput` | 3次元再構成（VGGT等） |
| video_text_to_text | 画像＋テキスト | テキスト | `TextOutput` | LLMベースVLM |

#### Prediction系

| タスク名 | 入力 | 出力 | 出力dataclass | 役割 |
|---|---|---|---|---|
| motion_forecasting | 未定 | 周囲オブジェクトの将来位置予測 | `MotionForecastOutput` | 周囲オブジェクトの将来位置予測 |
| occupancy_prediction | 未定 | Occupancy予測 | `OccupancyOutput` | Occupancy予測 |

#### Control系

| タスク名 | 入力 | 出力 | 出力dataclass | 役割 |
|---|---|---|---|---|
| control | 予測軌跡 | 制御出力 | `ControlOutput` | 制御出力を予測する |

#### 将来追加しても良いかもしれないタスク

将来的には以下も検討

| タスク名 | 入力 | 出力 | 出力dataclass | 役割 |
|---|---|---|---|---|
| novel_view_synthesis | 多視点画像 | 要求された視点でレンダリングした画像 | `NovelViewSynthesisOutput` | 3次元再構成 |



### Python APIのAgent本体と利用側の実装

#### Agent本体の実装

タスクごとに定義された継承元クラスを継承することで、そのタスクを実行するAgentを実装できます。

| タスク名 (`agent_config.json`の"task"に記載) | 継承元クラス | タスク内容 | 入力 | 出力 |
|---|---|---|---|
| sensing_to_planning | `E2EAgent` | **End-to-end自動運転** | `Sample` | `E2EOutput` |
| object_detection_3d | `Detection3DAgent` | **物体検出** | `Sample` | `Detection3DOutput` |

例えばEnd-to-end自動運転を行うAgentを作成するには、以下のように`E2EAgent`クラスを継承してオリジナルのAgentクラスを実装します。

```python
class UniADAgent(E2EAgent):

```

なお各タスクの継承元エージェントクラスは、以下のように`BaseAgent`クラスを継承して実装されています。

```python
class BaseAgent(ABC):
    config: AgentConfig

    @classmethod
    def from_pretrained(cls, repo_id_or_path, **kwargs) -> "BaseAgent":
        ...  # config読込 → auto_map解決 → 重みロード

    @abstractmethod
    def predict(self, sample: Sample) -> TaskOutput: ...

# タスクごとに出力型を固定した抽象クラス
class Detection3DAgent(BaseAgent):
    def predict(self, sample: Sample) -> Detection3DOutput: ...

class E2EAgent(BaseAgent):
    def predict(self, sample: Sample) -> E2EOutput: ...
```

#### 利用側の実装

Agentをダウンロードして推論に利用するユーザーは、以下のように実装します

```python
from adhub import AutoAgent

agent = AutoAgent.from_pretrained("*****/UniAD-tiny-nuScenes")
output = agent.predict(sample)   # -> E2EOutput
```

入力データとなるsample、および出力されるoutputのフォーマットは、`Sample`や`E2EOutput`のようなデータクラスでタスクごとに定型に限定されています。
jidohub-interfaceで提供される**Interfaceを用いること**で、少ない実装で**異なる入力・出力形式に対応**させることができます。

##### 実行環境の指定

`AutoAgent.from_pretrained`メソッドの`runner`引数で、Agentの実行環境をPython APIと同プロセスにするか、Dockerコンテナで動かすかを選択できます

```python
agent = AutoAgent.from_pretrained(
    "*****/UniAD-tiny-nuScenes",
    runner="docker",     # "inprocess" | "docker"
)
```

| `runner`引数| 動作環境（後述の`AutoAgent.from_pretrained`メソッドの`runner`引数） | Python APIとのやりとり |
|---|---|---|
| inprocess | Python APIと同プロセス（inprocess） | プロセス内で入出力データをやり取りする |
| docker | 独立したDockerコンテナ | `Sample`をシリアライズ（Arrow/msgpack等）してRPCで渡し、`Output`を受け取るProxy Agentを返す |

実行環境はAgentの実装方法と深く関連しており、`runner="inprocess"`は以下の理由から、基本的にnative実装のAgent（jidohub-agentにネイティブ実装された公式Agent）のみ対応しています

- inprocessだとユーザAgentの依存パッケージのインストールが難しい
- セキュリティ上ユーザAgentはDockerコンテナで隔離実行したい

よってユーザがアップロードしたAgentは、基本的に`runner="docker"`のみ対応します。

##### `runner="docker"`での画像転送

`runner="docker"`の場合、Python APIはホストPC上のプロセス（あるいは独自に立ち上げたDockerコンテナ）で、AgentはDockerコンテナ上で動作するため、両者間の画像転送が動作サイクルのボトルネックとなりえます。これに対処するためjidohubのPython APIでは、以下の2種類の転送方法を選択できます

||JPEG転送（使い勝手重視）|生画素メモリ共有（速度重視）|
|---|---|---|
|転送方法|HTTP / gRPC（バイトストリーム）|共有メモリ（/dev/shm）|
|転送時のデータ形式|JPEGバイト列（`EncodedImage`）|生画素（`pixels`）。将来的にはNV12も検討|
|メリット|動作条件が緩い。Python API呼び出しとAgentが別マシンでも動く|高速|
|ユースケース|viewerでの可視化、評価ジョブ、自動アノテーション、クロスホスト構成|実車のdockerデプロイ（画素がライブセンサから生で供給される）、同一ホストでの閉ループシミュレーション（デコードなしで画素が取得できるケース）|

**生画素メモリ共有**は高速ですが、**以下の3条件を満たさないと動作しない**ので注意してください。

- `runner="docker"`で、同一ホスト
- 画素がライブセンサから生で供給される（デコードが必要だと逆に遅くなるため）
- 呼び出し側が共有メモリバッファへ直接書き込める
- AgentがnativeまたはVerified（Agentが動作するコンテナ内外から互いの共有メモリセグメントに読み書きできるため、セキュリティ上の理由から認証済Agentのみ許可する設定としている）

両方法のnuScenesデータ（6カメラ 1600×900）における概算所要時間は以下となります

||JPEG転送 (JPEG/stream)|生画素メモリ共有 (raw/shm)|
|---|---|---|
|エンコード所要時間|0ms（JPEGをそのまま渡す）|0ms|
|ペイロード|1.7MB|26MB（共有メモリ上）|
|転送|1〜3ms|ほぼ0（ハンドルのみ）|
|デコード	5ms（nvJPEG）〜36ms（CPU）|0ms|
|合計|10〜40ms|0〜8ms|

実装時にはAgentのインスタンス作成時に`transport`引数を指定することで両方法を選択できます。`transport="stream"`ならJPEG転送、`transport="shm"`なら生画素メモリ共有が使用されます。

```python
# JPEG転送
agent = AutoAgent.from_pretrained("acme/UniAD-tiny@v1.2", transport="stream")
# 生画素メモリ共有
agent = AutoAgent.from_pretrained("acme/UniAD-tiny@v1.2", transport="shm")
```

デフォルトでは`transport="auto"`が利用され、「`runner="inprocess"`なら転送なし、`runner="docker"`なら`transport="shm"`を指定する」という動作をします。

JPEG転送、生画素メモリ共有双方の実装例を示します（**実際はSampleへの変換処理をInterfaceに任せるケースが多い**ですが、ここでは画像を適切な形式でSampleに渡す方法が分かりやすいようスクラッチで変換処理を実装します）

**JPEG転送の実装例**

```python
from jidohub.core.schemas import CameraFrame, EncodedImage, ImageFormat, LidarSweep, Sample

def build_sample_encoded(record: Any) -> Sample:
    """データセットの JPEG をデコードせずに載せる"""
    cameras = {}
    for channel, meta in record.cameras.items():
        raw = Path(meta.path).read_bytes()
        cameras[channel] = CameraFrame(
            intrinsic=meta.intrinsic,
            sensor_to_ego=meta.sensor_to_ego,
            channel=channel,
            encoded=EncodedImage.from_bytes(
                raw,
                ImageFormat.JPEG,
                height=meta.height,
                width=meta.width,
            ),
        )
    return Sample(
        timestamp=record.timestamp,
        ego_to_global=record.ego_to_global,
        cameras=cameras,
        lidar=LidarSweep(
            points=record.points,
            sensor_to_ego=record.lidar_sensor_to_ego,
        ),
        sample_id=record.token,
    )

agent = AutoAgent.from_pretrained("acme/UniAD-tiny@v1.2", transport="stream")
# ここにrecord取得処理を記載（ファイルから読込等）
sample = build_sample_encoded(record)
output = agent.predict(sample)
```

**プロファイルBの実装方法**

```python
def build_sample_raw(frame_buffers: dict[str, np.ndarray], record: Any) -> Sample:
    """カメラ / ISP から得た生画素をそのまま載せる"""
    cameras = {
        channel: CameraFrame(
            intrinsic=record.intrinsics[channel],
            sensor_to_ego=record.extrinsics[channel],
            channel=channel,
            pixels=pixels,  # (H, W, 3) uint8 RGB
        )
        for channel, pixels in frame_buffers.items()
    }
    return Sample(
        timestamp=record.timestamp,
        ego_to_global=record.ego_to_global,
        cameras=cameras,
    )

agent = AutoAgent.from_pretrained("acme/UniAD-tiny@v1.2", transport="shm")
# ここにrecord取得処理を記載（センサから画素値受取等）
sample = build_sample_encoded(record)
output = agent.predict(sample)
```

##### Python API呼び出し側もDockerコンテナの場合の注意

Python API呼び出し側もDockerコンテナの場合、`transport="shm"`を使用するための条件「呼び出し側が共有メモリバッファへ直接書き込める」を満たすためには、IPC名前空間の共有と、/dev/shmのサイズ拡張が必要となります。`docker-compose.yml`で表すと、以下の設定が必要となります。

```yml
# docker compose
services:
  producer:                    # Python API呼び出し側コンテナ
    ipc: shareable             # shareableにする必要
    shm_size: 512mb            # 送りたい画像サイズに合わせて拡張
  agent-runner:                # Agentが動作するコンテナ
    ipc: "service:producer"    # producer の IPC 名前空間に参加
    shm_size: 512mb            # こちらも拡張
```

## Dataset

## Interface

jidohub-agentが提供する入出力データのフォーマットは、`Sample`や`E2EOutput`のようなデータクラスでタスクごとに定型に限定されています。
jidohub-interfaceは、入力データを`Sample`のような**Agentの入力形式に変換**、または`E2EOutput`のような**Agentが出力する結果を他形式に変換**する**Interface**を提供します。

Interfaceを利用するユースケースの代表例として、以下が挙げられます

|ユースケース|入力側Interface|出力側Interface||
|---|---|---|---|
|nuScenes形式のDatasetと組み合わせた評価||||
|CARLA Leaderboardによるクローズドループ評価||||
