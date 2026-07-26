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

- Agent: 自動運転のPlanningをEnd-to-Endで実行、または自動運転の各種タスク（Object Detection、等）を実行するコンポーネント。センサ等のデータを入力し、予測軌跡等を出力する
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

Agentが実行するタスクは、以下のように分類されます。

| タスク名 (`agent_config.json`の"task"に記載) | 入力 | 出力 | タスク内容 |
|---|---|---|---|
| sensing-to-planning | センサデータ（`Sample`クラス） | 予測軌跡と中間出力（`E2EOutput`クラス） | いわゆる**End-to-end自動運転** |
| sensing-to-detection | センサデータ（`Sample`クラス）  | 3D bounding box（`Detection3DOutput`クラス） | いわゆる**物体検出** |

タスクごとに入力データクラス（`Sample`等）と出力データクラス（`E2EOutput`, `Detection3DOutput`等）を定めることで、異なるモデルを共通のAPI（後述の`AutoAgent`クラス）を用いた簡単な実装で動作させることができます。

また、Agentの実装方法は大きく以下の2種類に分けられます。

| 実装 (`agent_config.json`の"implementation"に記載) | 概要 | 動作環境（後述の`AutoAgent.from_pretrained`メソッドの`runner`引数） |
|---|---|---|
| native | jidohub-agentにネイティブ実装された公式Agent | Python APIと同プロセス（inprocess） |
| remote_code | ユーザーがjidobub-webにアップロードしたAgent | 独立したDockerコンテナ（docker） |

### Agentのリポジトリ構造

Agentリポジトリは、

```
*****/UniAD-tiny-nuScenes/
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

#### agent_config.json

```json
{
  "schema_version": "0.1",
  "agent_id": "acme/UniAD-tiny-nuScenes",
  "task": "sensing-to-planning",
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

### Python APIのAgent本体と利用側の実装

#### Agent本体の実装

タスクごとに定義された継承元クラスを継承することで、そのタスクを実行するAgentを実装できます。

| タスク名 (`agent_config.json`の"task"に記載) | 継承元クラス | タスク内容 | 入力 | 出力 |
|---|---|---|---|
| sensing-to-planning | `E2EAgent` | **End-to-end自動運転** | `Sample` | `E2EOutput` |
| sensing-to-detection | `Detection3DAgent` | **物体検出** | `Sample` | `Detection3DOutput` |

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

`AutoAgent.from_pretrained`メソッドの`runner`引数で、実行環境をDockerかPython APIと同プロセスかを選択できます

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

実行環境はAgentの実装方法と深く関連しており、`runner="inprocess"`は基本的にnative実装のAgent（jidohub-agentにネイティブ実装された公式Agent）のみ対応しています（ユーザAgentの依存パッケージのインストールが困難＆セキュリティ上ユーザAgentはDockerで隔離実行したいため）。
ユーザがアップロードしたAgent（大半のAgentを占めると思われる）は、基本的に`runner="inprocess"`

## Dataset

## Interface

jidohub-agentが提供する入出力データのフォーマットは、`Sample`や`E2EOutput`のようなデータクラスでタスクごとに定型に限定されています。
jidohub-interfaceは、入力データを`Sample`のような**Agentの入力形式に変換**、または`E2EOutput`のような**Agentが出力する結果を他形式に変換**する**Interface**を提供します。

Interfaceを利用するユースケースの代表例として、以下が挙げられます

|ユースケース|入力側Interface|出力側Interface||
|---|---|---|---|
|nuScenes形式のDatasetと組み合わせた評価||||
|CARLA Leaderboardによるクローズドループ評価||||
