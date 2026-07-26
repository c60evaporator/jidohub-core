# CLAUDE.md — jidohub-core

このファイルには「ソースやconfigを読んでも分からない設計判断」だけを記載する。
依存バージョン・ディレクトリの中身・コマンドの詳細は `pyproject.toml` / 各モジュールを参照すること。

---

## 1. このリポジトリの位置づけ

jidohub は自動運転向けの Agent / Dataset / Interface 共有プラットフォーム。
リポジトリ構成は以下の5つで、**core は他の4つすべてが依存する共通基盤**。

| リポジトリ | 役割 |
|---|---|
| jidohub-web | Agents / Datasets / Interfaces をホストするWebプラットフォーム |
| **jidohub-core（本リポジトリ）** | 標準スキーマ・Hubクライアント・configパーサ |
| jidohub-agents | Agentをロードして実行するPython API |
| jidohub-datasets | Datasetをロードして標準スキーマに変換するPython API |
| jidohub-interfaces | 実車・シミュレーションとの入出力変換 |

### 依存の原則（星形依存）

- **core は他の jidohub パッケージに一切依存しない。** 依存は常に `core ← 他` の一方向。
- 逆方向の依存（core が agents / datasets を import する）は**禁止**。
  必要になった時点で設計が間違っているので、実装で回避せず設計に戻ること。

### ディレクトリ構成と各ファイルの責務

各ファイルの中身ではなく、**どこに何を書くか（書かないか）**の判断を示す。
新しいコードを追加する際は、まずこの表のどこに属するかを決めること。
どこにも属さない場合は core の責務でない可能性が高い（5 章を参照）。

```
src/
└── jidohub/                     ← namespace package。__init__.py を置かない（2.2）
    └── core/
        ├── __init__.py          最小限の再エクスポートのみ。ロジックを書かない
        ├── tasks.py             TaskType enum（プラットフォームの語彙の唯一の正）
        ├── geometry.py          座標変換・回転表現のヘルパ（純 numpy）
        ├── schemas/             標準スキーマ（dataclass）
        │   ├── __init__.py      再エクスポート
        │   ├── version.py       SCHEMA_VERSION / assert_compatible / is_compatible（互換判定の唯一の正）
        │   ├── sample.py        入力: Sample / CameraFrame / LidarSweep /
        │   │                    RadarSweep / EgoState / DrivingCommand
        │   └── outputs.py       出力: Box3D / Detection3DOutput / MapOutput /
        │                        PlanningOutput / E2EOutput / AgentForecast など
        ├── config/              agent_config.json のパース・検証（Pydantic）
        ├── serialization/       Sample / Output の直列化・復元（pack / unpack）
        └── hub/                 Hub クライアント（snapshot / revision 解決 / チェックサム検証）
```

配置ルール

- **`schemas/`**: 型定義のみ。データセット固有・モデル固有の処理を書かない。
  検証は形状チェック等の軽量なものに留める（毎フレーム生成されるため）。
- **`geometry.py`**: 純粋関数のみ。状態を持たない。scipy / torch に依存しない。
  スキーマのメソッド（`points_in_ego` 等）はここの関数を呼ぶ薄いラッパに留める。
- **`tasks.py`**: enum の値は追加のみ可。既存の値の文字列を変更すると
  既存の `agent_config.json` が全て壊れるため、破壊的変更として扱う。
- **`config/` と `schemas/` の分離**: config は Pydantic、schemas は dataclass（4 章）。
  この境界を越えて型を混在させない。
- **テスト**: `tests/` に置き、`src/` にテスト用コードを混入させない。
  namespace package の規約検証（`src/jidohub/__init__.py` の不在）を必ず含める。


---

## 2. 絶対に守る規約

### 2.1 依存ライブラリを増やさない（最重要）

core のランタイム依存は **`numpy` と `pydantic` のみ**を原則とする。

- **`torch` を依存に入れてはならない。** datasets / interfaces / web が torch なしで動くことがプラットフォーム全体の前提。
- `mmcv` / `mmdet3d` / `opencv` / `open3d` / `nuscenes-devkit` も同様に禁止。
- 型ヒントのためだけに重いライブラリを import しない（`TYPE_CHECKING` 下でも避ける）。
- 新しい依存を追加したくなったら、まず「それは本当に core の責務か」を疑う。

### 2.2 namespace package の規約

jidohub は PEP 420 の implicit namespace package として構成する。

- **`src/jidohub/__init__.py` を作成してはならない。** 存在すると名前空間が占有され、
  他の jidohub パッケージが import できなくなる。
- `src/jidohub/core/__init__.py` は通常のパッケージなので**必要**。
- この規約は事故が起きても分かりにくいエラーになるため、
  `src/jidohub/__init__.py` が存在しないことを検証するテストを CI に必ず含める。

### 2.3 スキーマが唯一の正

- 標準スキーマ（`Sample`、各 `*Output`）はプラットフォーム全体の契約であり、
  **core の定義が唯一の正**。他リポジトリで同等の型を再定義しない。
- 型を変更する場合は `schema_version` の更新方針とセットで判断する。
- `schema_version` は現在 **`"0.1"`（破壊的変更を許容する期間）**。
  CenterPoint（Detection）と UniAD（E2E）の入出力を実際に押し込んで歪みがないことを
  確認するまで `"1.0"` にしない。

#### 3 つのバージョンを混同しない

core には**独立に動く 3 つのバージョン**がある。片方に合わせて他方を上げない。
互換判定ロジックを各モジュールに再実装しない（重複は drift の温床）。

| 種類 | 定義場所 | 表すもの |
|---|---|---|
| **スキーマ契約バージョン** | `SCHEMA_VERSION`（`schemas/version.py`） | 標準スキーマ型の互換性。全リポジトリの契約。**定義・互換判定（`assert_compatible`）はこの 1 箇所のみが正** |
| **コンテナ形式バージョン** | `MAGIC` 末尾バイト（`serialization/envelope.py`） | バイナリ framing の互換性。型が変わらなくても framing を変えれば上がる |
| **パッケージバージョン** | `pyproject.toml` | リリース管理。**スキーマ契約バージョンと連動させない** |

- スキーマ互換の判定が必要な箇所（config の検証・直列化の検証など）は
  `schemas.version.assert_compatible` へ**委譲**する。major/minor 比較や形式チェックを各所に複製しない。

### 2.4 用語：Agent と Model を混同しない

| 用語 | 意味 |
|---|---|
| **Agent** | プラットフォーム上の第一級エンティティ。重み・実装・実行環境・メタデータを含むパッケージ全体。`predict(Sample) -> Output` の契約を持つ実行主体 |
| **Model** | Agent 内部のニューラルネットワーク本体（`nn.Module`） |

設定ファイル名は `agent_config.json`、Webのタブ名も Agents で統一する。
コード・docstring・ドキュメントで "model" を Agent の意味で使わない。

---

## 3. 座標系・単位の規約（実装ミスが最も起きやすい箇所）

型シグネチャからは読み取れないが、間違うとパイプライン全体が壊れる。
**新しいフィールドを追加する際は、必ず docstring に座標系・単位・形状を明記すること。**

### 3.1 座標系

- **ego 座標系**: 右手系。**x = 前方、y = 左方、z = 上方**。原点は車両の基準点（nuScenes準拠）。
- **global 座標系**: データセットが定義するマップ座標系。
- 変換行列はすべて **4x4 同次変換行列**（`np.float64`）で表現し、
  **命名は必ず `<from>_to_<to>` の向きを明示する**。
  - `CameraFrame.sensor_to_ego` : センサ座標 → ego 座標
  - `Sample.ego_to_global` : ego 座標 → global 座標
  - 「extrinsic」という曖昧な名前を単体で使わない（向きが読み手に伝わらないため）。

### 3.2 各データの保持座標系

- **点群（`LidarSweep.points` / `RadarSweep`）はセンサ座標系のまま保持する。**
  生データの可逆性を保ち、Adapter 実装を単純にするため。
  ego 座標が必要な場合は同梱の `sensor_to_ego` を使って変換するヘルパを core が提供する。
- **出力（`Box3D` 等）は ego 座標系を既定とする。**
  ただし `Detection3DOutput.frame` フィールドで座標系を必ず明示し、暗黙の前提を作らない。

### 3.3 回転表現

- `Box3D` の回転は **quaternion `(w, x, y, z)` の `np.ndarray` shape (4,)** を正とする。
  nuScenes GT と同形式で情報欠落がないため。
- BEV 用途向けに `yaw`（rad）を**読み取り専用の派生プロパティ**として提供する。
  yaw を正として保持しない。

### 3.4 単位・型・形状

| 対象 | 規約 |
|---|---|
| 距離・寸法 | メートル（m） |
| 速度 | m/s |
| 角度 | ラジアン（rad） |
| 時刻 | UNIX epoch の**マイクロ秒 `int`**（nuScenes準拠） |
| 画像 | `np.uint8`, shape `(H, W, 3)`, **RGB順**（BGRにしない） |
| 点群 | `np.float32`, shape `(N, C)`。先頭3列は必ず x, y, z |
| 変換行列 | `np.float64`, shape `(4, 4)` |
| Box の size | `(length, width, height)` の順（x, y, z 軸方向に対応）。軸順との一致を優先した意図的な設計で、nuScenes（`(width, length, height)` 順）からは意図的に逸脱している。Adapter で必ず入れ替えが必要。個別アクセスは `size[0]` でなく `Box3D.length` / `width` / `height` プロパティを使う |

---

## 4. dataclass と Pydantic の使い分け

用途によって明確に分ける。片方に寄せない。

- **データ面（`Sample`, `CameraFrame`, `Box3D`, `*Output` など）→ 標準 `dataclass`**
  numpy 配列を大量に保持し毎フレーム生成されるため、バリデーションのオーバーヘッドを避ける。
  検証は必要最小限（形状チェック等）に留め、パフォーマンスを優先する。

- **config 面（`agent_config.json` のパース・検証）→ Pydantic**
  ユーザー入力の検証が本質で、エラーメッセージの質が重要。
  JSON Schema を自動生成でき、jidohub-web でのアップロード検証にそのまま再利用できる。

---

## 5. 責務の境界（core に書いてはいけないもの）

以下はすべて core の責務ではない。実装したくなったら該当リポジトリに置く。

- **モデル固有の処理**（前処理・後処理・推論ロジック）→ jidohub-agents
- **データセット固有の読み込み**（nuScenes devkit 呼び出し、ファイルパース）→ jidohub-datasets
- **ROS / シミュレータとの変換**→ jidohub-interfaces
- **評価メトリクスの計算**→ 当面 jidohub-agents 内の evaluate モジュール
- **可視化**（描画・色付け）→ 各アプリケーション（nuscenes-viewer 等）

core が提供するのは「型・規約・Hubアクセス・検証」だけ。
「便利だから」を理由にドメインロジックを core に入れない。

---

## 6. 行動原則

- スキーマ定義の変更は影響範囲が全リポジトリに及ぶ。
  型の追加・変更を提案する際は、**変更理由と `schema_version` への影響を明示する**こと。
- 座標系・単位・形状に関わるコードを書くときは、docstring の規約を先に確認する。
  推測で実装しない。曖昧な場合は実装せずに確認を求める。
- テストは round-trip（シリアライズ→デシリアライズで一致）と、
  バリデータの異常系を重視する。正常系だけのテストを書かない。
- 外部ライブラリの追加、ディレクトリ構成の変更、`schema_version` の更新は
  **自己判断で行わず、必ず確認を取る**。
