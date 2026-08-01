# ストリーミング Agent の契約

tracking や時系列モデルのように、**状態を持ちフレーム単位で推論する Agent** の契約を定める。

本書はタスク横断の設計であり、`object_tracking_2d` / `object_tracking_3d` /
`instance_segmentation_tracking_2d` に加え、**`sensing_to_planning` の時系列モデル**
（UniAD, SparseDrive 等）も対象とする。

関連文書: `docs/design/tasks/2d_tasks.md`（2D タスクの入出力型）

---

## 1. 対象となるモデルの分類

時系列を扱うモデルは、必要とする入力の性質によって 3 つに分かれる。

| 種類 | 入力 | 代表モデル |
|---|---|---|
| **A. Detection-based Tracker** | 現フレームの検出結果 + 過去状態 | AB3DMOT, SimpleTrack |
| **B. Sliding Window 型** | 直近 N フレームのセンサデータ | CenterPoint Tracking, BEVFormer Tracking, PETRv2 |
| **C. Sequence Model 型** | 長い時系列のセンサデータ（内部 memory） | UniAD, SparseDrive |

---

## 2. `list[...]` 入力を採用しない理由

`predict(list[Sample]) -> Tracking3DOutput` のように系列全体を受け取る契約は、
一見単純だが 2 つの型を壊す。

- **A**: 必要なのは現フレームの検出結果と内部状態だけなのに、
  毎フレーム履歴を渡し直すことになる。計算量とメモリが系列長に比例して無駄に増える
- **C**: **オンライン動作が原理的に不可能**になる。系列全体が揃うまで `predict` を呼べず、
  「今のフレームに対する出力を今返す」ことができない

実車・シミュレーションとの接続を担う jidohub-interfaces が存在する以上、
オンライン動作が不可能な契約は採用できない。

---

## 3. 契約: `reset()` / `step()` と既定の `predict()`

状態を持つことをタスク横断の**能力**として定義する。tracking 専用にしない
（`sensing_to_planning` の時系列モデルも同じ機構を使うため）。

```python
class StreamingMixin:
    """状態を持ち、フレーム単位で推論する Agent の契約。タスク種別に依存しない。

    Agent 作者が実装するのは reset() と step() のみ。
    predict() は基底クラスが既定実装を提供する。
    """

    def reset(self) -> None:
        """内部状態を初期化する。引数は取らない（7 章）。

        何度呼んでもよい（冪等）。track_id の採番もここでリセットする。
        """

    def step(self, input):
        """1 フレーム分の入力に対する出力を返す。

        呼び出しごとに内部状態を更新する。**reset() を呼ばずに呼ぶとエラー**（6 章）。
        """

    def predict(self, inputs: list):
        """系列全体をまとめて処理する（オフライン評価用の既定実装）。

        reset() してから step() をループし、各フレームの出力を集約して返す。
        **Agent 作者はこのメソッドを実装しない。**
        """
```

役割分担が自然に決まる。

- **jidohub-interfaces（実車・シミュレーション）** → `reset()` と `step()` を呼ぶ
- **評価ハーネス・jidohub-server（オフライン一括）** → `predict()` を呼ぶ

**1 つの実装からオンライン推論とオフライン評価の両方が出る**ことが、この設計の要点である。

### 3.1 検証可能な不変条件

`predict(inputs)` の結果は、`reset()` してから `step()` を手動でループした結果と
**一致しなければならない**。これはテストで機械的に検証できるため、
`StreamingMixin` を実装する Agent の共通テストに必ず含める。

一致しない場合、既定 `predict()` を上書きしているか、`step()` が
入力以外の状態（グローバル変数など）に依存している。

---

## 4. 入力型: 複合入力は型にまとめる

`step(sample, detections)` のように引数を複数取ってはならない。
`predict(input) -> output` の単一引数契約は、直列化と Runner の前提である
（`pack(input)` → RPC → `unpack` → 実行という経路が、引数が増えると
転送層での特別扱いを要求する）。

**上流タスクの出力を入力に取るタスクは、複合入力型を定義する。**

```python
@dataclass
class Tracking3DInput:
    sample: Sample
    detections: Detection3DOutput | None = None   # A（detection-based）で使用

@dataclass
class Tracking2DInput:
    image_sample: ImageSample
    detections: Detection2DOutput | None = None
```

命名規約は `<TaskName>Input`。この原則は tracking に限らず、
`track_map_to_planning`（`Detection3DOutput` + `MapOutput` + `EgoState` を取る）
にも同じ形で適用する。

---

## 5. 3 パターンの実装と config 宣言

契約は同一で、差異は `step()` の内部実装と `agent_config.json` の宣言に閉じる。

| | `step()` の実装 | config 宣言 |
|---|---|---|
| A | 内部 Kalman フィルタを更新 | `sensors: {}`, `requires: ["object_detection_3d"]`, `temporal.mode: "streaming"` |
| B | 内部リングバッファに積み、N フレームで推論 | `sensors: {...}`, `temporal.mode: "window"`, `temporal.window: N` |
| C | 内部 query / memory を更新 | `sensors: {...}`, `temporal.mode: "streaming"` |

```json
"temporal": {
  "mode": "streaming",
  "window": null
},
"requires": ["object_detection_3d"]
```

- `temporal.mode`: `"none"`（状態を持たない）/ `"window"` / `"streaming"`
- `temporal.window`: `mode == "window"` のときのフレーム数
- `requires`: 入力として必要な**上流タスクの出力**。空なら生センサのみで動作する

### 5.1 既存の検証ルールとの衝突（要対応）

現在の `SensorRequirement` は「カメラ・LiDAR・RADAR のいずれかが 1 つ以上」を要求する。
**A（detection-based tracker）はセンサを一切必要としない**ため、この検証に引っかかる。

検証ルールを次のように緩和すること。

> `sensors` が空でよいのは、`requires` が空でない場合のみ。
> 両方が空の Agent は入力を持たないため、引き続きエラーとする。

これは 2D タスク（`sensors` を空にして画像のみを入力とする）の扱いとも整合させる必要がある。
`docs/design/tasks/2d_tasks.md` の入力型の議論と合わせて実装すること。

---

## 6. 未 `reset()` 状態を作らない

`reset()` を呼ばずに `step()` を呼べる状態は、**前のシーンの状態が漏れる**という
静かな不具合を生む。track_id が引き継がれ、評価結果が汚染される。

基底クラスで初期化フラグを持ち、未 reset での `step()` は明示的なエラーとする。

```python
def step(self, input):
    if not self._initialized:
        raise RuntimeError(
            "reset() must be called before step(). "
            "Streaming agents keep internal state across frames."
        )
```

**評価時はシーンごとに `reset()` する。** 既定の `predict()` は先頭で `reset()` を
呼ぶため自動的に満たされるが、`step()` を直接使う経路（interfaces、
カスタム評価ループ）では呼び出し側の責務になる。

### 6.1 track_id のスコープ

`track_id` の一意性は **`reset()` から次の `reset()` までのセッション内**に限る。
セッションをまたいで同じ整数が別の物体を指してよい。
（nuScenes Adapter が `instance_token` を Adapter インスタンス全体で一貫させているのとは
スコープが異なるため、混同しないこと。）

---

## 7. プロンプトは `step()` の入力で受ける

`reset(prompts=...)` の形は採用しない。

SAM2 の video segmentation は、**系列の途中でプロンプトを追加できること**が
中核機能である（10 フレーム目でクリックして対象を追加する）。
`reset()` でしかプロンプトを渡せないと、t=0 のプロンプトしか表現できない。

`ImageSample` が既に `prompt` フィールドを持つため、`step(Tracking2DInput)` が
そのまま途中プロンプトを表現できる。`reset()` は引数を取らない。

---

## 8. 出力型

`step()` は**1 フレーム分の出力**を返し、`predict()` はそれを時刻順に集約した
**系列の出力**を返す。系列の出力型は、単体タスクの出力型を保持する薄いラッパとする
（中間出力を単体タスクの出力型で表す既存方針と同じ）。

```python
@dataclass
class Tracking3DOutput:
    frames: list[Detection3DOutput]        # 時刻昇順。各要素の track_id が埋まる
    timestamps: np.ndarray | None = None   # (T,) int64 μs。入力から引き写す

@dataclass
class InstanceSegmentation2DTrackingOutput:
    frames: list[InstanceSegmentation2DOutput]
    timestamps: np.ndarray | None = None
```

薄いラッパにすることで、**可視化コードを単体タスクのものから流用できる**
（1 フレーム取り出せば既存の描画がそのまま使える）。

`step()` の戻り値の型は `frames` の要素型と一致させる。この対応が崩れると、
既定 `predict()` の集約が書けなくなる。

---

## 9. 評価プロトコル（jidohub-server 側）

### 9.1 上流入力の供給源を評価設定に含める

A（detection-based tracker）は上流の検出結果を入力に取る。
**GT の検出結果を与えたトラッカーと、実モデルの検出を与えたトラッカーの MOTA を
並べても比較にならない。** nuScenes tracking challenge も検出器を固定した条件で比較している。

評価ランの「評価設定」に以下を含め、一致するラン同士のみ比較を許す。

- `requires` で指定された上流出力の**供給源**（GT 由来 / 特定モデルの出力 / 人手）
- モデル由来の場合は、その Agent の `repo_id` と `revision`

これは 2D タスクの**プロンプトの供給源**と同じルールであり、実装を共通化する
（`docs/design/tasks/2d_tasks.md` 7 章）。

### 9.2 セッション境界の記録

ストリーミング Agent の出力は**フレームの順序と reset のタイミングに依存する**。
評価設定に「シーンごとに reset した」ことを含め、再現性を担保する。

---

## 10. 運用上の帰結

ステートレス前提だった従来の設計からの変更点であり、jidohub-server の実装に影響する。

### 10.1 セッション管理が必要になる

docker runner は、**同一ストリームの連続フレームを同じコンテナインスタンスに
ルーティング**しなければならない。以下が必要になる。

- セッション ID の発行と、リクエストからインスタンスへの対応付け
- アイドルタイムアウトによるセッション破棄（状態がメモリを占有し続けるため）
- セッション終了時の `reset()` またはインスタンス破棄

### 10.2 インスタンスを共有できない

ステートフル Agent のインスタンスは、**複数ストリームで同時に使えない**。
ステートレス Agent なら 1 インスタンスで複数リクエストを捌けるが、
ストリーミング Agent は同時ストリーム数だけインスタンスが必要になる。
GPU メモリの見積もりに直結するため、`temporal.mode` を見て
スケジューリングを変える必要がある。

### 10.3 interfaces 側の呼び出し

ROS ノードは、シナリオ開始時に `reset()`、各メッセージで `step()` を呼ぶ。
`predict()` は使わない。

---

## 11. 未決事項（実装前に決める）

- **状態のシリアライズ**。コンテナが落ちた際のセッション復旧や、
  評価の途中再開のために `get_state()` / `set_state()` を契約に含めるか。
  当面は不要と判断するが、必要になった時点で `StreamingMixin` の拡張として追加する
  （既存の `reset` / `step` を壊さない形で足せる）
- **バッチストリーミング**。複数ストリームを 1 インスタンスで並列に扱う
  （`step()` が batch 次元を取る）方式。GPU 効率のために将来必要になり得るが、
  契約が大きく変わるため、実測で必要性が確認されるまで採用しない
