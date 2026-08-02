# 2D タスクの設計

`object_detection_2d` / `instance_segmentation_2d` / `image_classification_2d` の
core への実装にあたって確定した設計判断を記録する。

CLAUDE.md が「恒久的な規約」であるのに対し、本書は「なぜその形にしたか」の記録である。
実装後も、判断の背景を辿るために残す。

対象バージョン: `SCHEMA_VERSION` **0.1 → 0.2**（破壊的変更を含む）

---

## 1. タスク定義の原則

**タスク = 出力データと評価指標が等しい Agent の集合。**

同一指標での比較を容易にすることが目的なので、入力の差異でタスクを分けない。
入力のバリエーションは **`agent_config.json` の宣言**で表す。

この原則により、以下は**同一タスク**に統合される。

| 統合されるもの | タスク | 差異の表現 |
|---|---|---|
| 閉集合検出 / オープン語彙検出（Grounding DINO） | `object_detection_2d` | プロンプト宣言 |
| 全インスタンス分割 / プロンプタブル分割（SAM2, SAM3） | `instance_segmentation_2d` | プロンプト宣言 |
| 閉集合分類 / ゼロショット分類（SigLIP2） | `image_classification_2d` | プロンプト宣言 |

この「1 タスク + config での能力宣言」というパターンは、将来も踏襲する
（例: 相対深度と絶対深度を `depth_estimation` に統合し、`metric_depth` を宣言で区別する）。

### 帰結: 出力はクラス名を文字列で持つ

ゼロショット分類・オープン語彙検出ではラベル集合が実行時に決まるため、
クラスインデックスを出力にできない。`Box3D.label` と同様、**すべて文字列**とする。

---

## 2. 命名規約

`TaskType` の値は変更が破壊的（既存の `agent_config.json` が全て壊れる）なため、
規約を先に固定する。

- 形式は `<動作>_<次元>` の snake_case。**次元サフィックスは必ず末尾**
- サフィックスは、**画像平面に閉じる**タスクに `_2d`、
  **3D 空間座標を扱う**タスクに `_3d` を付ける
- どちらにも該当しない、または区別が意味を持たないタスクには**付けない**
  （`map_construction`, `depth_estimation`, `motion_forecasting`,
  `occupancy_prediction`, `multi_view_reconstruction`, `video_text_to_text`,
  `sensing_to_planning`, `vision_language_action`, `track_map_to_planning`, `control`）

例: `instance_segmentation_2d_tracking` は規約違反。
`instance_segmentation_tracking_2d` とする。

---

## 3. 画像の表現

### 3.1 `Image` の新設と `CameraFrame` の再構成

2D タスクの入力は「カメラ 1 台の 1 フレーム」ではなく「画像 1 枚」であり、
crop された領域や外部から与えられた画像も対象になる。
そこで画像そのものを表す型を分離する。

```python
# schemas/image.py
@dataclass
class EncodedPixels:        # 旧 EncodedImage
    data: np.ndarray        # (N,) uint8。ファイルそのもののバイト列
    format: ImageFormat
    height: int
    width: int

@dataclass
class ImageSource:
    """この画像の由来。crop / resize を経ても元画像へ写像できるようにする。"""
    channel: str | None = None
    crop: tuple[int, int, int, int] | None = None   # 元画像上の (x0, y0, x1, y1)
    scale: tuple[float, float] | None = None        # crop 後 → 現サイズ

@dataclass
class Image:
    """画像 1 枚。pixels か encoded のどちらか一方を保持する（排他）。"""
    pixels: np.ndarray | None = None
    encoded: EncodedPixels | None = None
    intrinsic: np.ndarray | None = None     # (3, 3)。未知なら None
    distortion: np.ndarray | None = None
    source: ImageSource | None = None

    @property
    def array(self) -> np.ndarray: ...      # 常に (H, W, 3) uint8 RGB。遅延デコード + キャッシュ
    @property
    def height(self) -> int: ...            # デコードせずに取得
    @property
    def width(self) -> int: ...
    @property
    def is_encoded(self) -> bool: ...
    def cropped(self, x0, y0, x1, y1) -> "Image": ...   # intrinsic と source を自動更新

@dataclass
class CameraFrame:
    image: Image
    sensor_to_ego: np.ndarray               # (4, 4) sensor → ego
    channel: str
    timestamp: int | None = None
```

判断の理由

- **`EncodedPixels` を入力型にしない。** これは 2 表現のうちの片方であって画像そのものではない。
  入力型にすると、生画素を持つ呼び出し側が符号化してから渡すことになり
  （JPEG なら劣化も乗る）、「デコードコストはどちらの設計でも 1 回」という前提が崩れる。
  排他の吸収は `Image` が担い、利用側は `image.array` だけを見る
- **`intrinsic` の住所は `Image` 一箇所。** `CameraFrame` にも持たせると二重の正になる。
  `CameraFrame.__post_init__` では `image.intrinsic is not None` の検証のみ行う
- **`sensor_to_ego` は `Image` に入れない。** 外部パラメータは「カメラの取り付け」の情報であり、
  crop された画像片には付随しない。2D タスクは ego 座標を必要としない
- **改名の理由**: `Image` 導入後、`EncodedImage` は名前上 `Image` の一種に見えるが、
  実際は `Image` の一表現にすぎず is-a が成立しない。`EncodedPixels` なら
  `pixels` / `encoded` が対称になる

### 3.2 画素座標の規約

- **原点は左上**、x が右方向、y が下方向
- 単位は画素。**サブピクセルを許す float**（ボックスは float、マスクは整数格子）
- 基準は**その `Image` の現サイズ**。crop 後の `Image` なら crop 後の座標系
- **プロンプト座標も出力座標もすべてこの規約に従う**
- 元画像の座標へ戻す必要がある場合は `Image.source` を使う

`ImageSource` は連鎖（リンクリスト）ではなく、**元画像に対する実効的な crop + scale を
1 組に合成**して保持する。crop の crop でも深さが伸びず、逆写像が 1 回で済む。

### 3.3 intrinsic の crop / scale 追従

crop 時に intrinsic を更新する手段を core が提供しない場合、パイプライン側は
`intrinsic=None` で逃げるか、**親の intrinsic をそのままコピーする**。
後者は誤りだが実行時には通るため最も危険である。純関数として提供する。

```python
# geometry.py
def crop_intrinsic(K, x0, y0) -> np.ndarray: ...    # 主点を平行移動
def scale_intrinsic(K, sx, sy) -> np.ndarray: ...   # 焦点距離と主点をスケール
```

`Image.cropped()` はこれらを呼ぶ薄いラッパとする
（`LidarSweep.points_in_ego()` を `geometry` のラッパに留めたのと同じ方針）。

**`resize` は core に置かない。** 実際のリサンプリングは画像処理ライブラリへの依存であり、
コーデック非依存の原則と衝突する。core は `scale_intrinsic` のみを提供し、
画素の変換と組み合わせるのは agents 側の責務とする。

**`distortion` は crop / scale で変化しない。** OpenCV の歪み係数は正規化座標
（`(u - cx) / fx`）に対して定義されるため、`K` を正しく更新している限り係数は不変。
これを知らないと「crop したから歪み係数も補正しなければ」という誤った処理を入れがちなので、
`Image.cropped()` の docstring に明記する。

---

## 4. 入力型

```python
# schemas/prompts.py
@dataclass
class ImagePrompt:
    """2D タスクのオプション入力。指定された項目のみが有効。"""
    points: np.ndarray | None = None        # (P, 2) 画像座標
    point_labels: np.ndarray | None = None  # (P,) 1=前景, 0=背景
    boxes: np.ndarray | None = None         # (B, 4) x0, y0, x1, y1
    texts: list[str] | None = None          # テキストプロンプト / ラベル候補

@dataclass
class ImageSample:
    """2D タスクの入力。`Sample`（センサ入力）と対になる位置づけ。"""
    image: Image
    prompt: ImagePrompt | None = None
    metadata: dict = field(default_factory=dict)
```

- **`texts` は分類のラベル候補と検出のテキストプロンプトで共用する。**
  意味的に同じもの（対象を言語で指定する）であり、分けると型が増えるだけ
- **入力は単一引数に保つ。** `predict(input) -> output` の契約を崩さないため、
  画像とプロンプトを `ImageSample` にまとめる
- 多フレーム入力（tracking, video depth）は `list[ImageSample]` とする（9 章）

---

## 5. プロンプトの宣言（`agent_config.json`）

入力をオプションにすると「この Agent はプロンプトが必要か / 何を受け付けるか」が
型から読めなくなる。`sensors` と同じ形で宣言させる。

```json
"prompt": {
  "required": true,
  "supported": ["point", "box", "text"]
}
```

- Web の検索フィルタ（「テキストプロンプト対応の検出器」）に使う
- 実行前の検証に使う。宣言がないと、実行して初めて失敗するか、
  **黙って空の結果を返す**ことになる
- `supported` にない種別が指定された場合は Runner がエラーにする

---

## 6. 出力型

```python
# schemas/outputs.py
@dataclass
class Box2D:
    xyxy: np.ndarray            # (4,) float64。(x0, y0, x1, y1)
    label: str | None = None    # プロンプタブル系では None になり得る
    score: float | None = None
    track_id: int | None = None # tracking タスクで使用（9 章）

@dataclass
class Instance2D:
    box: Box2D
    mask: np.ndarray | None = None                      # (h, w) bool
    mask_region: tuple[int, int, int, int] | None = None # (x0, y0, x1, y1) int

@dataclass
class Detection2DOutput:
    boxes: list[Box2D]

@dataclass
class InstanceSegmentation2DOutput:
    instances: list[Instance2D]

@dataclass
class Classification2DOutput:
    labels: list[str]           # スコア降順
    scores: np.ndarray          # (K,) float64
```

### 6.1 ボックス形式

`(x0, y0, x1, y1)` の float。フィールド名を `xyxy` にすることで、
xywh との取り違えをコード上で防ぐ（`Box3D.size` の並び順と同じ発想）。
COCO 形式（xywh）への変換は評価層の責務であり、core では扱わない。

### 6.2 マスク表現: bbox + crop 内マスク

素朴な `(N, H, W)` の全画面マスクは、1600x900 で 50 インスタンスなら 72MB になる。
プロセス境界を越える設計では実用にならないため、**ボックス内に限定したマスク**を持つ。

- `mask` は `(h, w)` の bool 配列で、`mask_region` が示す整数画素領域を覆う
- `mask_region` は `(x0, y0, x1, y1)` の**整数**で、`x1 - x0 == w`、`y1 - y0 == h`
  を `__post_init__` で検証する
- `box.xyxy`（float）とは別に整数領域を持つのは、**丸め規則の曖昧さを排除する**ため。
  float から暗黙に丸めると、実装ごとに 1 画素ずれる
- RLE（COCO 形式）への変換は評価層に置く。純 numpy で実装でき、core の責務ではない

### 6.3 出力の座標系

2D 出力の座標は、**入力 `Image` の現サイズ基準**（3.2 の規約）。
crop された画像に対する推論結果は crop 後の座標であり、元画像へ戻すには
`Image.source` を用いる。この写像は core が提供する（`ImageSource` の逆変換）。

### 6.4 `CoordinateFrame.CAMERA` の追加

単眼深度の出力は ego 座標に置けない（外部パラメータが必要で、それを持つのは
`CameraFrame` を握っている呼び出し側）。将来 `DepthOutput` を追加する際に必要になるため、
`TaskType` の棚卸しと同時に enum 値を追加しておく（後から追加すると
`schema_version` を上げる変更になる）。

---

## 7. 評価プロトコル（jidohub-server 側の設計メモ）

**タスクを統合したことで、評価条件の記録が必須になった。**

SAM2 に GT ボックスをプロンプトとして与えて測った mask AP と、
Mask R-CNN の mask AP を並べると、前者は正解情報を受け取っているため
**比較が成立しない**。同じ指標で測れることと、比較可能であることは別である。

評価ランの「評価設定」に以下を含め、**設定が一致するラン同士のみ比較を許す**。

- プロンプトの有無
- プロンプトの種別（point / box / text）
- **プロンプトの供給源**（GT 由来 / 別モデルの出力 / 人手）

供給源が最も重要で、GT 由来のプロンプトを使ったランと使わないランが同じ表に並ぶと、
リーダーボードが無意味になる。

**タスクは 1 つ、評価プロトコルは複数**という構造になる。これは正しい姿であり、
SAM 系の論文も「GT box prompt」「everything mode」を別条件として報告している。

---

## 8. 移行

### 8.1 `SCHEMA_VERSION` を 0.2 に上げる

`CameraFrame` が破壊的に変わるため必須。0.x は minor 差も非互換なので、
**core を上げた瞬間に jidohub-datasets の CI が落ちる**。これは想定内であり、
「片方が古い」を検出する仕組みが働いた証拠と見なす。

### 8.2 追随作業

| 対象 | 内容 |
|---|---|
| core | `TYPE_REGISTRY` に新型を登録。`schemas.__all__` の網羅性テスト更新 |
| core | round-trip テスト（マスク配列・プロンプト） |
| core | `crop_intrinsic` の手計算テスト、`distortion` 不変のテスト |
| core | config の異常系（プロンプト宣言、`supported` 外の指定） |
| datasets | `adapter.py` の `_build_camera_frame` を `Image` 構築に変更 |
| datasets | `tests/test_adapter.py` の `CameraFrame(...)` 呼び出し |
| datasets | **`scripts/smoke_real_data.py` の再実行（31 項目）** — ここが通って移行完了 |
| datasets | 生成済み fixture の再生成（直列化形式が変わるため） |

`CameraFrame.array` のようなショートカットは**作らない**。
書き方が二通りになるため、examples とテストは `frame.image.array` に統一する。

---

## 9. 将来のための予約（今回は実装しない）

中期のタスク構成を踏まえ、**後から変更すると破壊的になる事項**のみ先に決めておく。

### 9.1 `Box2D.track_id` を最初から持たせる

`object_tracking_2d` を追加する際に `Box2D` へフィールドを足すと破壊的変更になる。
`Box3D` が既に `track_id` を持つのと同じ形で、最初から用意する。
単発検出では `None`。

### 9.2 多フレーム入力は `list[ImageSample]`

tracking、video depth、VGGT、`video_text_to_text` はいずれも複数フレームを取る。
専用のコンテナ型を作らず、`list[ImageSample]` とする。
新しい型を増やさず、単一フレームの規約をそのまま適用できるため。

### 9.3 tracking / 時系列モデルの契約（決定済み）

**詳細は `docs/design/streaming_agents.md` を参照。** 2D タスクに関わる部分のみ要約する。

`predict(list[ImageSample])` 方式は採用しない。detection-based tracker は毎フレーム
履歴を渡し直すことになり、系列モデル（UniAD 等）はオンライン動作が原理的に不可能になる
（系列全体が揃うまで呼べない）。実車・ROS 接続を担う jidohub-interfaces の存在を
考えると許容できない。

採用する契約は `reset()` / `step(input)` と、それをループする既定の `predict()` である。
Agent 作者は前者 2 つのみを実装し、オンライン推論とオフライン一括評価の双方を
同じ実装から得る。この機構は tracking 専用ではなくタスク横断の能力として定義する
（`sensing_to_planning` の時系列モデルも同じ機構を使う）。

2D タスクへの帰結

- **プロンプトは `reset()` ではなく `step()` の入力で受ける。** SAM2 の video
  segmentation は系列途中でのプロンプト追加が中核機能であり、`ImageSample.prompt`
  がそのまま使える。`reset()` は引数を取らない
- 多フレーム入力の `list[ImageSample]`（9.2）は、**ストリーミングしないタスク**
  （video depth、VGGT、VLM など系列全体を一度に見るもの）に限定して用いる
- `instance_segmentation_tracking_2d` の 1 フレーム分の出力は
  `InstanceSegmentation2DOutput`（`track_id` が埋まる）とし、
  系列の出力型はそれを時刻順に保持する薄いラッパとする

### 9.4 E2E の中間出力は将来、単体タスクの出力型に置き換える

`E2EOutput.occupancy` は現在、生の `np.ndarray` + メタ情報 dict である。
`occupancy_prediction` タスクを追加して `OccupancyOutput` を定義した時点で、
このフィールドをその型に置き換える（中間出力は単体タスクの出力型を再利用する、
という既存の方針に合わせる）。これは破壊的変更になるため、
`SCHEMA_VERSION` を上げるタイミングでまとめて行う。

### 9.5 `depth_estimation` の相対 / 絶対は capability で分ける

相対深度（Depth Anything 系）と絶対深度（Metric3D 系）は評価指標が異なるが、
出力型は同じ（深度マップ）である。1 章の原則に従い、タスクは 1 つとし、
`agent_config.json` で `metric_depth` を宣言させる。
評価側は宣言を見て指標を選ぶ（7 章と同じ「1 タスク・複数プロトコル」の構造）。

### 9.6 `InputKind` の予約タスク分類は暫定（実装時に見直す）

`TASK_INPUT_KINDS` の予約タスク分類は、現時点では**検証ルール（センサ要求・プロンプト宣言）が
正しく働くこと**を基準に決めており、入力 dataclass の厳密な対応ではない。

- 2D tracking 系を `IMAGE` としているが、`streaming_agents.md` 4 章の入力型は
  `Tracking2DInput`（`ImageSample` + 上流検出）であり意味論的には複合入力
- `vision_language_action` は `Sample` + テキストを取るため `COMPOSITE` とする
  （`Sample` に prompt フィールドが無く、`SENSOR` だとテキスト宣言が拒否されるため）

ストリーミング契約と複合入力型を実装する段階で、`InputKind` を細分化するか、
検証を入力種別とは別のフラグで駆動するかを判断する。
