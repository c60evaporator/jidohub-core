# 座標変換ヘルパ

出力型の座標変換をどう提供するかを定める。2D（画像平面）と 3D（空間）の双方を扱う。

関連文書: `docs/design/tasks/2d_tasks.md`（画素座標の規約、`ImageSource`）

---

## 1. 解決したい問題

座標変換は**間違えても例外が出ない**種類の処理であり、実装ごとに散らばると事故が蓄積する。

- **3D**: 出力が ego 座標か global 座標かを取り違える。二重に変換する
- **2D**: 前処理の crop / resize / 正規化を経た座標を、元画像に戻す際に取り違える
  （torchvision 系で最も事故が多い箇所）

いずれも「変換を各所で手書きさせない」ことで構造的に防ぐ。

---

## 2. 原則: メタ情報はコンテナが持つ

**座標の解釈に関わるメタ情報（座標系・正規化の有無）は、要素ではなくコンテナが持つ。**

| 型 | メタ情報 |
|---|---|
| `Detection3DOutput` | `frame: CoordinateFrame` |
| `MapOutput` | `frame` |
| `MotionForecastOutput` | `frame`（**要素の `AgentForecast` から移設**。4 章） |
| `PlanningOutput` | `frame`（単一軌跡なので自身がコンテナ） |
| `Detection2DOutput` | `normalized: bool` |
| `InstanceSegmentation2DOutput` | `normalized: bool` |

要素（`Box3D` / `Box2D` / `AgentForecast`）は**持たない**。

理由

- 要素に持たせると、要素間で不整合な状態（`boxes[0]` は ego、`boxes[1]` は global）が
  型として表現可能になる。防ぐには全要素の検証が必要
- コンテナ単位の変換 API が定義できなくなる
- コンテナ側を消して要素側に寄せると、**検出 0 件の出力の座標系が表現できない**
  （空リストは正常な結果である）
- `Instance2D` では破綻する。マスクは常に画素座標（`mask_region` は整数）なので、
  `Box2D.normalized = True` の `Instance2D` は内部で矛盾する

単一要素を変換したい場合は、`geometry` の純関数を明示的に呼ぶ（6 章）。

---

## 3. API の三層構造

```
層 3  出力型のメソッド        利用者の入口。冪等・自己判断
層 2  geometry.py の純関数    変換の実体。手計算で検証できる
層 1  スキーマのメタ情報      frame / normalized / ImageSource
```

層 3 は層 2 への薄い委譲とする（`LidarSweep.points_in_ego()` が
`transform_points` のラッパであるのと同じ方針）。

---

## 4. 3D の変換

### 4.1 `AgentForecast.frame` の移設（破壊的変更）

現状 `frame` は要素の `AgentForecast` にあり、コンテナの `MotionForecastOutput` には無い。
2 章の原則に反するため、**`MotionForecastOutput` へ移す**。

`SCHEMA_VERSION` 0.2 を pre-release として扱えるうちに実施する。

### 4.2 出力型のメソッド

```python
class Detection3DOutput:
    def to_ego(self, ego_to_global: np.ndarray) -> "Detection3DOutput": ...
    def to_global(self, ego_to_global: np.ndarray) -> "Detection3DOutput": ...

class MapOutput:
    def to_ego(self, ego_to_global: np.ndarray) -> "MapOutput": ...
    def to_global(self, ego_to_global: np.ndarray) -> "MapOutput": ...

class MotionForecastOutput:
    def to_ego(self, ego_to_global: np.ndarray) -> "MotionForecastOutput": ...
    def to_global(self, ego_to_global: np.ndarray) -> "MotionForecastOutput": ...

class PlanningOutput:
    def to_ego(self, ego_to_global: np.ndarray) -> "PlanningOutput": ...
    def to_global(self, ego_to_global: np.ndarray) -> "PlanningOutput": ...
```

要件

- **冪等**。既に目的の座標系なら `self` を返す（二重変換を構造的に防ぐ）
- **非破壊**。新しいオブジェクトを返し、入力を変更しない
- 引数は常に `ego_to_global`（`Sample.ego_to_global` をそのまま渡せる）。
  逆変換が必要な場合は内部で `invert_transform` を使う
- `frame` を更新した新しいコンテナを返す（更新漏れが起きない）
- `CoordinateFrame.CAMERA` の出力に対しては `NotImplementedError`。
  カメラ→ego には外部パラメータが必要で、`ego_to_global` だけでは足りない。
  **黙って誤変換しないこと**

### 4.3 変換対象（見落としやすい）

`Detection3DOutput` の変換では、以下を**すべて**変換する。

| フィールド | 変換 |
|---|---|
| `center` | 回転 + 平行移動 |
| `rotation` | 回転のみ（クォータニオンの合成） |
| `velocity` | **回転のみ**（平行移動を適用しない。速度はベクトル量） |
| `size` / `label` / `score` / `track_id` | 不変 |

`velocity` に平行移動を適用する誤りは、静止物体が高速で動いているように見える形で現れる。
テストで固定すること。

`MapElement.points` / `AgentForecast.trajectories` / `PlanningOutput.trajectory` は
位置なので回転 + 平行移動。

---

## 5. 2D の変換

### 5.1 なぜ enum で足りないか

3D の座標系は `EGO` / `GLOBAL` / `CAMERA` という**有限個の名前**で表せるが、
2D は「どの画像に対する座標か」であり、crop 位置とスケールという**連続量**を含む。

```
元画像 1600x900
  → crop (400, 200, 1200, 700) → 800x500
  → モデル入力に resize → 640x640
  → 出力は正規化座標 [0, 1]
```

`frame = "resized"` のような enum では、元画像に戻すのに必要な情報が失われる。
**必要なのは名前ではなく変換そのもの**であり、それは `Image.source`（`ImageSource`）が持つ。

### 5.2 `normalized` フィールド

正規化 `[0, 1]` か画素かの区別は**二値**なのでフィールドで持つ価値がある
（連続量である crop / scale とは性質が異なる）。

```python
@dataclass
class Detection2DOutput:
    boxes: list[Box2D] = field(default_factory=list)
    normalized: bool = False
    """True なら `Box2D.xyxy` は [0, 1] の正規化座標。False（既定）なら画素座標。

    どの画像に対する座標かは、この出力を生んだ `Image` が `source` として保持する。
    ここには持たない（5.1）。
    """
```

`InstanceSegmentation2DOutput` にも同様に持たせる。ただし
**マスクは常に画素座標**であり、`normalized` の対象外（`mask_region` は整数）。
docstring に明記すること。

Agent 側で必ず画素に戻させる案（core は画素のみを認める）も検討したが、
正規化出力のモデルが多く、変換忘れが**型に現れない**状態を作るため採用しない。

### 5.3 出力型のメソッド

```python
class Detection2DOutput:
    def to_source_image(self, image: Image) -> "Detection2DOutput": ...

class InstanceSegmentation2DOutput:
    def to_source_image(self, image: Image) -> "InstanceSegmentation2DOutput": ...
```

`image` は**推論に使った `Image`**。1 回の呼び出しで以下を完了する。

1. `normalized` なら `image` のサイズで画素座標へ戻す
2. `image.source` の `scale` と `crop` を逆適用し、元画像基準の座標にする
3. `normalized=False` の新しい出力を返す

要件

- **冪等**。`image.source` が `None` なら正規化解除のみ。既に元画像基準なら実質 no-op
- **非破壊**
- `InstanceSegmentation2DOutput` では **`mask` と `mask_region` も元画像基準へ移す**。
  `scale` を伴う場合はマスクを**最近傍リサイズ**（`resize_mask_nearest`）で移す。
  bool マスクに適用できる補間は**最近傍のみ**であり（bilinear で補間すると bool でなくなり
  再度しきい値処理が必要になる）、最近傍リサイズは**純 numpy のインデックス操作で書ける**ため
  画像処理依存は生じない。品質面では、Agent 側で**二値化の前**に float（logit）マスクを
  入力解像度へ補間するのが本来の経路であり、ここはそれを経ていない bool 出力を扱う
  フォールバックである。
  処理順序が重要で、**先に整数の目標領域を確定してからそのサイズへマスクを合わせる**
  （逆順にすると丸めで 1 画素ずれ、`Instance2D` の「`mask.shape` と領域サイズが一致」検証に落ちる）。

利用者のコード

```python
crop = frame.image.cropped(400, 200, 1200, 700)
output = agent.predict(ImageSample(image=crop))
output_full = output.to_source_image(crop)   # 元画像座標に戻る
```

crop 位置・スケール・正規化の有無を利用者が覚える必要がなくなる。

### 5.4 前提: `Image.source` が必ず埋まっていること

`cropped()` は `source` を自動更新するが、**resize は core の外**（agents 側）である。
そこで `source` の更新が漏れると全体が破綻する。

core は更新用の純関数を提供する。

```python
def scaled_source(source: ImageSource | None, scale_x: float, scale_y: float) -> ImageSource
```

agents の Processor は「画素の resize」「`scale_intrinsic`」「`scaled_source`」を
**3 点セットで行う**。この規約を `2d_tasks.md` に記載し、
将来は共通テストで縛ることも検討する。

---

## 6. `geometry.py` の純関数

層 3 のメソッドはすべてここへ委譲する。**手計算で検証できるテストを必ず書く。**

```python
# 2D
def denormalize_boxes(xyxy, width, height) -> np.ndarray      # [0,1] → 画素
def normalize_boxes(xyxy, width, height) -> np.ndarray        # 画素 → [0,1]
def boxes_to_source(xyxy, source) -> np.ndarray               # 現画像 → 元画像
def boxes_from_source(xyxy, source) -> np.ndarray             # 元画像 → 現画像
def points_to_source(points, source) -> np.ndarray            # プロンプト座標用
def points_from_source(points, source) -> np.ndarray
def scaled_source(source, scale_x, scale_y) -> ImageSource
def resize_mask_nearest(mask, height, width) -> np.ndarray    # bool マスクの最近傍リサイズ

# 3D
def transform_boxes(...)          # center / rotation / velocity をまとめて変換
def rotate_vectors(vectors, R)    # 速度など、平行移動を適用しない量
```

いずれも入力を破壊せず、新しい配列を返す。

**単一要素（`Box2D` / `Box3D` 単体）を変換する高レベル API は提供しない。**
座標系はコンテナが持つという原則（2 章）を構造的に守るため。
低レベルで扱いたい場合はこれらの純関数を直接呼び、画像サイズや変換行列を明示的に渡す。

---

## 7. テストの要点

以下はいずれも**取り違えても例外が出ない**ため、機械的な検証が必要。

- `to_ego()` / `to_global()` の**往復が恒等**であること
- **冪等性**: 同じ変換を 2 回適用しても結果が変わらないこと
- **`velocity` に平行移動が適用されないこと**（静止物体が動いて見える誤りの検出）
- `to_source_image()` の往復（`cropped()` した画像で推論した結果を戻すと、
  元画像座標の既知の位置に一致すること）
- **正規化解除とスケール逆適用の順序**が正しいこと
  （順序を誤ると crop 原点の扱いがずれる）
- `CoordinateFrame.CAMERA` に対する `to_ego()` が `NotImplementedError`
- `scale` を伴うマスクの `to_source_image()` で、拡大・縮小・非整数倍率のいずれでも
  `mask.shape` と `mask_region` のサイズが常に一致すること
- マスクの**貼り戻し往復**（元画像の既知位置に置いたマスクを変換 → `Instance2D.paste()` で
  全画面に戻すと元の位置に一致すること。整数 scale で最近傍が厳密に可逆になるよう構成する）
