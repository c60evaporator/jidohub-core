# jidohub-core

自動運転向けプラットフォーム **jidohub** の共通基盤ライブラリ。
標準スキーマ・Hub クライアント・config パーサ／バリデータ・直列化を提供する。

jidohub は 5 リポジトリ構成で、**core は他の 4 つすべてが依存する共通基盤**。
依存は常に `core ← 他` の一方向（星形依存）で、**core は他の jidohub パッケージに一切依存しない**。

```
              ┌──────────────┐
              │  jidohub-web │
              └──────┬───────┘
                     │
   jidohub-agents ───┤
                     ├──────▶  jidohub-core  (本リポジトリ)
 jidohub-datasets ───┤        標準スキーマ / Hub / config / 直列化
                     │
jidohub-interfaces ──┘
```

| リポジトリ | 役割 |
|---|---|
| jidohub-web | Agents / Datasets / Interfaces をホストする Web プラットフォーム |
| **jidohub-core** | 標準スキーマ・Hub クライアント・config パーサ |
| jidohub-agents | Agent をロードして実行する Python API |
| jidohub-datasets | Dataset をロードして標準スキーマに変換する Python API |
| jidohub-interfaces | 実車・シミュレーションとの入出力変換 |

## インストール

```bash
pip install jidohub-core          # numpy / pydantic のみに依存
pip install 'jidohub-core[s3]'    # s3:// 参照を使う場合（boto3 を追加）
```

Python 3.10 以上が必要。ランタイム依存は **numpy と pydantic のみ**（`boto3` は optional）。

## 使い方

### Sample の構築と直列化（pack / unpack）

```python
import numpy as np
from jidohub.core.schemas import Sample, LidarSweep
from jidohub.core.serialization import pack, unpack

sample = Sample(
    timestamp=1_600_000_000_000_000,  # UNIX epoch のマイクロ秒（int）
    ego_to_global=np.eye(4),  # 4x4 同次変換行列
    lidar=LidarSweep(
        points=np.random.rand(1000, 4).astype(np.float32),  # (N, C>=3), 先頭3列 = x,y,z
        sensor_to_ego=np.eye(4),
    ),
)

blob = pack(sample)  # JSON ヘッダ + 生バッファの自己記述コンテナ（bytes）
restored = unpack(blob)  # 既定はゼロコピー = 読み取り専用ビュー
restored = unpack(blob, copy=True)  # 書き込みが必要な場合はコピー
```

### agent_config.json の読み込みと検証

```python
from jidohub.core.config import load_agent_config, validate_strict

config = load_agent_config("examples/centerpoint_agent_config.json")
print(config.agent_id, config.task.value)

# 公開前の厳格チェック（例外ではなく違反理由のリストを返す）
violations = validate_strict(config, repo_path="path/to/agent-repo")
```

### Hub からの取得

```python
from jidohub.core.hub import HubClient

client = HubClient()
# ローカルディレクトリ・s3:// URI・Agent ID（<namespace>/<name>）を受け付ける。
snapshot = client.snapshot("s3://my-bucket/agents/my-agent")
config, path = client.load_config(snapshot)
client.verify_weights(config, path)  # sha256 でチェックサム検証
```

座標系・単位・配列形状の規約は `CLAUDE.md` の 3 章を参照。

## スキーマバージョン

`schema_version` は現在 **`"0.1"`（破壊的変更を許容する期間）**。
CenterPoint（Detection）と UniAD（E2E）の入出力を実際に押し込んで歪みがないことを
確認するまで `"1.0"` にしない。0.x の間は minor 差も非互換として扱う。

## 開発

```bash
pip install -e '.[dev,s3]'
pytest            # テスト
ruff check .      # lint
ruff format .     # 整形
mypy              # 型チェック
```

設計判断・規約は `CLAUDE.md` に集約している。スキーマ・座標規約・依存の変更は
影響範囲が全リポジトリに及ぶため、`CLAUDE.md` を必ず確認すること。
