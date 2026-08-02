# jidohub-core

jidohub-coreの設計ドキュメント。コーディングエージェント用の設計情報は[CLAUDE.md](../CLAUDE.md)に記載しているので、ここでは人間が参照するための設計情報を記載する。

## jidohub-coreのフォルダ構成

```
src/
└── jidohub/                     ← namespace package。__init__.pyを置かないので`import jidohub.core`できる
    └── core/
        ├── __init__.py
        ├── tasks.py             タスク種別、中間出力、実行プラットフォームの一覧を定義
        ├── geometry.py          座標変換・回転表現・intrinsic の crop/scale 追従（純 numpy）
        ├── schemas/             標準スキーマ（dataclass）
        │   ├── __init__.py      再エクスポート
        │   ├── version.py       SCHEMA_VERSION / assert_compatible / is_compatible（互換判定の唯一の正）
        │   ├── image.py         画像: Image / EncodedPixels / ImageSource / デコーダ注入
        │   ├── sample.py        入力(センサ): Sample / CameraFrame / LidarSweep /
        │   │                    RadarSweep / EgoState / DrivingCommand
        │   ├── prompts.py       入力(2D): ImageSample / ImagePrompt
        │   └── outputs.py       出力: Box3D / Detection3DOutput / Box2D / Instance2D /
        │                        Detection2DOutput / Classification2DOutput / E2EOutput など
        ├── config/              agent_config.json のパース・検証（Pydantic）
        ├── serialization/       Sample / Output の直列化・復元（pack / unpack）
        └── hub/                 Hub クライアント（snapshot / revision 解決 / チェックサム検証）
```


## ユースケース別実装ガイダンス

### タスク種別を増やしたいとき

- `src/jidohub/core/tasks.py`の`TaskType`クラスのクラス定数にタスク種別名を追加
- `src/jidohub/core/schemas/outputs.py`に出力形式を定義するdataclassを追加
