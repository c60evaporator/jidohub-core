"""``agent_config.json`` のスキーマとバリデータ。

``agent_config.json`` は Agent リポジトリのルートに置かれる**機械可読メタデータ**であり、
プラットフォーム上での Agent の正体（タスク・必要センサ・実装の在り処・実行環境）を表す。

責務の境界
    - ここで扱うのは**リポジトリ所有者が自己申告する情報**のみ。
      ``revision`` / Verified バッジ / プラットフォーム検証済みベンチマーク値のような
      **プラットフォーム側が付与する情報は含めない**（jidohub-web / server が持つ）。
    - 人間向けの発見用メタデータ（description, tags, citation 等）は
      ``README.md`` の frontmatter に置く。ここには機械が判断に使う情報だけを置く。
    - 自己申告のベンチマーク値は ``benchmark.json`` に分離する。

設計方針
    - config 面なので **Pydantic** を使う（データ面の dataclass と使い分ける。CLAUDE.md 4 章）。
    - トップレベルは ``extra="forbid"``。typo を検出するため。
      モデル固有のハイパーパラメータは :attr:`AgentConfig.params` に入れる。
    - 出力スキーマは ``task`` から一意に決まるため、config には持たせない
      （二重の正を作らない）。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jidohub.core.schemas.version import assert_compatible
from jidohub.core.tasks import (
    TASK_INPUT_KINDS,
    InputKind,
    IntermediateOutput,
    Platform,
    TaskType,
)

__all__ = [
    "AgentConfig",
    "SensorRequirement",
    "PromptRequirement",
    "Implementation",
    "WeightsSpec",
    "RuntimeSpec",
    "FrameworkSpec",
    "ConfigValidationError",
]

_AGENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_AUTO_MAP_PATTERN = re.compile(r"^[\w./-]+\.py:[A-Za-z_]\w*$")

# E2E 以外のタスクは中間出力を持たない（出力型に対応フィールドが存在しないため）
_TASKS_WITH_INTERMEDIATE_OUTPUTS = {TaskType.SENSING_TO_PLANNING}


class ConfigValidationError(ValueError):
    """``agent_config.json`` の検証に失敗した場合に送出される例外。"""


def _validate_relative_path(value: str, field_name: str) -> str:
    """リポジトリ内の相対パスとして安全かを検証する。

    絶対パスや ``..`` を許すと、Agent の取得時にリポジトリ外のファイルを
    参照させられる（パストラバーサル）。信頼できないユーザー投稿を
    受け付ける以上、パスは必ずここで閉じる。
    """
    if value.startswith("/") or "\\" in value or value.startswith("~"):
        raise ValueError(f"{field_name} must be a relative POSIX path, got {value!r}")
    if ".." in value.split("/"):
        raise ValueError(f"{field_name} must not contain '..', got {value!r}")
    return value


class SensorRequirement(BaseModel):
    """Agent が要求するセンサ構成。

    jidohub-web の検索フィルタ（対応センサ構成）と、
    実行時の :class:`~jidohub.core.schemas.Sample` の充足チェックの双方に使う。

    チャンネル名はデータセット依存のため enum 化せず自由文字列とするが、
    nuScenes 準拠の命名（``CAM_FRONT``、``LIDAR_TOP``、``RADAR_FRONT`` 等）を推奨する。
    """

    model_config = ConfigDict(extra="forbid")

    cameras: list[str] = Field(default_factory=list)
    """必要なカメラチャンネル名。``Sample.cameras`` のキーと対応する。"""

    lidar: list[str] = Field(default_factory=list)
    """必要な LiDAR チャンネル名。単一 LiDAR 構成では要素 1 個。"""

    radars: list[str] = Field(default_factory=list)
    """必要な RADAR チャンネル名。"""

    requires_ego_state: bool = False
    """``Sample.ego_state`` を必要とするか（プランナ系で ``True`` になる）。"""

    requires_command: bool = False
    """``Sample.command``（高レベル走行指令）を必要とするか。"""

    history_length: int = 0
    """必要な過去フレーム数。``Sample.history`` の最小長。0 なら単一フレームで動作する。"""

    def is_empty(self) -> bool:
        """センサチャンネルを 1 つも宣言していないか。

        センサ 1 つ以上の要否はタスクの入力種別（:class:`~jidohub.core.tasks.InputKind`）に
        依存するため、この判定は :class:`SensorRequirement` 単体では行わず
        :class:`AgentConfig` の相互検証で行う。
        """
        return not (self.cameras or self.lidar or self.radars)


class PromptRequirement(BaseModel):
    """2D タスクが受け付けるプロンプトの宣言。

    入力（``ImageSample.prompt``）をオプションにすると「この Agent はプロンプトが
    必要か / 何を受け付けるか」が型から読めなくなる。``sensors`` と同じ形で宣言させ、
    jidohub-web の検索フィルタ（「テキストプロンプト対応の検出器」）と実行前検証に使う
    （`2d_tasks.md` 5 章）。宣言がないと、実行して初めて失敗するか、黙って空の結果を返す。
    """

    model_config = ConfigDict(extra="forbid")

    required: bool = False
    """プロンプトが必須か。``True`` なら ``supported`` が空であってはならない。"""

    supported: list[Literal["point", "box", "text"]] = Field(default_factory=list)
    """受け付けるプロンプト種別。``ImageSample.prompt`` の各項目に対応する。"""

    @model_validator(mode="after")
    def _check_consistency(self) -> "PromptRequirement":
        if len(set(self.supported)) != len(self.supported):
            raise ValueError("prompt.supported must not contain duplicates")
        if self.required and not self.supported:
            raise ValueError("prompt.supported must be non-empty when prompt.required is true")
        return self


class Implementation(BaseModel):
    """Agent の実装コードの在り処。

    2 通りの提供形態を区別する（プラットフォームの二層構造）。

    - ``native``: jidohub-agents に同梱されたリファレンス実装を使う。
      コードは審査済みで、リモートコード実行を伴わない。
    - ``remote_code``: リポジトリに同梱されたコードを動的ロードする
      （Transformers の ``trust_remote_code=True`` に相当）。
      **未審査コードの実行になるため、隔離実行が必須**（:class:`RuntimeSpec` 参照）。
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["native", "remote_code"]
    """実装の提供形態。"""

    native_class: str | None = None
    """``native`` の場合に使う、jidohub-agents に登録されたクラス名（例: ``"CenterPointAgent"``）。"""

    auto_map: dict[str, str] = Field(default_factory=dict)
    """``remote_code`` の場合のクラス解決表。

    キーは ``"AutoAgent"`` / ``"AutoProcessor"``、値は ``"src/modeling.py:UniADAgent"``
    形式の「リポジトリ内相対パス:クラス名」。
    """

    @field_validator("auto_map")
    @classmethod
    def _validate_auto_map(cls, value: dict[str, str]) -> dict[str, str]:
        allowed_keys = {"AutoAgent", "AutoProcessor"}
        for key, target in value.items():
            if key not in allowed_keys:
                raise ValueError(f"auto_map key must be one of {sorted(allowed_keys)}, got {key!r}")
            if not _AUTO_MAP_PATTERN.match(target):
                raise ValueError(
                    f"auto_map[{key!r}] must be '<relative/path.py>:<ClassName>', got {target!r}"
                )
            _validate_relative_path(target.split(":", 1)[0], f"auto_map[{key!r}]")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "Implementation":
        if self.type == "native":
            if not self.native_class:
                raise ValueError("implementation.native_class is required when type is 'native'")
            if self.auto_map:
                raise ValueError("implementation.auto_map must be empty when type is 'native'")
        else:
            if "AutoAgent" not in self.auto_map:
                raise ValueError(
                    "implementation.auto_map must contain 'AutoAgent' when type is 'remote_code'"
                )
            if self.native_class:
                raise ValueError(
                    "implementation.native_class must be null when type is 'remote_code'"
                )
        return self


class WeightsSpec(BaseModel):
    """重みファイルの指定。"""

    model_config = ConfigDict(extra="forbid")

    path: str
    """リポジトリ内の相対パス（例: ``"weights/model.safetensors"``）。"""

    format: Literal["safetensors", "pytorch"] = "safetensors"
    """重みの形式。

    ``pytorch``（``.pth`` / ``.pt``）は pickle ベースであり、ロードするだけで
    任意コード実行が可能なため**それ自体が攻撃ベクトル**になる。
    受け入れは可能だが、Verified の対象にはしない（``strict`` 検証で拒否される）。
    """

    sha256: str | None = None
    """改竄検出用のチェックサム。"""

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validate_relative_path(value, "weights.path")


class FrameworkSpec(BaseModel):
    """Agent が要求する主要ライブラリのバージョン。

    依存衝突の事前検出と、隔離実行が必要かの判断材料に使う。
    """

    model_config = ConfigDict(extra="forbid")

    torch: str | None = None
    cuda: str | None = None
    library: str | None = None
    """ベースにしているフレームワーク（例: ``"mmdet3d"``、``"custom"``）。"""

    library_version: str | None = None


class RuntimeSpec(BaseModel):
    """実行環境の指定。

    Runner（``inprocess`` / ``docker``）の選択と、セキュリティ境界の決定に使う。
    """

    model_config = ConfigDict(extra="forbid")

    isolation: Literal["required", "recommended", "not-required"] = "required"
    """隔離実行（docker runner）の要否。

    ``required`` の場合、``inprocess`` での実行を**禁止**する。
    未審査コード（``remote_code``）と依存衝突するモデルはこれに該当する。
    """

    dockerfile: str | None = None
    """隔離実行に使う Dockerfile のリポジトリ内相対パス。"""

    image: str | None = None
    """ビルド済みイメージを使う場合の参照（例: ``"ghcr.io/org/uniad:0.1.0"``）。"""

    gpu_required: bool = True
    """GPU を必要とするか。server 側のスケジューリングに使う。"""

    @field_validator("dockerfile")
    @classmethod
    def _validate_dockerfile(cls, value: str | None) -> str | None:
        return None if value is None else _validate_relative_path(value, "runtime.dockerfile")

    @model_validator(mode="after")
    def _require_runtime_source(self) -> "RuntimeSpec":
        if self.isolation == "required" and not (self.dockerfile or self.image):
            raise ValueError(
                "runtime.dockerfile or runtime.image is required when isolation is 'required'"
            )
        return self


class AgentConfig(BaseModel):
    """``agent_config.json`` のルートスキーマ。

    Example:
        >>> config = AgentConfig.model_validate_json(text)
        >>> config.task
        <TaskType.SENSING_TO_PLANNING: 'sensing_to_planning'>
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    schema_version: str
    """準拠する標準スキーマのバージョン（``"0.1"`` 形式）。

    core の :data:`~jidohub.core.schemas.SCHEMA_VERSION` との互換性が検証される。
    """

    agent_id: str
    """``<namespace>/<name>`` 形式の識別子（例: ``"acme/UniAD-tiny-nuScenes"``）。"""

    task: TaskType
    """タスク種別。入出力の契約はこの値から一意に決まる。"""

    sensors: SensorRequirement = Field(default_factory=SensorRequirement)
    """要求するセンサ構成。

    センサ入力タスク（:attr:`~jidohub.core.tasks.InputKind.SENSOR`）では 1 つ以上を要求し、
    画像入力タスク（:attr:`~jidohub.core.tasks.InputKind.IMAGE`）では空でなければならない。
    2D タスクは ``ImageSample`` を入力に取りセンサを使わないため、既定を空にして省略可能とする。
    """

    prompt: PromptRequirement = Field(default_factory=PromptRequirement)
    """受け付けるプロンプトの宣言（主に 2D タスク）。既定はプロンプトなし。"""

    implementation: Implementation
    """実装コードの在り処。"""

    runtime: RuntimeSpec
    """実行環境の指定。"""

    weights: list[WeightsSpec] = Field(default_factory=list)
    """重みファイル。学習済み重みを持たない Agent（ルールベース等）では空でよい。"""

    intermediate_outputs: list[IntermediateOutput] = Field(default_factory=list)
    """公開する中間出力。``sensing_to_planning`` 以外では空でなければならない。

    ここで宣言した項目は、``E2EOutput`` の同名フィールドに実際に値が入ることを保証する契約。
    """

    platforms: list[Platform] = Field(default_factory=lambda: [Platform.PYTHON])
    """動作するプラットフォーム。"""

    framework: FrameworkSpec = Field(default_factory=FrameworkSpec)
    """要求ライブラリのバージョン。"""

    license: str | None = None
    """SPDX 識別子（例: ``"Apache-2.0"``）。"""

    base_agent: str | None = None
    """ファインチューニング元の Agent ID。系譜の追跡に使う。"""

    training_datasets: list[str] = Field(default_factory=list)
    """学習に使ったデータセットの ID。評価時のデータリーク検出に使う。"""

    params: dict[str, Any] = Field(default_factory=dict)
    """モデル固有のハイパーパラメータ。

    トップレベルが ``extra="forbid"`` なので、標準フィールドで表せない設定は
    すべてここに入れる。**core はこの中身を解釈しない**（各 Agent の実装が読む）。
    """

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, value: str) -> str:
        if not _AGENT_ID_PATTERN.match(value):
            raise ValueError(f"agent_id must be '<namespace>/<name>', got {value!r}")
        return value

    @model_validator(mode="after")
    def _check_cross_field_rules(self) -> "AgentConfig":
        # 1. スキーマ互換性
        _assert_schema_compatible(self.schema_version)

        # 2. 入力種別に応じたセンサ要求の検証。
        #    センサの要否はタスクの入力種別に依存するため、SensorRequirement 単体ではなく
        #    ここで判定する（2D タスクは ImageSample 入力でセンサを持たない）。
        input_kind = TASK_INPUT_KINDS[self.task]
        if input_kind is InputKind.SENSOR and self.sensors.is_empty():
            raise ValueError(
                f"at least one sensor (cameras / lidar / radars) is required "
                f"for sensor-input task {self.task.value!r}"
            )
        if input_kind is InputKind.IMAGE and not self.sensors.is_empty():
            raise ValueError(
                f"sensors must be empty for image-input task {self.task.value!r} "
                "(it takes an ImageSample, not sensor data)"
            )
        # 入力種別に応じたプロンプト宣言の検証。
        # Sample に prompt フィールドは無いため、センサ入力タスクでの宣言は実行時に何の意味も持たない（Webのフィルタで誤解を招く）
        # COMPOSITE は複合入力型がテキストを含み得るため許容する（例: vision_language_action は Sample + テキストを取る）
        if input_kind is InputKind.SENSOR and (self.prompt.required or self.prompt.supported):
            raise ValueError(
                f"prompt is not applicable to sensor-input task {self.task.value!r} "
                "(Sample has no prompt field)"
            )

        # 3. 中間出力は E2E タスクのみ
        if self.intermediate_outputs and self.task not in _TASKS_WITH_INTERMEDIATE_OUTPUTS:
            raise ValueError(
                f"intermediate_outputs is only allowed for "
                f"{sorted(t.value for t in _TASKS_WITH_INTERMEDIATE_OUTPUTS)}, "
                f"but task is {self.task.value!r}"
            )
        if len(set(self.intermediate_outputs)) != len(self.intermediate_outputs):
            raise ValueError("intermediate_outputs must not contain duplicates")

        # 4. セキュリティ不変条件: 未審査コードは必ず隔離実行
        if self.implementation.type == "remote_code" and self.runtime.isolation != "required":
            raise ValueError(
                "runtime.isolation must be 'required' when implementation.type is 'remote_code' "
                "(unreviewed code must never run in-process)"
            )

        return self


def _assert_schema_compatible(declared: str) -> None:
    """宣言された schema_version が core と互換かを検証する。

    互換ルール（形式チェックを含む）の実体は
    :func:`jidohub.core.schemas.version.assert_compatible` にある。
    判定ロジックをここに複製しないこと。config 固有の追加検証を将来挟む余地として
    薄いラッパのみ残す。
    """
    assert_compatible(declared)
