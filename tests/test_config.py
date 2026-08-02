"""agent_config.json のパースと検証（Pydantic）。

正常系（examples/）に加えて、プロンプト表の異常系を網羅する。
インスタンス直接生成は ``pydantic.ValidationError``、``parse_agent_config`` 経由は
``ConfigValidationError`` になる。ここでは一貫して ``parse_agent_config`` を通し、
``ConfigValidationError`` を期待する。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from jidohub.core.config import load_agent_config, parse_agent_config, validate_strict
from jidohub.core.config.agent import ConfigValidationError

from .conftest import EXAMPLES_DIR

SHA = "0" * 64


def native_config() -> dict[str, Any]:
    return {
        "schema_version": "0.2",
        "agent_id": "jidohub/CenterPoint",
        "task": "object_detection_3d",
        "sensors": {"lidar": ["LIDAR_TOP"]},
        "implementation": {"type": "native", "native_class": "CenterPointAgent"},
        "runtime": {"isolation": "not-required", "gpu_required": True},
        "weights": [{"path": "weights/model.safetensors", "format": "safetensors", "sha256": SHA}],
        "license": "Apache-2.0",
    }


def remote_config() -> dict[str, Any]:
    return {
        "schema_version": "0.2",
        "agent_id": "acme/UniAD",
        "task": "sensing_to_planning",
        "sensors": {"cameras": ["CAM_FRONT"], "requires_ego_state": True},
        "implementation": {
            "type": "remote_code",
            "auto_map": {"AutoAgent": "src/modeling.py:UniADAgent"},
        },
        "runtime": {"isolation": "required", "dockerfile": "runtime/Dockerfile"},
        "weights": [{"path": "weights/model.safetensors", "format": "safetensors", "sha256": SHA}],
        "intermediate_outputs": ["detection", "tracking"],
        "license": "Apache-2.0",
    }


def image_config() -> dict[str, Any]:
    """2D（画像入力）タスクの config。センサを宣言しない。"""
    return {
        "schema_version": "0.2",
        "agent_id": "acme/YOLO",
        "task": "object_detection_2d",
        "implementation": {"type": "native", "native_class": "YOLOAgent"},
        "runtime": {"isolation": "not-required", "gpu_required": True},
        "weights": [{"path": "weights/model.safetensors", "format": "safetensors", "sha256": SHA}],
        "prompt": {"required": False, "supported": ["text"]},
        "license": "Apache-2.0",
    }


def mutate(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(base)
    data.update(changes)
    return data


# --- 正常系 -----------------------------------------------------------------


def test_valid_native_config_parses() -> None:
    config = parse_agent_config(native_config())
    assert config.agent_id == "jidohub/CenterPoint"
    # use_enum_values=False なので task は enum のまま。
    assert config.task.value == "object_detection_3d"


def test_valid_remote_config_parses() -> None:
    config = parse_agent_config(remote_config())
    assert config.implementation.type == "remote_code"
    assert [o.value for o in config.intermediate_outputs] == ["detection", "tracking"]


def test_example_files_load() -> None:
    for name in ("centerpoint_agent_config.json", "uniad_agent_config.json"):
        config = load_agent_config(EXAMPLES_DIR / name)
        assert config.schema_version == "0.2"


# --- 異常系 -----------------------------------------------------------------


def test_remote_code_requires_isolation_required() -> None:
    data = remote_config()
    data["runtime"] = {"isolation": "recommended", "dockerfile": "runtime/Dockerfile"}
    with pytest.raises(ConfigValidationError, match="isolation must be 'required'"):
        parse_agent_config(data)


def test_intermediate_outputs_only_for_e2e() -> None:
    data = mutate(native_config(), intermediate_outputs=["detection"])
    with pytest.raises(ConfigValidationError, match="intermediate_outputs is only allowed"):
        parse_agent_config(data)


def test_intermediate_outputs_no_duplicates() -> None:
    data = mutate(remote_config(), intermediate_outputs=["detection", "detection"])
    with pytest.raises(ConfigValidationError, match="must not contain duplicates"):
        parse_agent_config(data)


def test_intermediate_outputs_unknown_value() -> None:
    data = mutate(remote_config(), intermediate_outputs=["bogus"])
    with pytest.raises(ConfigValidationError):
        parse_agent_config(data)


@pytest.mark.parametrize(
    "target",
    ["/abs/modeling.py:Cls", "../modeling.py:Cls", "modeling.py", "modeling:Cls"],
)
def test_auto_map_invalid(target: str) -> None:
    data = remote_config()
    data["implementation"]["auto_map"] = {"AutoAgent": target}
    with pytest.raises(ConfigValidationError):
        parse_agent_config(data)


@pytest.mark.parametrize("bad_path", ["/abs/model.safetensors", "../model.safetensors"])
def test_weights_path_must_be_relative(bad_path: str) -> None:
    data = native_config()
    data["weights"][0]["path"] = bad_path
    with pytest.raises(ConfigValidationError, match="weights.path"):
        parse_agent_config(data)


@pytest.mark.parametrize("bad_path", ["/abs/Dockerfile", "../Dockerfile"])
def test_dockerfile_must_be_relative(bad_path: str) -> None:
    data = remote_config()
    data["runtime"]["dockerfile"] = bad_path
    with pytest.raises(ConfigValidationError, match="runtime.dockerfile"):
        parse_agent_config(data)


@pytest.mark.parametrize("bad_id", ["noslash", "/leading", "a/b/c", "trailing/"])
def test_agent_id_format(bad_id: str) -> None:
    data = mutate(native_config(), agent_id=bad_id)
    with pytest.raises(ConfigValidationError, match="agent_id"):
        parse_agent_config(data)


def test_unknown_top_level_field_rejected() -> None:
    data = mutate(native_config(), typo_field=123)
    with pytest.raises(ConfigValidationError):
        parse_agent_config(data)


def test_native_requires_native_class() -> None:
    data = native_config()
    data["implementation"] = {"type": "native"}
    with pytest.raises(ConfigValidationError, match="native_class is required"):
        parse_agent_config(data)


def test_native_forbids_auto_map() -> None:
    data = native_config()
    data["implementation"] = {
        "type": "native",
        "native_class": "X",
        "auto_map": {"AutoAgent": "src/m.py:C"},
    }
    with pytest.raises(ConfigValidationError, match="auto_map must be empty"):
        parse_agent_config(data)


def test_remote_code_forbids_native_class() -> None:
    data = remote_config()
    data["implementation"]["native_class"] = "X"
    with pytest.raises(ConfigValidationError, match="native_class must be null"):
        parse_agent_config(data)


def test_remote_code_requires_auto_agent() -> None:
    data = remote_config()
    data["implementation"]["auto_map"] = {"AutoProcessor": "src/p.py:P"}
    with pytest.raises(ConfigValidationError, match="must contain 'AutoAgent'"):
        parse_agent_config(data)


def test_no_sensors_rejected() -> None:
    # object_detection_3d は SENSOR 入力なのでセンサ空は拒否。
    data = mutate(native_config(), sensors={})
    with pytest.raises(ConfigValidationError, match="at least one sensor"):
        parse_agent_config(data)


# --- 2D（画像入力）タスクのセンサ / プロンプト規則 --------------------------


def test_image_task_config_parses_without_sensors() -> None:
    config = parse_agent_config(image_config())
    assert config.task.value == "object_detection_2d"
    assert config.sensors.is_empty()
    assert config.prompt.supported == ["text"]


def test_image_task_forbids_sensors() -> None:
    # object_detection_2d は IMAGE 入力。センサを宣言したら拒否。
    data = mutate(image_config(), sensors={"cameras": ["CAM_FRONT"]})
    with pytest.raises(ConfigValidationError, match="sensors must be empty"):
        parse_agent_config(data)


def test_prompt_required_needs_supported() -> None:
    data = mutate(image_config(), prompt={"required": True, "supported": []})
    with pytest.raises(ConfigValidationError, match="supported must be non-empty"):
        parse_agent_config(data)


def test_prompt_supported_no_duplicates() -> None:
    data = mutate(image_config(), prompt={"required": True, "supported": ["text", "text"]})
    with pytest.raises(ConfigValidationError, match="must not contain duplicates"):
        parse_agent_config(data)


def test_prompt_supported_unknown_value_rejected() -> None:
    data = mutate(image_config(), prompt={"required": True, "supported": ["mask"]})
    with pytest.raises(ConfigValidationError):
        parse_agent_config(data)


@pytest.mark.parametrize("version", ["0.1", "1.0"])
def test_incompatible_schema_version(version: str) -> None:
    data = mutate(native_config(), schema_version=version)
    with pytest.raises(ConfigValidationError, match="incompatible"):
        parse_agent_config(data)


@pytest.mark.parametrize("version", ["1", "abc", "1.2.3"])
def test_malformed_schema_version_rejected(version: str) -> None:
    # 形式検証は schemas.version.assert_compatible に委譲済み。
    # major.minor でない値は引き続き ConfigValidationError で拒否される。
    data = mutate(native_config(), schema_version=version)
    with pytest.raises(ConfigValidationError, match="major.minor"):
        parse_agent_config(data)


def test_isolation_required_needs_source() -> None:
    data = native_config()
    data["runtime"] = {"isolation": "required", "gpu_required": True}
    with pytest.raises(ConfigValidationError, match="dockerfile or runtime.image is required"):
        parse_agent_config(data)


# --- validate_strict（例外でなく list を返す）--------------------------------


def test_validate_strict_clean_config() -> None:
    config = parse_agent_config(native_config())
    assert validate_strict(config) == []


def test_validate_strict_pytorch_format() -> None:
    data = native_config()
    data["weights"][0]["format"] = "pytorch"
    config = parse_agent_config(data)
    violations = validate_strict(config)
    assert any("safetensors is required" in v for v in violations)


def test_validate_strict_missing_sha256() -> None:
    data = native_config()
    data["weights"][0]["sha256"] = None
    config = parse_agent_config(data)
    violations = validate_strict(config)
    assert any("missing sha256" in v for v in violations)


def test_validate_strict_missing_license() -> None:
    data = mutate(native_config(), license=None)
    config = parse_agent_config(data)
    violations = validate_strict(config)
    assert any("license is required" in v for v in violations)


def test_validate_strict_missing_referenced_file(tmp_path) -> None:
    config = parse_agent_config(native_config())
    violations = validate_strict(config, repo_path=tmp_path)
    assert any("missing file" in v for v in violations)
