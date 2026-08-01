#!/usr/bin/env python3
"""jidohub-core の委譲成果を検証するスクリプト。

配置場所や実行ディレクトリに依存しない。どこからでも実行できる::

    python scripts/verify_core.py

委譲プロンプトの「完了条件」を機械的に確認する。
pytest とは別に用意しているのは、**テスト自体が正しく書かれているか**を
含めて確認したいため（テストを緩めて通すことを検知する）。
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path


def _find_repo_root() -> Path:
    """pyproject.toml を上位に辿ってリポジトリルートを特定する。

    スクリプトの配置場所や実行時のカレントディレクトリに依存しないようにするため。
    """
    for candidate in [Path.cwd().resolve(), *Path(__file__).resolve().parents]:
        for directory in [candidate, *candidate.parents]:
            if (directory / "pyproject.toml").is_file() and (directory / "src").is_dir():
                return directory
    raise SystemExit("リポジトリルート（pyproject.toml と src/ を持つ階層）が見つかりません")


ROOT = _find_repo_root()
SRC = ROOT / "src"

_results: list[tuple[bool, str, str]] = []


def check(name: str) -> "object":
    def decorator(fn):
        try:
            detail = fn() or ""
            _results.append((True, name, str(detail)))
        except AssertionError as error:
            _results.append((False, name, str(error)))
        except Exception as error:  # noqa: BLE001
            _results.append((False, name, f"{type(error).__name__}: {error}"))
        return fn

    return decorator


# ---------------------------------------------------------------
# 1. パッケージング / namespace package
# ---------------------------------------------------------------


@check("namespace package: src/jidohub/__init__.py が存在しない")
def _check_namespace_package() -> str:
    path = SRC / "jidohub" / "__init__.py"
    assert not path.exists(), f"{path} が存在する（名前空間が壊れる）"
    return "OK"


@check("py.typed が存在する")
def _check_py_typed() -> str:
    assert (SRC / "jidohub" / "core" / "py.typed").exists(), "py.typed がない"
    return "OK"


@check("import が通る")
def _check_imports() -> str:
    import jidohub.core.hub  # noqa: F401
    import jidohub.core.schemas  # noqa: F401
    import jidohub.core.serialization  # noqa: F401

    return "OK"


# ---------------------------------------------------------------
# 2. バージョン集約（single source of truth）
# ---------------------------------------------------------------


@check("SCHEMA_VERSION の代入が 1 箇所のみ")
def _check_schema_version_once() -> str:
    hits = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if re.search(r"^SCHEMA_VERSION\s*=", path.read_text(encoding="utf-8"), re.M)
    ]
    expected = ["jidohub/core/schemas/version.py"]
    assert hits == expected, f"定義箇所: {hits}"
    return hits[0]


@check("serialization/version.py が削除されている")
def _check_serialization_version_removed() -> str:
    path = SRC / "jidohub" / "core" / "serialization" / "version.py"
    assert not path.exists(), f"{path} が残っている"
    return "OK"


@check("assert_compatible が全モジュールで同一オブジェクト")
def _check_assert_compatible_identity() -> str:
    from jidohub.core.config import agent
    from jidohub.core.schemas import version
    from jidohub.core.serialization import envelope

    assert agent.assert_compatible is version.assert_compatible, "config が別実装を参照"
    assert envelope.assert_compatible is version.assert_compatible, "envelope が別実装を参照"
    return "OK"


# ---------------------------------------------------------------
# 3. Box3D の寸法アクセサ
# ---------------------------------------------------------------


@check("Box3D: length/width/height がプロパティであり、フィールドではない")
def _check_box3d_properties() -> str:
    from jidohub.core.schemas import Box3D

    field_names = {f.name for f in dataclasses.fields(Box3D)}
    for name in ("length", "width", "height", "yaw"):
        assert name not in field_names, f"{name} がフィールドになっている（直列化が壊れる）"
        attribute = inspect.getattr_static(Box3D, name, None)
        assert isinstance(attribute, property), f"{name} がプロパティでない"
    return f"fields={sorted(field_names)}"


@check("Box3D: 寸法アクセサが size の並びと一致する")
def _check_box3d_accessor_order() -> str:
    import numpy as np

    from jidohub.core.schemas import Box3D

    box = Box3D(
        center=np.zeros(3),
        size=np.array([4.5, 1.9, 1.6]),
        rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        label="car",
    )
    assert (box.length, box.width, box.height) == (4.5, 1.9, 1.6), (
        f"取り違え: l={box.length} w={box.width} h={box.height}"
    )
    assert box.length > box.width, "乗用車で length <= width（順序が逆の可能性）"
    return "l=4.5 w=1.9 h=1.6"


@check("Box3D.from_dimensions が正しい順序で size を構築する")
def _check_box3d_from_dimensions() -> str:
    import numpy as np

    from jidohub.core.schemas import Box3D

    box = Box3D.from_dimensions(
        center=np.zeros(3),
        length=4.5,
        width=1.9,
        height=1.6,
        rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        label="car",
        score=0.9,
        track_id=7,
    )
    assert list(box.size) == [4.5, 1.9, 1.6], f"size={box.size}"
    assert box.score == 0.9 and box.track_id == 7, "kwargs が伝播していない"
    return "OK"


# ---------------------------------------------------------------
# 4. スキーマと直列化の整合
# ---------------------------------------------------------------


@check("TYPE_REGISTRY が schemas の全 dataclass / Enum を網羅")
def _check_type_registry_complete() -> str:
    import jidohub.core.schemas as schemas
    from jidohub.core.serialization import TYPE_REGISTRY

    targets = {
        name
        for name in schemas.__all__
        if isinstance(getattr(schemas, name), type)
        and (
            dataclasses.is_dataclass(getattr(schemas, name))
            or issubclass(getattr(schemas, name), Enum)
        )
    }
    missing = sorted(targets - set(TYPE_REGISTRY))
    assert not missing, f"未登録: {missing}"
    return f"{len(targets)} 型を確認"


@check("Image: pixels / encoded が排他、CameraFrame は Image を保持")
def _check_camera_frame_exclusive() -> str:
    import numpy as np

    from jidohub.core.schemas import CameraFrame, Image

    # 排他は Image 側の責務（両方 None は不正）。
    try:
        Image(intrinsic=np.eye(3))
        raise AssertionError("pixels / encoded が両方 None の Image が許容された")
    except ValueError:
        pass

    pixels = np.zeros((4, 4, 3), np.uint8)
    frame = CameraFrame(
        image=Image(pixels=pixels, intrinsic=np.eye(3)),
        sensor_to_ego=np.eye(4),
        channel="CAM_FRONT",
    )
    assert frame.image.array.shape == (4, 4, 3) and not frame.image.is_encoded

    # intrinsic の唯一の正は Image。intrinsic 未知の Image は CameraFrame に載せられない。
    try:
        CameraFrame(image=Image(pixels=pixels), sensor_to_ego=np.eye(4), channel="CAM_FRONT")
        raise AssertionError("intrinsic なしの Image が CameraFrame に許容された")
    except ValueError:
        pass
    return "OK"


@check("直列化: Sample / E2EOutput の round-trip")
def _check_serialization_roundtrip() -> str:
    import numpy as np

    from jidohub.core.schemas import (
        Box3D,
        CameraFrame,
        Detection3DOutput,
        DrivingCommand,
        Image,
        LidarSweep,
        Sample,
    )
    from jidohub.core.serialization import pack, unpack

    sample = Sample(
        timestamp=1533151603547590,
        ego_to_global=np.eye(4),
        cameras={
            "CAM_FRONT": CameraFrame(
                image=Image(pixels=np.zeros((4, 4, 3), np.uint8), intrinsic=np.eye(3)),
                sensor_to_ego=np.eye(4),
                channel="CAM_FRONT",
            )
        },
        lidar=LidarSweep(
            points=np.random.rand(10, 5).astype(np.float32),
            sensor_to_ego=np.eye(4),
            fields=("x", "y", "z", "intensity", "ring"),
        ),
        command=DrivingCommand.TURN_LEFT,
    )
    restored = unpack(pack(sample))

    # str Enum は == で比較すると型が壊れていても True になるため is で確認する
    assert restored.command is DrivingCommand.TURN_LEFT, "enum が文字列に落ちている"
    assert isinstance(restored.lidar.fields, tuple), "tuple が list に落ちている"
    assert isinstance(restored.timestamp, int), "timestamp が int でない"
    assert restored.lidar.points.dtype == np.float32, "dtype が保持されていない"

    output = Detection3DOutput(
        boxes=[
            Box3D(
                center=np.zeros(3),
                size=np.array([4.5, 1.9, 1.6]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                label="car",
            )
        ]
    )
    assert unpack(pack(output)).boxes[0].length == 4.5
    return "OK"


@check("examples/*.json が読み込める")
def _check_examples_load() -> str:
    from jidohub.core.config import load_agent_config

    paths = sorted((ROOT / "examples").glob("*agent_config.json"))
    assert paths, "examples/ に agent_config.json がない"
    ids = [load_agent_config(path).agent_id for path in paths]
    return ", ".join(ids)


@check("config: セキュリティ不変条件が有効")
def _check_config_security_invariant() -> str:
    import json

    from jidohub.core.config import ConfigValidationError, parse_agent_config

    path = ROOT / "examples" / "uniad_agent_config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["runtime"]["isolation"] = "not-required"
    try:
        parse_agent_config(data)
        raise AssertionError("remote_code + isolation=not-required が許容された")
    except ConfigValidationError:
        return "OK"


# ---------------------------------------------------------------
# 5. 外部ツール
# ---------------------------------------------------------------


def run_tool(label: str, command: list[str]) -> None:
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        _results.append((False, label, "コマンドが見つからない"))
        return
    tail = (result.stdout + result.stderr).strip().splitlines()
    _results.append((result.returncode == 0, label, tail[-1] if tail else ""))


def main() -> int:
    run_tool("pytest", [sys.executable, "-m", "pytest", "-q"])
    run_tool("ruff check", [sys.executable, "-m", "ruff", "check", "."])
    run_tool("ruff format --check", [sys.executable, "-m", "ruff", "format", "--check", "."])
    run_tool("mypy", [sys.executable, "-m", "mypy", "src"])

    width = max(len(name) for _, name, _ in _results)
    failed = 0
    for ok, name, detail in _results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{mark}] {name.ljust(width)}  {detail}")

    print()
    print(f"{len(_results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
