"""PEP 420 namespace package の規約検証（CLAUDE.md 2.2）。

``src/jidohub/__init__.py`` が存在すると名前空間が占有され、他の jidohub
パッケージが import できなくなる。事故が起きても分かりにくいエラーになるため、
機械的に不在を検証する。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_jidohub_namespace_has_no_init() -> None:
    offending = REPO_ROOT / "src" / "jidohub" / "__init__.py"
    assert not offending.exists(), (
        "src/jidohub/__init__.py must not exist (PEP 420 namespace package); "
        "its presence breaks imports of sibling jidohub packages"
    )


def test_core_package_has_init() -> None:
    # core は通常のパッケージなので __init__.py が必要。
    assert (REPO_ROOT / "src" / "jidohub" / "core" / "__init__.py").exists()


def test_public_imports_succeed() -> None:
    # 完了条件 #5 の smoke test。
    import jidohub.core  # noqa: F401
    from jidohub.core import SCHEMA_VERSION, TaskType  # noqa: F401
    from jidohub.core.hub import HubClient  # noqa: F401
    from jidohub.core.schemas import Sample  # noqa: F401
    from jidohub.core.serialization import pack, unpack  # noqa: F401
