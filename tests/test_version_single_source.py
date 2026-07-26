"""バージョン定義の single source of truth 化を CI で機械的に守るテスト。

`SCHEMA_VERSION` と互換判定ロジックは `schemas/version.py` の 1 箇所のみが正。
将来また重複（drift）が生まれたら、ここで落とす。
"""

from __future__ import annotations

import ast

from .conftest import REPO_ROOT

SRC = REPO_ROOT / "src"
CANONICAL = "src/jidohub/core/schemas/version.py"


def _files_assigning(name: str) -> list[str]:
    """`name` へ代入している src 配下の .py（リポジトリ相対パス）を返す。

    f-string 内の参照や `x = SCHEMA_VERSION.split(...)` を誤検出しないよう、
    正規表現ではなく ast で代入ターゲットだけを見る。
    """
    hits: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    hits.append(str(py.relative_to(REPO_ROOT)))
    return hits


def test_schema_version_is_defined_exactly_once() -> None:
    hits = _files_assigning("SCHEMA_VERSION")
    assert hits == [CANONICAL], (
        f"SCHEMA_VERSION は {CANONICAL} の 1 箇所のみで定義すること。検出: {hits}"
    )


def test_compatibility_logic_is_not_reimplemented() -> None:
    """挙動一致ではなく、同一オブジェクトを参照していることを検証する。

    「別実装だが同じ結果」を許さないため（`==` ではなく `is`）。
    """
    from jidohub.core.config import agent
    from jidohub.core.schemas import version as schema_version
    from jidohub.core.serialization import envelope

    assert agent.assert_compatible is schema_version.assert_compatible
    assert envelope.assert_compatible is schema_version.assert_compatible
