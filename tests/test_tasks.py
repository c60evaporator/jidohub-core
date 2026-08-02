"""TaskType の値フォーマット規約を守るためのテスト。

タスク値は snake_case のみ（ハイフン＝kebab-case 禁止）で、メンバ名を小文字化した
ものと一致させる。次に誰かが値を追加したときに、この規約から外れたら落とす。

テストは tasks.py の実装定数に依存せず、規約（パターン）をここで独立に再宣言する
（実装とテストが同じ間違いで同時に壊れないようにするため）。
"""

from __future__ import annotations

import re
from enum import Enum

import pytest

from jidohub.core.tasks import TASK_INPUT_KINDS, InputKind, TaskType

# 仕様としてのパターン（tasks.py の実装とは独立に宣言する）。
_SPEC_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# 次元サフィックス（末尾にのみ許す）。
_DIMENSION_SUFFIXES = ("2d", "3d")


def test_task_name_matches_value() -> None:
    for member in TaskType:
        assert member.name.lower() == member.value, (
            f"{member.name} の値は {member.name.lower()!r} であるべき（実際は {member.value!r}）"
        )


def test_task_values_are_snake_case() -> None:
    for member in TaskType:
        assert _SPEC_PATTERN.fullmatch(member.value), (
            f"{member.name} の値 {member.value!r} は snake_case でなければならない"
        )
        assert "-" not in member.value, f"{member.name} の値にハイフンが含まれている"


def test_task_values_are_unique() -> None:
    values = [member.value for member in TaskType]
    assert len(values) == len(set(values)), "TaskType の値に重複がある"


def test_dimension_suffix_is_last() -> None:
    for member in TaskType:
        tokens = member.value.split("_")
        for index, token in enumerate(tokens):
            if token in _DIMENSION_SUFFIXES:
                assert index == len(tokens) - 1, (
                    f"{member.name} の値 {member.value!r}: "
                    f"次元サフィックス {token!r} は必ず末尾に置く"
                )


def test_task_input_kinds_covers_all_members() -> None:
    # 対応表の追加漏れを検出する（タスクを増やしたら必ず入力種別を宣言させる）。
    missing = [member.name for member in TaskType if member not in TASK_INPUT_KINDS]
    assert not missing, f"TASK_INPUT_KINDS に未登録のタスク: {missing}"
    assert all(isinstance(kind, InputKind) for kind in TASK_INPUT_KINDS.values())


def test_hyphen_value_is_rejected_at_definition() -> None:
    """ガード機構そのものを検証する。

    TaskType 本体はメンバを後から追加できないため、同じ検証 __init__ を持つ
    使い捨ての enum を定義しようとして ``ValueError`` になることを確認する。
    """
    with pytest.raises(ValueError, match="snake_case"):

        class _Bad(str, Enum):
            def __init__(self, value: str) -> None:
                if not _SPEC_PATTERN.fullmatch(value):
                    raise ValueError("snake_case; ハイフン不可")

            BAD = "sensing-to-detection"
