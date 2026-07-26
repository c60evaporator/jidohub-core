"""Agent 参照の解析。

利用者が指定する文字列（``"acme/UniAD-tiny@abc123"``、``"s3://bucket/agents/uniad"``、
``"./my-agent"`` など）を、取得方法が確定した :class:`AgentReference` に正規化する。

参照の種別
    - :attr:`RefKind.LOCAL`
        ローカルファイルシステム上のディレクトリ。開発中の Agent を直接指す。
    - :attr:`RefKind.S3`
        S3 上のプレフィックス。jidohub-web 稼働前の当面の配布経路。
    - :attr:`RefKind.HUB`
        jidohub-web 上の Agent。**未実装**（web の稼働後に対応する）。

リビジョン
    ``<参照>@<revision>`` の形式で指定する。revision は取得元によって意味が異なる
    （S3 ならプレフィックス配下のバージョン、Hub なら commit ハッシュ）が、
    **キャッシュのキーとして一意に扱える文字列**であることだけを要求する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = ["RefKind", "AgentReference"]

_AGENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_REVISION_PATTERN = re.compile(r"^[\w.-]+$")


class RefKind(str, Enum):
    """Agent 参照の種別。"""

    LOCAL = "local"
    S3 = "s3"
    HUB = "hub"


@dataclass(frozen=True)
class AgentReference:
    """正規化された Agent 参照。

    Attributes:
        kind: 取得元の種別。
        raw: 元の文字列（revision 部分を除く）。
        revision: リビジョン識別子。指定がなければ ``None``。
        agent_id: ``<namespace>/<name>`` 形式の識別子。HUB 参照でのみ設定される。
        path: ローカルパス。LOCAL 参照でのみ設定される。
        bucket: S3 バケット名。S3 参照でのみ設定される。
        key_prefix: S3 のキープレフィックス。S3 参照でのみ設定される。
    """

    kind: RefKind
    raw: str
    revision: str | None = None
    agent_id: str | None = None
    path: Path | None = None
    bucket: str | None = None
    key_prefix: str | None = None

    @classmethod
    def parse(cls, value: str | Path, revision: str | None = None) -> "AgentReference":
        """文字列またはパスを :class:`AgentReference` に解析する。

        判定順序
            1. :class:`~pathlib.Path` が渡された場合は無条件に LOCAL
            2. ``s3://`` / ``file://`` スキーム
            3. ディスク上に実在するパス
            4. ``<namespace>/<name>`` 形式なら HUB

        ``"acme/UniAD"`` のような文字列はローカルパスとも Agent ID とも解釈できるため、
        **実在するパスを優先**する。曖昧さを避けたい場合は ``Path`` を渡すか
        ``"./acme/UniAD"`` のように明示すること。

        Args:
            value: 参照文字列またはパス。
            revision: リビジョン。``value`` に ``@`` 表記がある場合は指定できない。

        Raises:
            ValueError: 形式が不正、または revision が二重指定された場合。
        """
        if isinstance(value, Path):
            return cls(
                kind=RefKind.LOCAL,
                raw=str(value),
                path=value.expanduser().resolve(),
                revision=_validated_revision(revision),
            )

        text = value.strip()
        if not text:
            raise ValueError("agent reference must not be empty")

        text, embedded_revision = _split_revision(text)
        if embedded_revision and revision:
            raise ValueError(
                f"revision specified twice: '@{embedded_revision}' and revision={revision!r}"
            )
        resolved_revision = _validated_revision(embedded_revision or revision)

        if text.startswith("s3://"):
            bucket, _, key_prefix = text[len("s3://") :].partition("/")
            if not bucket:
                raise ValueError(f"invalid S3 URI: {text!r}")
            return cls(
                kind=RefKind.S3,
                raw=text,
                revision=resolved_revision,
                bucket=bucket,
                key_prefix=key_prefix.strip("/"),
            )

        if text.startswith("file://"):
            return cls(
                kind=RefKind.LOCAL,
                raw=text,
                revision=resolved_revision,
                path=Path(text[len("file://") :]).expanduser().resolve(),
            )

        candidate = Path(text).expanduser()
        if candidate.exists():
            return cls(
                kind=RefKind.LOCAL, raw=text, revision=resolved_revision, path=candidate.resolve()
            )

        if _AGENT_ID_PATTERN.match(text):
            return cls(kind=RefKind.HUB, raw=text, revision=resolved_revision, agent_id=text)

        raise ValueError(
            f"cannot interpret {text!r} as an agent reference "
            "(expected an existing path, an s3:// URI, or '<namespace>/<name>')"
        )

    @property
    def cache_key(self) -> str:
        """キャッシュディレクトリ名に使う一意な識別子。

        パス区切りを ``--`` に置換し、revision をサフィックスに付ける。
        revision が未指定の場合は ``"main"`` を用いる。
        """
        if self.kind is RefKind.LOCAL:
            raise ValueError("local references are used in place and have no cache key")
        if self.kind is RefKind.S3:
            base = f"s3--{self.bucket}--{(self.key_prefix or '').replace('/', '--')}"
        else:
            base = f"hub--{(self.agent_id or '').replace('/', '--')}"
        return f"{base.rstrip('-')}/{self.revision or 'main'}"


def _split_revision(text: str) -> tuple[str, str | None]:
    """``<参照>@<revision>`` を分割する。S3 URI のスキーム部を誤検出しないこと。"""
    marker = text.rfind("@")
    if marker <= 0 or "/" in text[marker:]:
        return text, None
    return text[:marker], text[marker + 1 :]


def _validated_revision(revision: str | None) -> str | None:
    if revision is None:
        return None
    if not _REVISION_PATTERN.match(revision):
        raise ValueError(f"invalid revision: {revision!r}")
    return revision
