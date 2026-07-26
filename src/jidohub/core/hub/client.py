"""Hub クライアント。

Agent 参照を受け取り、**ローカルに展開されたリポジトリのパス**を返すことだけを担う。
config の解釈（:mod:`jidohub.core.config`）や Agent のロード（jidohub-agents）は行わない。

Example:
    >>> client = HubClient()
    >>> repo_path = client.snapshot("s3://my-bucket/agents/uniad-tiny@v1")
    >>> config = load_agent_config(repo_path)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jidohub.core.config import AgentConfig, load_agent_config
from jidohub.core.hub.backends import (
    BackendError,
    HubBackend,
    S3Backend,
    StorageBackend,
)
from jidohub.core.hub.cache import (
    SNAPSHOT_METADATA_FILENAME,
    compute_sha256,
    default_cache_dir,
    staging_directory,
)
from jidohub.core.hub.reference import AgentReference, RefKind

__all__ = ["HubClient", "ChecksumError"]


class ChecksumError(RuntimeError):
    """重みファイルのチェックサムが一致しない場合に送出される例外。"""


class HubClient:
    """Agent リポジトリの取得とキャッシュを行うクライアント。"""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        backends: dict[RefKind, StorageBackend] | None = None,
    ) -> None:
        """
        Args:
            cache_dir: キャッシュのルート。``None`` なら ``JIDOHUB_HOME`` または
                ``~/.cache/jidohub``。
            backends: 取得元種別ごとのバックエンド。独自ストレージを使う場合に注入する。
        """
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
        self.backends: dict[RefKind, StorageBackend] = backends or {
            RefKind.S3: S3Backend(),
            RefKind.HUB: HubBackend(),
        }

    def snapshot(
        self,
        reference: str | Path | AgentReference,
        revision: str | None = None,
        force: bool = False,
    ) -> Path:
        """Agent リポジトリをローカルに用意し、そのパスを返す。

        ローカル参照の場合はコピーせず、**そのパスをそのまま返す**
        （開発中の Agent を編集しながら実行できるようにするため）。

        Args:
            reference: 参照文字列、パス、または :class:`AgentReference`。
            revision: リビジョン。``reference`` に ``@`` 表記がある場合は指定できない。
            force: ``True`` ならキャッシュを無視して再取得する。

        Raises:
            FileNotFoundError: ローカル参照の指す先が存在しない場合。
            BackendError: 取得に失敗した場合。
        """
        ref = (
            reference
            if isinstance(reference, AgentReference)
            else AgentReference.parse(reference, revision)
        )

        if ref.kind is RefKind.LOCAL:
            if ref.path is None or not ref.path.is_dir():
                raise FileNotFoundError(f"local agent directory not found: {ref.raw}")
            return ref.path

        target = self.cache_dir / "agents" / ref.cache_key
        if target.is_dir() and not force:
            return target

        backend = self.backends.get(ref.kind)
        if backend is None:
            raise BackendError(f"no backend registered for {ref.kind.value} references")

        with staging_directory(target) as staging:
            backend.fetch(ref, staging)
            _write_snapshot_metadata(staging, ref)

        return target

    def load_config(
        self,
        reference: str | Path | AgentReference,
        revision: str | None = None,
    ) -> tuple[AgentConfig, Path]:
        """Agent を取得し、``agent_config.json`` を読み込んで返す。

        Returns:
            ``(config, repo_path)``。
        """
        repo_path = self.snapshot(reference, revision)
        return load_agent_config(repo_path), repo_path

    def verify_weights(self, config: AgentConfig, repo_path: str | Path) -> None:
        """``agent_config.json`` に宣言された重みのチェックサムを検証する。

        ``sha256`` が宣言されていない重みは検証をスキップする
        （宣言を必須にするかは :func:`~jidohub.core.config.validate_strict` の判断）。

        Raises:
            FileNotFoundError: 重みファイルが存在しない場合。
            ChecksumError: チェックサムが一致しない場合。
        """
        root = Path(repo_path)
        for weight in config.weights:
            path = root / weight.path
            if not path.is_file():
                raise FileNotFoundError(f"weight file not found: {path}")
            if weight.sha256 is None:
                continue
            actual = compute_sha256(path)
            if actual != weight.sha256:
                raise ChecksumError(
                    f"checksum mismatch for {weight.path}: "
                    f"expected {weight.sha256}, got {actual}"
                )


def _write_snapshot_metadata(directory: Path, reference: AgentReference) -> None:
    """取得元とリビジョンを記録する。後からキャッシュの出自を追跡できるようにする。"""
    metadata = {
        "source": reference.raw,
        "kind": reference.kind.value,
        "revision": reference.revision,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / SNAPSHOT_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
