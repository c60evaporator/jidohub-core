"""Hub クライアント（取得・キャッシュ・リビジョン解決）。

**現時点ではローカルパスと S3 参照のみ対応**する。jidohub-web 上の Agent
（``"<namespace>/<name>"`` 形式）は :class:`HubBackend` が明示的なエラーを返す。

責務の境界
    ここで扱うのは「参照 → ローカルに展開されたリポジトリのパス」まで。
    ``agent_config.json`` の解釈は :mod:`jidohub.core.config`、
    Agent のロードと実行は jidohub-agents が担う。

Example:
    >>> from jidohub.core.hub import HubClient
    >>> client = HubClient()
    >>> config, repo_path = client.load_config("s3://my-bucket/agents/uniad@v1")
    >>> client.verify_weights(config, repo_path)
"""

from __future__ import annotations

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
from jidohub.core.hub.client import ChecksumError, HubClient
from jidohub.core.hub.reference import AgentReference, RefKind

__all__ = [
    "HubClient",
    "AgentReference",
    "RefKind",
    "StorageBackend",
    "S3Backend",
    "HubBackend",
    "BackendError",
    "ChecksumError",
    "SNAPSHOT_METADATA_FILENAME",
    "default_cache_dir",
    "compute_sha256",
    "staging_directory",
]
