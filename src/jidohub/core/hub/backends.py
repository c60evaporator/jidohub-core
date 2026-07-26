"""取得バックエンド。

取得元ごとの実装を :class:`StorageBackend` プロトコルの背後に隠す。
:class:`~jidohub.core.hub.client.HubClient` はこのプロトコルにのみ依存するため、
jidohub-server が独自のバックエンド（社内ストレージ等）を注入できる。

依存について
    ``boto3`` は **optional 依存**であり、:class:`S3Backend` の中で遅延 import する。
    core を入れただけで boto3 が付いてくることを避けるため、
    トップレベルで import しないこと（CLAUDE.md 2.1）。
    S3 を使う場合は ``pip install 'jidohub-core[s3]'``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from jidohub.core.hub.reference import AgentReference, RefKind

__all__ = ["StorageBackend", "S3Backend", "HubBackend", "BackendError"]


class BackendError(RuntimeError):
    """取得処理に失敗した場合に送出される例外。"""


@runtime_checkable
class StorageBackend(Protocol):
    """Agent リポジトリを取得するバックエンドのプロトコル。"""

    kind: RefKind

    def fetch(self, reference: AgentReference, destination: Path) -> None:
        """``reference`` の内容を ``destination`` 配下に展開する。

        ``destination`` は空のディレクトリとして渡される。
        失敗した場合は :class:`BackendError` を送出すること
        （呼び出し側が一時ディレクトリを破棄する）。
        """
        ...


class S3Backend:
    """S3 プレフィックスから Agent リポジトリを取得するバックエンド。

    リビジョンの扱い
        ``revision`` が指定された場合、``<key_prefix>/<revision>/`` 配下を取得する。
        未指定なら ``<key_prefix>/`` 配下をそのまま取得する。
        jidohub-web 稼働後は commit ハッシュに置き換わる想定の、暫定的な規約。
    """

    kind = RefKind.S3

    def __init__(self, client=None) -> None:
        """
        Args:
            client: boto3 の S3 クライアント。``None`` なら遅延生成する。
                テストや独自の認証設定を注入する場合に使う。
        """
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import boto3  # optional 依存: ここでのみ import する
            except ImportError as error:
                raise BackendError(
                    "boto3 is required for s3:// references. "
                    "Install it with: pip install 'jidohub-core[s3]'"
                ) from error
            self._client = boto3.client("s3")
        return self._client

    def fetch(self, reference: AgentReference, destination: Path) -> None:
        if reference.kind is not RefKind.S3:
            raise BackendError(f"S3Backend cannot handle {reference.kind.value} references")

        client = self._get_client()
        prefix = "/".join(part for part in [reference.key_prefix, reference.revision] if part)
        prefix = f"{prefix.rstrip('/')}/" if prefix else ""

        keys = self._list_keys(client, reference.bucket, prefix)
        if not keys:
            raise BackendError(f"no objects found at s3://{reference.bucket}/{prefix}")

        for key in keys:
            relative = key[len(prefix) :]
            if not relative or relative.endswith("/"):
                continue
            local_path = _safe_join(destination, relative)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(reference.bucket, key, str(local_path))

    @staticmethod
    def _list_keys(client, bucket: str, prefix: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = client.list_objects_v2(**kwargs)
            keys.extend(item["Key"] for item in response.get("Contents", []))
            if not response.get("IsTruncated"):
                return keys
            token = response.get("NextContinuationToken")


class HubBackend:
    """jidohub-web から取得するバックエンド。**未実装**。

    web の稼働後に、revision 解決（タグ・ブランチ → commit ハッシュ）と
    認証付きダウンロードを実装する。それまでは明示的なエラーを返し、
    利用者にローカルパスか S3 参照を促す。
    """

    kind = RefKind.HUB

    def fetch(self, reference: AgentReference, destination: Path) -> None:
        raise BackendError(
            f"hub references are not supported yet (got {reference.raw!r}). "
            "Use a local path or an s3:// URI until jidohub-web is available."
        )


def _safe_join(root: Path, relative: str) -> Path:
    """``root`` の外に出ないことを検証してパスを結合する。

    オブジェクトキーに ``..`` が含まれる場合、素朴に結合すると
    キャッシュ外へ書き込めてしまう（Zip Slip と同種の問題）。
    """
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise BackendError(f"refusing to write outside the destination: {relative!r}")
    return candidate
