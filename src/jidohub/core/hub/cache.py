"""ローカルキャッシュとファイル整合性。

取得済みの Agent リポジトリを保持するディレクトリの管理を担う。

キャッシュの場所
    環境変数 ``JIDOHUB_HOME`` があればそれを、なければ ``~/.cache/jidohub`` を使う。

原子性
    ダウンロードは一時ディレクトリに展開してから :func:`os.replace` で移動する。
    途中で失敗した場合に中途半端なスナップショットが残ると、次回の実行が
    「取得済み」と誤認して壊れた Agent を読み込むため。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "SNAPSHOT_METADATA_FILENAME",
    "default_cache_dir",
    "compute_sha256",
    "staging_directory",
]

SNAPSHOT_METADATA_FILENAME = ".jidohub_snapshot.json"
"""スナップショットの取得元・リビジョンを記録するファイル名。

キャッシュされた Agent がどの参照から来たかを後から追跡できるようにする。
Agent リポジトリ本来の内容ではないため、ドット始まりの名前にしている。
"""


def default_cache_dir() -> Path:
    """キャッシュのルートディレクトリを返す。"""
    override = os.environ.get("JIDOHUB_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "jidohub"


def compute_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """ファイルの SHA-256 を 16 進文字列で返す。

    重みファイルは数 GB になり得るため、逐次読み込みで計算する。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def staging_directory(target: Path) -> Iterator[Path]:
    """一時ディレクトリを作り、成功時のみ ``target`` へ原子的に移動する。

    例外が発生した場合は一時ディレクトリごと破棄し、``target`` は変更しない。

    Args:
        target: 最終的な配置先。親ディレクトリは自動生成される。

    Yields:
        書き込み先の一時ディレクトリ。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        yield staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)
