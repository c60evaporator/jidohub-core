"""Hub クライアント: 参照解決・キャッシュ・チェックサム検証。

ネットワークにはアクセスしない。S3 取得は ``S3Backend(client=fake)`` に fake クライアントを
注入して検証する（boto3 は使わない）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jidohub.core.config import parse_agent_config
from jidohub.core.hub import (
    AgentReference,
    BackendError,
    ChecksumError,
    HubBackend,
    HubClient,
    RefKind,
    S3Backend,
    compute_sha256,
)

SHA_PLACEHOLDER = "0" * 64


class FakeS3Client:
    """S3Backend が使う最小限の API を模したフェイク。"""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.list_calls = 0
        self.download_calls = 0
        self.fail_download = False

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.list_calls += 1
        prefix = kwargs.get("Prefix", "")
        contents = [{"Key": key} for key in self.files if key.startswith(prefix)]
        return {"Contents": contents, "IsTruncated": False}

    def download_file(self, bucket: str, key: str, path: str) -> None:
        self.download_calls += 1
        if self.fail_download:
            raise RuntimeError("simulated download failure")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.files[key])


# --- AgentReference.parse ---------------------------------------------------


def test_parse_s3_uri() -> None:
    ref = AgentReference.parse("s3://my-bucket/agents/foo")
    assert ref.kind is RefKind.S3
    assert ref.bucket == "my-bucket"
    assert ref.key_prefix == "agents/foo"


def test_parse_file_uri(tmp_path) -> None:
    ref = AgentReference.parse(f"file://{tmp_path}")
    assert ref.kind is RefKind.LOCAL
    assert ref.path == tmp_path.resolve()


def test_parse_agent_id_with_revision() -> None:
    ref = AgentReference.parse("acme/Foo@v1.2")
    assert ref.kind is RefKind.HUB
    assert ref.agent_id == "acme/Foo"
    assert ref.revision == "v1.2"


def test_parse_existing_local_path(tmp_path) -> None:
    ref = AgentReference.parse(str(tmp_path))
    assert ref.kind is RefKind.LOCAL
    assert ref.path == tmp_path.resolve()


def test_parse_double_revision_rejected() -> None:
    with pytest.raises(ValueError, match="revision specified twice"):
        AgentReference.parse("acme/Foo@v1", revision="v2")


def test_parse_empty_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        AgentReference.parse("   ")


def test_parse_garbage_rejected() -> None:
    with pytest.raises(ValueError, match="cannot interpret"):
        AgentReference.parse("this is not valid")


# --- snapshot: LOCAL --------------------------------------------------------


def test_local_snapshot_returns_path_without_copying(tmp_path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "marker.txt").write_text("x")

    client = HubClient(cache_dir=tmp_path / "cache")
    result = client.snapshot(str(agent_dir))
    # コピーされず、同じディレクトリがそのまま返る。
    assert result.samefile(agent_dir)
    assert not (tmp_path / "cache").exists()


def test_local_snapshot_missing_dir(tmp_path) -> None:
    client = HubClient(cache_dir=tmp_path / "cache")
    # file:// URI は存在チェックなしで LOCAL に解決されるため、
    # snapshot 側の「ディレクトリ不在」分岐に到達する。
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="local agent directory not found"):
        client.snapshot(f"file://{missing}")


# --- snapshot: S3（fake 注入）----------------------------------------------


def make_s3_client(tmp_path, files: dict[str, bytes]) -> tuple[HubClient, FakeS3Client]:
    fake = FakeS3Client(files)
    client = HubClient(
        cache_dir=tmp_path / "cache",
        backends={RefKind.S3: S3Backend(client=fake), RefKind.HUB: HubBackend()},
    )
    return client, fake


def test_s3_snapshot_downloads_and_caches(tmp_path) -> None:
    files = {
        "agents/foo/agent_config.json": b"{}",
        "agents/foo/weights/model.safetensors": b"weights",
    }
    client, fake = make_s3_client(tmp_path, files)

    target = client.snapshot("s3://bucket/agents/foo")
    assert (target / "agent_config.json").read_bytes() == b"{}"
    assert (target / "weights" / "model.safetensors").read_bytes() == b"weights"
    first_list_calls = fake.list_calls
    assert first_list_calls >= 1

    # 2 回目はキャッシュヒット（バックエンドを呼ばない）。
    target2 = client.snapshot("s3://bucket/agents/foo")
    assert target2 == target
    assert fake.list_calls == first_list_calls

    # force=True で再取得される。
    client.snapshot("s3://bucket/agents/foo", force=True)
    assert fake.list_calls > first_list_calls


def test_s3_key_escaping_destination_rejected(tmp_path) -> None:
    files = {"agents/foo/../escape.txt": b"evil"}
    client, _ = make_s3_client(tmp_path, files)
    with pytest.raises(BackendError, match="refusing to write outside"):
        client.snapshot("s3://bucket/agents/foo")


def test_partial_failure_leaves_no_cache_dir(tmp_path) -> None:
    files = {"agents/foo/model.bin": b"data"}
    client, fake = make_s3_client(tmp_path, files)
    fake.fail_download = True

    with pytest.raises(RuntimeError, match="simulated download failure"):
        client.snapshot("s3://bucket/agents/foo")

    # 中途半端なキャッシュディレクトリが残らない。
    cache_agents = tmp_path / "cache" / "agents"
    leftovers = list(cache_agents.rglob("*")) if cache_agents.exists() else []
    assert not any(p.is_file() for p in leftovers)


def test_hub_reference_not_supported(tmp_path) -> None:
    client, _ = make_s3_client(tmp_path, {})
    with pytest.raises(BackendError):
        client.snapshot("acme/Foo")


# --- verify_weights ---------------------------------------------------------


def weights_config(sha: str | None) -> Any:
    data = {
        "schema_version": "0.1",
        "agent_id": "jidohub/Foo",
        "task": "object_detection_3d",
        "sensors": {"lidar": ["LIDAR_TOP"]},
        "implementation": {"type": "native", "native_class": "FooAgent"},
        "runtime": {"isolation": "not-required"},
        "weights": [{"path": "weights/model.safetensors", "format": "safetensors", "sha256": sha}],
        "license": "Apache-2.0",
    }
    return parse_agent_config(data)


def write_weight(repo: Path, content: bytes = b"model-bytes") -> str:
    weight_path = repo / "weights" / "model.safetensors"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(content)
    return compute_sha256(weight_path)


def test_verify_weights_matching(tmp_path) -> None:
    sha = write_weight(tmp_path)
    client = HubClient(cache_dir=tmp_path / "cache")
    client.verify_weights(weights_config(sha), tmp_path)  # 例外にならない


def test_verify_weights_mismatch(tmp_path) -> None:
    write_weight(tmp_path)
    client = HubClient(cache_dir=tmp_path / "cache")
    with pytest.raises(ChecksumError, match="checksum mismatch"):
        client.verify_weights(weights_config("f" * 64), tmp_path)


def test_verify_weights_missing_file(tmp_path) -> None:
    client = HubClient(cache_dir=tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="weight file not found"):
        client.verify_weights(weights_config("a" * 64), tmp_path)


def test_verify_weights_skips_when_sha_none(tmp_path) -> None:
    write_weight(tmp_path)
    client = HubClient(cache_dir=tmp_path / "cache")
    # sha256=None は検証をスキップ（例外にならない）。
    client.verify_weights(weights_config(None), tmp_path)
