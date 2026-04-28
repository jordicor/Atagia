"""Local filesystem storage for artifact payload bytes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StoredArtifactBlob:
    """Stored artifact payload metadata ready for artifact_blobs."""

    storage_kind: str
    blob_bytes: bytes | None
    storage_uri: str | None
    byte_size: int
    sha256: str


class ArtifactBlobStore:
    """Content-addressed local artifact blob storage."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir).expanduser().resolve()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def store_bytes(self, *, user_id: str, content_bytes: bytes) -> StoredArtifactBlob:
        sha256 = hashlib.sha256(content_bytes).hexdigest()
        blob_path = self._path_for(user_id=user_id, sha256=sha256)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        if blob_path.exists():
            existing_sha256 = hashlib.sha256(blob_path.read_bytes()).hexdigest()
            if existing_sha256 != sha256:
                blob_path.write_bytes(content_bytes)
        else:
            blob_path.write_bytes(content_bytes)
        return StoredArtifactBlob(
            storage_kind="local_file",
            blob_bytes=None,
            storage_uri=str(blob_path),
            byte_size=len(content_bytes),
            sha256=sha256,
        )

    def read_bytes(self, storage_uri: str) -> bytes:
        blob_path = self._resolve_storage_uri(storage_uri)
        return blob_path.read_bytes()

    def _path_for(self, *, user_id: str, sha256: str) -> Path:
        safe_user_id = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        return self._base_dir / safe_user_id / sha256[:2] / sha256

    def _resolve_storage_uri(self, storage_uri: str) -> Path:
        raw_path = Path(storage_uri).expanduser()
        candidate = raw_path if raw_path.is_absolute() else self._base_dir / raw_path
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self._base_dir)
        except ValueError:
            raise ValueError("Artifact blob path escapes configured storage directory") from None
        return resolved
