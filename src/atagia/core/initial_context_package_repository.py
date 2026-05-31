"""Persistence for prepared initial context packages."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, _decode_json_columns, _encode_json
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageBlocks,
    InitialContextPackageBuildStatus,
    InitialContextPackageCoordinateSignature,
    InitialContextPackageDiagnostics,
    InitialContextPackageKey,
    InitialContextPackageKind,
    InitialContextPackagePolicySignature,
    InitialContextPackageReadResult,
    InitialContextPackageRecord,
    InitialContextPackageSourceFingerprint,
    initial_context_package_key_hash,
)

_UNSET = object()

_PACKAGE_COLUMNS = """
    id,
    package_key_hash,
    package_kind,
    version,
    user_id,
    conversation_id,
    retrieval_profile_id,
    key_json,
    policy_signature_json,
    coordinate_signature_json,
    source_fingerprint_json,
    blocks_json,
    source_refs_json,
    diagnostics_json,
    build_status,
    created_at,
    updated_at,
    valid_until
"""


def _json_payload(value: BaseModel | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


class InitialContextPackageRepository(BaseRepository):
    """Raw-SQL CRUD for durable prepared-context package rows."""

    async def upsert_package(
        self,
        *,
        package_kind: InitialContextPackageKind | str,
        version: int,
        user_id: str,
        retrieval_profile_id: str,
        key_json: InitialContextPackageKey | dict[str, Any],
        policy_signature_json: InitialContextPackagePolicySignature | dict[str, Any] | None = None,
        coordinate_signature_json: InitialContextPackageCoordinateSignature | dict[str, Any] | None = None,
        source_fingerprint_json: InitialContextPackageSourceFingerprint | dict[str, Any] | None = None,
        blocks_json: InitialContextPackageBlocks | dict[str, Any] | None = None,
        source_refs_json: dict[str, Any] | None = None,
        diagnostics_json: InitialContextPackageDiagnostics | dict[str, Any] | None = None,
        conversation_id: str | None = None,
        package_id: str | None = None,
        package_key_hash: str | None = None,
        build_status: InitialContextPackageBuildStatus | str = InitialContextPackageBuildStatus.ACTIVE,
        valid_until: str | None = None,
        commit: bool = True,
    ) -> InitialContextPackageRecord:
        resolved_kind = InitialContextPackageKind(package_kind)
        key = InitialContextPackageKey.model_validate(_json_payload(key_json))
        if resolved_kind != key.package_kind:
            raise ValueError("package_kind must match key_json")
        if user_id != key.user_id:
            raise ValueError("user_id must match key_json")
        resolved_conversation_id = (
            key.conversation_id if conversation_id is None else conversation_id
        )
        if resolved_conversation_id != key.conversation_id:
            raise ValueError("conversation_id must match key_json")
        if retrieval_profile_id != key.retrieval_profile_id:
            raise ValueError("retrieval_profile_id must match key_json")
        if int(version) != key.version:
            raise ValueError("version must match key_json")

        policy_signature = InitialContextPackagePolicySignature.model_validate(
            _json_payload(policy_signature_json)
        )
        coordinate_signature = InitialContextPackageCoordinateSignature.model_validate(
            _json_payload(coordinate_signature_json)
        )
        source_fingerprint = InitialContextPackageSourceFingerprint.model_validate(
            _json_payload(source_fingerprint_json)
        )
        blocks = InitialContextPackageBlocks.model_validate(_json_payload(blocks_json))
        diagnostics = InitialContextPackageDiagnostics.model_validate(
            _json_payload(diagnostics_json)
        )
        resolved_status = InitialContextPackageBuildStatus(build_status)
        timestamp = self._timestamp()
        resolved_package_id = package_id or generate_prefixed_id("icp")
        derived_key_hash = initial_context_package_key_hash(key)
        if package_key_hash is not None and package_key_hash != derived_key_hash:
            raise ValueError("package_key_hash must match key_json")
        resolved_key_hash = derived_key_hash

        cursor = await self._connection.execute(
            f"""
            INSERT INTO initial_context_packages(
                id,
                package_key_hash,
                package_kind,
                version,
                user_id,
                conversation_id,
                retrieval_profile_id,
                key_json,
                policy_signature_json,
                coordinate_signature_json,
                source_fingerprint_json,
                blocks_json,
                source_refs_json,
                diagnostics_json,
                build_status,
                created_at,
                updated_at,
                valid_until
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(package_key_hash) DO UPDATE SET
                package_kind = excluded.package_kind,
                version = excluded.version,
                user_id = excluded.user_id,
                conversation_id = excluded.conversation_id,
                retrieval_profile_id = excluded.retrieval_profile_id,
                key_json = excluded.key_json,
                policy_signature_json = excluded.policy_signature_json,
                coordinate_signature_json = excluded.coordinate_signature_json,
                source_fingerprint_json = excluded.source_fingerprint_json,
                blocks_json = excluded.blocks_json,
                source_refs_json = excluded.source_refs_json,
                diagnostics_json = excluded.diagnostics_json,
                build_status = excluded.build_status,
                updated_at = excluded.updated_at,
                valid_until = excluded.valid_until
            RETURNING {_PACKAGE_COLUMNS}
            """,
            (
                resolved_package_id,
                resolved_key_hash,
                resolved_kind.value,
                int(version),
                user_id,
                resolved_conversation_id,
                retrieval_profile_id,
                _encode_json(key.model_dump(mode="json")),
                _encode_json(policy_signature.model_dump(mode="json")),
                _encode_json(coordinate_signature.model_dump(mode="json")),
                _encode_json(source_fingerprint.model_dump(mode="json")),
                _encode_json(blocks.model_dump(mode="json")),
                _encode_json(source_refs_json or {}),
                _encode_json(diagnostics.model_dump(mode="json")),
                resolved_status.value,
                timestamp,
                timestamp,
                valid_until,
            ),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if commit:
            await self._connection.commit()
        if row is None:
            raise RuntimeError("Failed to upsert initial context package")
        return self._record_from_row(row)

    async def get_by_key_hash(
        self,
        *,
        user_id: str,
        package_key_hash: str,
        include_inactive: bool = False,
    ) -> InitialContextPackageRecord | None:
        status_clause = "" if include_inactive else "AND build_status = 'active'"
        row = await self._fetch_one(
            f"""
            SELECT {_PACKAGE_COLUMNS}
            FROM initial_context_packages
            WHERE user_id = ?
              AND package_key_hash = ?
              {status_clause}
            LIMIT 1
            """,
            (user_id, package_key_hash),
        )
        return None if row is None else InitialContextPackageRecord.model_validate(row)

    async def read_by_key_hash(
        self,
        *,
        user_id: str,
        package_key_hash: str,
    ) -> InitialContextPackageReadResult:
        package = await self.get_by_key_hash(
            user_id=user_id,
            package_key_hash=package_key_hash,
            include_inactive=True,
        )
        if package is None:
            return InitialContextPackageReadResult(status="miss")
        if package.build_status == InitialContextPackageBuildStatus.ACTIVE:
            return InitialContextPackageReadResult(status="hit", package=package)
        if package.build_status == InitialContextPackageBuildStatus.STALE:
            return InitialContextPackageReadResult(
                status="stale",
                package=package,
                fallback_reason="package_stale",
            )
        if package.build_status == InitialContextPackageBuildStatus.DELETED:
            return InitialContextPackageReadResult(
                status="deleted",
                package=package,
                fallback_reason="package_deleted",
            )
        return InitialContextPackageReadResult(
            status="unavailable",
            package=package,
            fallback_reason=f"package_{package.build_status.value}",
        )

    async def get_latest_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        retrieval_profile_id: str | None = None,
        include_inactive: bool = False,
    ) -> InitialContextPackageRecord | None:
        clauses = [
            "user_id = ?",
            "conversation_id = ?",
            "package_kind = 'conversation'",
        ]
        parameters: list[Any] = [user_id, conversation_id]
        if retrieval_profile_id is not None:
            clauses.append("retrieval_profile_id = ?")
            parameters.append(retrieval_profile_id)
        if not include_inactive:
            clauses.append("build_status = 'active'")
        row = await self._fetch_one(
            f"""
            SELECT {_PACKAGE_COLUMNS}
            FROM initial_context_packages
            WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC, id ASC
            LIMIT 1
            """,
            tuple(parameters),
        )
        return None if row is None else InitialContextPackageRecord.model_validate(row)

    async def get_latest_for_baseline_subject(
        self,
        *,
        user_id: str,
        retrieval_profile_id: str,
        subject_json: dict[str, Any],
        include_inactive: bool = False,
    ) -> InitialContextPackageRecord | None:
        clauses = [
            "user_id = ?",
            "package_kind = 'baseline'",
            "retrieval_profile_id = ?",
            "conversation_id IS NULL",
            "json_extract(key_json, '$.subject_json.user_persona_id') IS ?",
            "json_extract(key_json, '$.subject_json.platform_id') IS ?",
            "json_extract(key_json, '$.subject_json.character_id') IS ?",
            "json_extract(key_json, '$.subject_json.workspace_id') IS ?",
            "json_extract(key_json, '$.subject_json.assistant_mode_id') IS ?",
            "json_extract(key_json, '$.subject_json.mode') IS ?",
        ]
        parameters: list[Any] = [
            user_id,
            retrieval_profile_id,
            subject_json.get("user_persona_id"),
            subject_json.get("platform_id"),
            subject_json.get("character_id"),
            subject_json.get("workspace_id"),
            subject_json.get("assistant_mode_id"),
            subject_json.get("mode"),
        ]
        if not include_inactive:
            clauses.append("build_status = 'active'")
        row = await self._fetch_one(
            f"""
            SELECT {_PACKAGE_COLUMNS}
            FROM initial_context_packages
            WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC, id ASC
            LIMIT 1
            """,
            tuple(parameters),
        )
        return None if row is None else InitialContextPackageRecord.model_validate(row)

    async def mark_stale_by_key_hash(
        self,
        *,
        user_id: str,
        package_key_hash: str,
        commit: bool = True,
    ) -> int:
        cursor = await self._connection.execute(
            """
            UPDATE initial_context_packages
            SET build_status = ?,
                updated_at = ?
            WHERE user_id = ?
              AND package_key_hash = ?
              AND build_status <> ?
            """,
            (
                InitialContextPackageBuildStatus.STALE.value,
                self._timestamp(),
                user_id,
                package_key_hash,
                InitialContextPackageBuildStatus.DELETED.value,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def mark_stale_for_user(
        self,
        user_id: str,
        *,
        commit: bool = True,
    ) -> int:
        cursor = await self._connection.execute(
            """
            UPDATE initial_context_packages
            SET build_status = ?,
                updated_at = ?
            WHERE user_id = ?
              AND build_status <> ?
            """,
            (
                InitialContextPackageBuildStatus.STALE.value,
                self._timestamp(),
                user_id,
                InitialContextPackageBuildStatus.DELETED.value,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def mark_stale_for_key_family(
        self,
        *,
        user_id: str,
        package_kind: InitialContextPackageKind | str | None = None,
        retrieval_profile_id: str | None = None,
        conversation_id: str | None = None,
        privacy_enforcement: str | None = None,
        operational_profile_token: object = _UNSET,
        exclude_package_key_hashes: list[str] | None = None,
        commit: bool = True,
    ) -> int:
        clauses, parameters = self._family_clauses(
            user_id=user_id,
            package_kind=package_kind,
            retrieval_profile_id=retrieval_profile_id,
            conversation_id=conversation_id,
        )
        self._append_variant_clauses(
            clauses,
            parameters,
            privacy_enforcement=privacy_enforcement,
            operational_profile_token=operational_profile_token,
        )
        self._append_excluded_package_hashes_clause(
            clauses,
            parameters,
            exclude_package_key_hashes=exclude_package_key_hashes,
        )
        cursor = await self._connection.execute(
            f"""
            UPDATE initial_context_packages
            SET build_status = ?,
                updated_at = ?
            WHERE {" AND ".join(clauses)}
              AND build_status <> ?
            """,
            (
                InitialContextPackageBuildStatus.STALE.value,
                self._timestamp(),
                *parameters,
                InitialContextPackageBuildStatus.DELETED.value,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def mark_stale_for_baseline_subject(
        self,
        *,
        user_id: str,
        retrieval_profile_id: str,
        subject_json: dict[str, Any],
        privacy_enforcement: str | None = None,
        operational_profile_token: object = _UNSET,
        exclude_package_key_hashes: list[str] | None = None,
        commit: bool = True,
    ) -> int:
        clauses = [
            "user_id = ?",
            "package_kind = ?",
            "retrieval_profile_id = ?",
            "conversation_id IS NULL",
            "json_extract(key_json, '$.subject_json.user_persona_id') IS ?",
            "json_extract(key_json, '$.subject_json.platform_id') IS ?",
            "json_extract(key_json, '$.subject_json.character_id') IS ?",
            "json_extract(key_json, '$.subject_json.workspace_id') IS ?",
            "json_extract(key_json, '$.subject_json.assistant_mode_id') IS ?",
            "json_extract(key_json, '$.subject_json.mode') IS ?",
        ]
        parameters: list[Any] = [
            user_id,
            InitialContextPackageKind.BASELINE.value,
            retrieval_profile_id,
            subject_json.get("user_persona_id"),
            subject_json.get("platform_id"),
            subject_json.get("character_id"),
            subject_json.get("workspace_id"),
            subject_json.get("assistant_mode_id"),
            subject_json.get("mode"),
        ]
        self._append_variant_clauses(
            clauses,
            parameters,
            privacy_enforcement=privacy_enforcement,
            operational_profile_token=operational_profile_token,
        )
        self._append_excluded_package_hashes_clause(
            clauses,
            parameters,
            exclude_package_key_hashes=exclude_package_key_hashes,
        )
        cursor = await self._connection.execute(
            f"""
            UPDATE initial_context_packages
            SET build_status = ?,
                updated_at = ?
            WHERE {" AND ".join(clauses)}
              AND build_status <> ?
            """,
            (
                InitialContextPackageBuildStatus.STALE.value,
                self._timestamp(),
                *parameters,
                InitialContextPackageBuildStatus.DELETED.value,
            ),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def delete_for_user(
        self,
        user_id: str,
        *,
        commit: bool = True,
    ) -> int:
        cursor = await self._connection.execute(
            """
            DELETE FROM initial_context_packages
            WHERE user_id = ?
            """,
            (user_id,),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def delete_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        commit: bool = True,
    ) -> int:
        cursor = await self._connection.execute(
            """
            DELETE FROM initial_context_packages
            WHERE user_id = ?
              AND conversation_id = ?
            """,
            (user_id, conversation_id),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def delete_for_key_family(
        self,
        *,
        user_id: str,
        package_kind: InitialContextPackageKind | str | None = None,
        retrieval_profile_id: str | None = None,
        conversation_id: str | None = None,
        commit: bool = True,
    ) -> int:
        clauses, parameters = self._family_clauses(
            user_id=user_id,
            package_kind=package_kind,
            retrieval_profile_id=retrieval_profile_id,
            conversation_id=conversation_id,
        )
        cursor = await self._connection.execute(
            f"""
            DELETE FROM initial_context_packages
            WHERE {" AND ".join(clauses)}
            """,
            tuple(parameters),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    @staticmethod
    def _record_from_row(row: Any) -> InitialContextPackageRecord:
        decoded = _decode_json_columns(row)
        if decoded is None:
            raise RuntimeError("Failed to decode initial context package row")
        return InitialContextPackageRecord.model_validate(decoded)

    @staticmethod
    def _family_clauses(
        *,
        user_id: str,
        package_kind: InitialContextPackageKind | str | None,
        retrieval_profile_id: str | None,
        conversation_id: str | None,
    ) -> tuple[list[str], list[Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        if package_kind is not None:
            clauses.append("package_kind = ?")
            parameters.append(InitialContextPackageKind(package_kind).value)
        if retrieval_profile_id is not None:
            clauses.append("retrieval_profile_id = ?")
            parameters.append(retrieval_profile_id)
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            parameters.append(conversation_id)
        if len(clauses) == 1:
            raise ValueError("key-family operations require at least one family filter")
        return clauses, parameters

    @staticmethod
    def _append_variant_clauses(
        clauses: list[str],
        parameters: list[Any],
        *,
        privacy_enforcement: str | None,
        operational_profile_token: object,
    ) -> None:
        if privacy_enforcement is not None:
            clauses.append("json_extract(key_json, '$.policy_json.privacy_enforcement') IS ?")
            parameters.append(privacy_enforcement)
        if operational_profile_token is not _UNSET:
            clauses.append("json_extract(key_json, '$.operational_json.operational_profile.token') IS ?")
            parameters.append(operational_profile_token)

    @staticmethod
    def _append_excluded_package_hashes_clause(
        clauses: list[str],
        parameters: list[Any],
        *,
        exclude_package_key_hashes: list[str] | None,
    ) -> None:
        hashes = [
            str(package_key_hash).strip()
            for package_key_hash in (exclude_package_key_hashes or [])
            if str(package_key_hash).strip()
        ]
        if not hashes:
            return
        placeholders = ", ".join("?" for _ in hashes)
        clauses.append(f"package_key_hash NOT IN ({placeholders})")
        parameters.extend(hashes)
