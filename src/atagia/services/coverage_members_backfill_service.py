"""Backfill ``coverage_members`` payload metadata for existing memory rows.

Legacy rows predate the ``coverage_members`` extraction card, so their
``payload_json`` carries no ``coverage_members`` key. This service re-derives the
key by running ONLY the coverage-members enrichment card over each row's
``canonical_text`` and writing the parsed result back into the existing
``payload_json`` dict.

Key presence is the processed marker: ``[]`` means "processed, no enumerable
members"; absence means "not yet processed" (a legacy row). A row that fails to
process keeps no key, so it is re-runnable.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from atagia.core import json_utils
from atagia.core.config import Settings
from atagia.memory.extraction_cards import (
    CandidateDraft,
    build_enrichment_prompt,
    parse_coverage_members_card_output,
    _CARD_MAX_OUTPUT_TOKENS,
    _CARD_SYSTEM_PROMPTS,
)
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    CoverageMember,
    ExtractionConversationContext,
    MemoryStatus,
    RetrievalProfileId,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMMessage,
    StructuredOutputError,
)
from atagia.services.model_resolution import resolve_component_model


logger = logging.getLogger(__name__)

_COVERAGE_MEMBERS_PAYLOAD_KEY = "coverage_members"
# Resolved policy is unused by the coverage-members card builder (it calls
# ``del resolved_policy`` immediately), so the profile choice is immaterial to
# the card output. A single fixed default keeps the backfill from depending on
# every stored row's ``assistant_mode_id`` being a valid retrieval profile.
_DEFAULT_PROFILE_ID = RetrievalProfileId.GENERAL_QA
_CANDIDATE_ID = "cand_001"

_BACKFILL_STATUSES = frozenset(
    {
        MemoryStatus.ACTIVE.value,
        MemoryStatus.SUPERSEDED.value,
        MemoryStatus.REVIEW_REQUIRED.value,
        MemoryStatus.PENDING_USER_CONFIRMATION.value,
    }
)

ProgressCallback = Callable[["CoverageMembersBackfillResult"], Awaitable[None] | None]


class CoverageMembersBackfillResult(BaseModel):
    """Counters and parameters for one coverage-members backfill run."""

    model_config = ConfigDict(extra="forbid")

    examined: int = 0
    processed: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    dry_run: bool
    batch_size: int = Field(ge=1)
    delay_ms: int = Field(ge=0)
    user_id: str | None = None


class CoverageMembersBackfillService:
    """Scan memory rows missing the ``coverage_members`` key and backfill them."""

    def __init__(
        self,
        *,
        connection: aiosqlite.Connection,
        llm_client: LLMClient[Any],
        settings: Settings | None = None,
        progress_callback: ProgressCallback | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._connection = connection
        self._llm_client = llm_client
        self._settings = settings or Settings.from_env()
        self._card_model = resolve_component_model(self._settings, "extractor")
        self._resolved_policy = PolicyResolver().resolve(
            ManifestLoader(self._settings.manifests_dir()).get(
                _DEFAULT_PROFILE_ID.value
            ),
            None,
            None,
        )
        self._progress_callback = progress_callback
        self._sleep = sleep or asyncio.sleep

    async def run(
        self,
        *,
        batch_size: int,
        delay_ms: int,
        user_id: str | None = None,
        dry_run: bool = True,
    ) -> CoverageMembersBackfillResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")

        result = CoverageMembersBackfillResult(
            dry_run=dry_run,
            batch_size=batch_size,
            delay_ms=delay_ms,
            user_id=user_id,
        )
        cursor = await self._connection.execute(*self._scan_query(user_id))
        try:
            rows = await cursor.fetchmany(batch_size)
            while rows:
                for row in rows:
                    result.examined += 1
                    await self._process_row(row, result, dry_run=dry_run)
                await self._emit_progress(result)
                rows = await cursor.fetchmany(batch_size)
                if rows and delay_ms > 0:
                    await self._sleep(delay_ms / 1000.0)
        finally:
            await cursor.close()
        return result

    def _scan_query(self, user_id: str | None) -> tuple[str, tuple[Any, ...]]:
        clauses = [
            "json_extract(mo.payload_json, '$.{key}') IS NULL".format(
                key=_COVERAGE_MEMBERS_PAYLOAD_KEY
            ),
            "mo.status IN ({statuses})".format(
                statuses=", ".join("?" for _status in _BACKFILL_STATUSES)
            ),
        ]
        parameters: list[Any] = [*sorted(_BACKFILL_STATUSES)]
        if user_id is not None:
            clauses.append("mo.user_id = ?")
            parameters.append(user_id)
        return (
            """
            SELECT
                mo.id,
                mo.user_id,
                mo.conversation_id,
                mo.assistant_mode_id,
                mo.canonical_text,
                mo.payload_json,
                mo.status
            FROM memory_objects AS mo
            WHERE {where_clause}
            ORDER BY mo.created_at ASC, mo.id ASC
            """.format(where_clause=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def _process_row(
        self,
        row: aiosqlite.Row,
        result: CoverageMembersBackfillResult,
        *,
        dry_run: bool,
    ) -> None:
        try:
            members = await self._derive_members(row)
            result.processed += 1
            if dry_run:
                return
            await self._write_members(row, members)
            result.updated += 1
        except (StructuredOutputError, ValueError):
            result.failed += 1
            logger.warning(
                "Coverage members backfill derivation failed for memory_id=%s",
                row["id"],
                exc_info=True,
            )
        except Exception:
            result.failed += 1
            logger.warning(
                "Coverage members backfill failed for memory_id=%s",
                row["id"],
                exc_info=True,
            )

    async def _derive_members(self, row: aiosqlite.Row) -> list[CoverageMember]:
        candidate = CandidateDraft(
            candidate_id=_CANDIDATE_ID,
            canonical_text=str(row["canonical_text"]),
        )
        context = ExtractionConversationContext(
            user_id=str(row["user_id"]),
            conversation_id=str(row["conversation_id"]),
            source_message_id=str(row["id"]),
            assistant_mode_id=str(row["assistant_mode_id"]),
        )
        prompt = build_enrichment_prompt(
            "coverage_members",
            message_text=str(row["canonical_text"]),
            role="user",
            context=context,
            resolved_policy=self._resolved_policy,
            allowed_write_scopes=("user",),
            occurred_at=None,
            prior_chunk_context=None,
            candidates=(candidate,),
        )
        request = LLMCompletionRequest(
            model=self._card_model,
            messages=[
                LLMMessage(
                    role="system",
                    content=_CARD_SYSTEM_PROMPTS["coverage_members"],
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_output_tokens=_CARD_MAX_OUTPUT_TOKENS["coverage_members"],
            metadata={
                "user_id": str(row["user_id"]),
                "memory_id": str(row["id"]),
                "purpose": "memory_extraction_coverage_members_card",
            },
        )
        response = await self._llm_client.complete(request)
        parsed, malformed = parse_coverage_members_card_output(response.output_text)
        if malformed:
            entry_word = "entry" if malformed == 1 else "entries"
            raise ValueError(
                f"coverage-members card returned {malformed} malformed {entry_word}"
            )
        return list(parsed.get(_CANDIDATE_ID) or ())

    async def _write_members(
        self,
        row: aiosqlite.Row,
        members: list[CoverageMember],
    ) -> None:
        payload = self._decode_payload(row["payload_json"])
        payload[_COVERAGE_MEMBERS_PAYLOAD_KEY] = [
            member.model_dump(mode="json") for member in members
        ]
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET payload_json = ?
            WHERE id = ?
              AND user_id = ?
              AND json_extract(payload_json, '$.{key}') IS NULL
            """.format(key=_COVERAGE_MEMBERS_PAYLOAD_KEY),
            (
                json_utils.dumps(payload, sort_keys=True),
                str(row["id"]),
                str(row["user_id"]),
            ),
        )
        await self._connection.commit()

    @staticmethod
    def _decode_payload(raw_payload: Any) -> dict[str, Any]:
        if raw_payload is None:
            return {}
        if isinstance(raw_payload, dict):
            return dict(raw_payload)
        decoded = json_utils.loads(str(raw_payload))
        if not isinstance(decoded, dict):
            raise ValueError("payload_json is not a JSON object")
        return decoded

    async def _emit_progress(self, result: CoverageMembersBackfillResult) -> None:
        if self._progress_callback is None:
            return
        maybe_awaitable = self._progress_callback(result.model_copy())
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
