"""Backfill content-language metadata for existing memory rows."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import html
import inspect
import logging
from typing import Any, Awaitable, Callable

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from atagia.core import json_utils
from atagia.core.config import Settings
from atagia.core.language_codes import (
    ISO_639_1_LANGUAGE_CODES,
    normalize_optional_iso_639_1_code,
)
from atagia.models.schemas_memory import MemoryStatus
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMMessage,
    StructuredOutputError,
)
from atagia.services.model_resolution import resolve_component_model


logger = logging.getLogger(__name__)

CONTENT_LANGUAGE_BACKFILL_MAX_OUTPUT_TOKENS = 8192
_BACKFILL_STATUSES = frozenset(
    {
        MemoryStatus.ACTIVE.value,
        MemoryStatus.SUPERSEDED.value,
        MemoryStatus.REVIEW_REQUIRED.value,
        MemoryStatus.PENDING_USER_CONFIRMATION.value,
    }
)
ProgressCallback = Callable[["ContentLanguageBackfillResult"], Awaitable[None] | None]

CONTENT_LANGUAGE_CLASSIFICATION_PROMPT_TEMPLATE = """Classify content-language metadata for one assistant memory row.

Return JSON only, matching the provided schema exactly.
Do not include markdown fences, preambles, tags, or explanations.

Task:
- Identify the ISO 639-1 language code(s) of the language actually used in
  <canonical_text>.
- Do not translate the text.
- Do not infer the user's language ability or preference.
- Use multiple language codes only when <canonical_text> genuinely mixes
  languages.
- If the language cannot be identified safely, return an empty language_codes
  list and a low confidence.
- Do not use dataset-specific, benchmark-specific, or persona-specific
  assumptions.

<canonical_text>
{canonical_text}
</canonical_text>

<index_text>
{index_text}
</index_text>
"""


class ContentLanguageClassification(BaseModel):
    """LLM result for one content-language classification."""

    model_config = ConfigDict(extra="forbid")

    language_codes: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("language_codes", mode="before")
    @classmethod
    def validate_language_codes(cls, values: Any) -> list[str]:
        if values is None:
            return []
        raw_values = values if isinstance(values, list) else [values]
        normalized: list[str] = []
        seen: set[str] = set()
        for value in raw_values:
            code = normalize_optional_iso_639_1_code(value)
            if code is None:
                continue
            if code in seen:
                continue
            seen.add(code)
            normalized.append(code)
        return normalized


class ContentLanguageBackfillResult(BaseModel):
    """Counters and parameters for one content-language backfill run."""

    model_config = ConfigDict(extra="forbid")

    examined: int = 0
    classified: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    dry_run: bool
    batch_size: int = Field(ge=1)
    delay_ms: int = Field(ge=0)
    user_id: str | None = None


class ContentLanguageBackfillService:
    """Scan memory rows missing language metadata and backfill them."""

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
        self._classification_model = resolve_component_model(
            self._settings,
            "extractor",
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
        min_confidence: float = 0.45,
    ) -> ContentLanguageBackfillResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        result = ContentLanguageBackfillResult(
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
                    await self._process_row(
                        row,
                        result,
                        dry_run=dry_run,
                        min_confidence=min_confidence,
                    )
                await self._emit_progress(result)
                rows = await cursor.fetchmany(batch_size)
                if rows and delay_ms > 0:
                    await self._sleep(delay_ms / 1000.0)
        finally:
            await cursor.close()
        return result

    def _scan_query(self, user_id: str | None) -> tuple[str, tuple[Any, ...]]:
        valid_code_placeholders = ", ".join("?" for _code in ISO_639_1_LANGUAGE_CODES)
        clauses = [
            "mo.status IN ({statuses})".format(
                statuses=", ".join("?" for _status in _BACKFILL_STATUSES)
            ),
            """
            CASE
                WHEN mo.language_codes_json IS NULL THEN 1
                WHEN TRIM(mo.language_codes_json) = '' THEN 1
                WHEN json_valid(mo.language_codes_json) = 0 THEN 1
                WHEN json_valid(mo.language_codes_json) = 1
                 AND COALESCE(json_type(mo.language_codes_json), '') != 'array' THEN 1
                WHEN json_array_length(
                    CASE
                        WHEN json_valid(mo.language_codes_json) = 1
                         AND json_type(mo.language_codes_json) = 'array'
                        THEN mo.language_codes_json
                        ELSE '[]'
                    END
                ) = 0 THEN 1
                WHEN EXISTS (
                    SELECT 1
                    FROM json_each(
                        CASE
                            WHEN json_valid(mo.language_codes_json) = 1
                             AND json_type(mo.language_codes_json) = 'array'
                            THEN mo.language_codes_json
                            ELSE '[]'
                        END
                    ) AS language_code
                    WHERE language_code.type != 'text'
                       OR LOWER(language_code.value) NOT IN ({valid_codes})
                ) THEN 1
                ELSE 0
            END = 1
            """.format(valid_codes=valid_code_placeholders),
        ]
        parameters: list[Any] = [
            *sorted(_BACKFILL_STATUSES),
            *sorted(ISO_639_1_LANGUAGE_CODES),
        ]
        if user_id is not None:
            clauses.append("mo.user_id = ?")
            parameters.append(user_id)
        return (
            """
            SELECT
                mo.id,
                mo.user_id,
                mo.canonical_text,
                mo.index_text,
                mo.status,
                mo.updated_at
            FROM memory_objects AS mo
            WHERE {where_clause}
            ORDER BY mo.created_at ASC, mo.id ASC
            """.format(where_clause=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def _process_row(
        self,
        row: aiosqlite.Row,
        result: ContentLanguageBackfillResult,
        *,
        dry_run: bool,
        min_confidence: float,
    ) -> None:
        try:
            classification = await self._classify_row(row)
            if not classification.language_codes or classification.confidence < min_confidence:
                result.skipped += 1
                return
            result.classified += 1
            if dry_run:
                return
            await self._write_language_codes(row, classification.language_codes)
            result.updated += 1
        except (StructuredOutputError, ValueError):
            result.failed += 1
            logger.warning(
                "Content language backfill classification failed for memory_id=%s",
                row["id"],
                exc_info=True,
            )
        except Exception:
            result.failed += 1
            logger.warning(
                "Content language backfill failed for memory_id=%s",
                row["id"],
                exc_info=True,
            )

    async def _classify_row(
        self,
        row: aiosqlite.Row,
    ) -> ContentLanguageClassification:
        prompt = CONTENT_LANGUAGE_CLASSIFICATION_PROMPT_TEMPLATE.format(
            canonical_text=html.escape(str(row["canonical_text"])),
            index_text=(
                html.escape(str(row["index_text"]))
                if row["index_text"] is not None
                else "(none)"
            ),
        )
        request = LLMCompletionRequest(
            model=self._classification_model,
            messages=[
                LLMMessage(
                    role="system",
                    content="Classify content-language metadata as JSON only.",
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_output_tokens=CONTENT_LANGUAGE_BACKFILL_MAX_OUTPUT_TOKENS,
            response_schema=TypeAdapter(ContentLanguageClassification).json_schema(),
            metadata={
                "user_id": str(row["user_id"]),
                "memory_id": str(row["id"]),
                "purpose": "content_language_backfill",
            },
        )
        return await self._llm_client.complete_structured(
            request,
            ContentLanguageClassification,
        )

    async def _write_language_codes(
        self,
        row: aiosqlite.Row,
        language_codes: list[str],
    ) -> None:
        valid_code_placeholders = ", ".join("?" for _code in ISO_639_1_LANGUAGE_CODES)
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET language_codes_json = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND CASE
                    WHEN language_codes_json IS NULL THEN 1
                    WHEN TRIM(language_codes_json) = '' THEN 1
                    WHEN json_valid(language_codes_json) = 0 THEN 1
                    WHEN json_valid(language_codes_json) = 1
                     AND COALESCE(json_type(language_codes_json), '') != 'array' THEN 1
                    WHEN json_array_length(
                        CASE
                            WHEN json_valid(language_codes_json) = 1
                             AND json_type(language_codes_json) = 'array'
                            THEN language_codes_json
                            ELSE '[]'
                        END
                    ) = 0 THEN 1
                    WHEN EXISTS (
                        SELECT 1
                        FROM json_each(
                            CASE
                                WHEN json_valid(memory_objects.language_codes_json) = 1
                                 AND json_type(memory_objects.language_codes_json) = 'array'
                                THEN memory_objects.language_codes_json
                                ELSE '[]'
                            END
                        ) AS language_code
                        WHERE language_code.type != 'text'
                           OR LOWER(language_code.value) NOT IN ({valid_codes})
                    ) THEN 1
                    ELSE 0
                  END = 1
            """.format(valid_codes=valid_code_placeholders),
            (
                json_utils.dumps(language_codes),
                datetime.now(tz=timezone.utc).isoformat(),
                str(row["id"]),
                str(row["user_id"]),
                *sorted(ISO_639_1_LANGUAGE_CODES),
            ),
        )
        await self._connection.commit()

    async def _emit_progress(self, result: ContentLanguageBackfillResult) -> None:
        if self._progress_callback is None:
            return
        maybe_awaitable = self._progress_callback(result.model_copy())
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
