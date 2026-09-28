"""Durable per-lane reservation at the provider dispatch boundary.

The installer wraps providers in one LLMClient, so its normal retry and
technical-recovery paths reserve each actual attempt independently. It is for
isolated benchmark processes, not the production runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator
from uuid import uuid4

from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    LLMStreamEvent,
)


class BudgetError(RuntimeError):
    """The benchmark cannot prove that another provider attempt is affordable."""


_NANODOLLARS = Decimal("1000000000")
_MILLION = Decimal("1000000")
GLOBAL_CAP_USD = Decimal("30")
_CURRENT_SLOT: ContextVar[str | None] = ContextVar("jev_benchmark_slot", default=None)


class GlobalGrantRegistry:
    """Issue immutable, disjoint lane grants within one local program cap."""

    @classmethod
    def create(cls, path: str | Path, *, program_id: str) -> "GlobalGrantRegistry":
        path = Path(path)
        if not path.is_absolute() or path.exists() or not program_id:
            raise BudgetError("A new program needs an unused absolute registry path")
        path.parent.mkdir(parents=True, exist_ok=True)
        registry = cls(path)
        try:
            with registry._db:
                registry._db.execute(
                    "INSERT INTO program (id, cap) VALUES (?, ?)",
                    (program_id, _money(GLOBAL_CAP_USD)),
                )
        except BaseException:
            registry.close()
            raise
        return registry

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise BudgetError("Grant registry path must be absolute")
        self._db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS program (id TEXT PRIMARY KEY, cap INTEGER NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS grants ("
            "assignment_id TEXT PRIMARY KEY, cap INTEGER NOT NULL, "
            "path TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL UNIQUE, "
            "state TEXT NOT NULL CHECK(state IN ('active', 'closed')), "
            "settled INTEGER NOT NULL DEFAULT 0)"
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def issue(self, path: str | Path, grant: dict[str, Any]) -> None:
        """Commit the allocation before publishing a grant file."""
        path = Path(path)
        if not path.is_absolute() or path.exists():
            raise BudgetError("Grant path must be absolute and unused")
        if path.parent.resolve() != self.path.parent.resolve():
            raise BudgetError("Grant must remain beside the program registry")
        if grant.get("registry_path") != str(self.path.resolve()):
            raise BudgetError("Grant does not name its program registry")
        if grant.get("journal_path") is None or not Path(grant["journal_path"]).is_absolute():
            raise BudgetError("Grant needs an absolute journal path")
        if Path(grant["journal_path"]).parent.resolve() != self.path.parent.resolve():
            raise BudgetError("Lane journal must remain beside the program registry")
        if not grant.get("assignment_id") or not grant.get("freeze_sha256"):
            raise BudgetError("Grant needs an assignment and frozen code")
        if not isinstance(grant.get("paid_dispatch_enabled"), bool):
            raise BudgetError("Grant needs an explicit dispatch flag")
        if not grant.get("price_verified_utc") or not grant.get("prices"):
            raise BudgetError("Grant needs verified prices")
        if not isinstance(grant.get("concurrency"), dict) or sum(
            grant["concurrency"].values()
        ) > 4 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > 2
            for value in grant["concurrency"].values()
        ):
            raise BudgetError("Concurrency exceeds the program limits")
        for price in grant["prices"]:
            if not price.get("source_url") or not price.get("verified_utc"):
                raise BudgetError("Every price needs a verifiable source and time")
            for field in ("input_per_million", "output_per_million"):
                rate = Decimal(str(price[field]))
                if not rate.is_finite() or rate < 0:
                    raise BudgetError("Invalid price")
            if int(price["context_tokens"]) <= 0:
                raise BudgetError("Missing token bound")
            if (
                Decimal(str(price["input_per_million"])) == 0
                and Decimal(str(price["output_per_million"])) == 0
                and not price.get("verified_free")
            ):
                raise BudgetError("A zero-cost route needs explicit evidence")
            for override in price.get("overrides", []):
                threshold = override.get("min_prompt_tokens")
                if (
                    not isinstance(threshold, int) or isinstance(threshold, bool)
                    or threshold < 1 or threshold > int(price["context_tokens"])
                ):
                    raise BudgetError("Invalid price-tier threshold")
                for field in ("prompt", "completion"):
                    rate = Decimal(str(override[field]))
                    if not rate.is_finite() or rate < 0:
                        raise BudgetError("Invalid price-tier rate")
        amount = _money(Decimal(str(grant["cap_usd"])))
        if amount <= 0:
            raise BudgetError("Grant cap must be positive")
        payload = (json.dumps(grant, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        self._db.execute("BEGIN IMMEDIATE")
        try:
            program = self._db.execute("SELECT cap FROM program").fetchone()
            if program is None or program[0] != _money(GLOBAL_CAP_USD):
                raise BudgetError("Program cap is missing or changed")
            allocated = self._db.execute(
                "SELECT COALESCE(SUM(CASE WHEN state='active' THEN cap ELSE settled END), 0) "
                "FROM grants"
            ).fetchone()[0]
            if allocated + amount > program[0]:
                raise BudgetError("Global program cap would be exceeded")
            self._db.execute(
                "INSERT INTO grants (assignment_id, cap, path, sha256, state) "
                "VALUES (?, ?, ?, ?, 'active')",
                (grant["assignment_id"], amount, str(path.resolve()), digest),
            )
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise
        path.write_bytes(payload)

    def validate(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        payload = path.read_bytes()
        grant = json.loads(payload)
        row = self._db.execute(
            "SELECT cap, path, sha256, state FROM grants WHERE assignment_id=?",
            (grant.get("assignment_id"),),
        ).fetchone()
        if (
            row is None
            or row[0] != _money(Decimal(str(grant["cap_usd"])))
            or row[1] != str(path.resolve())
            or row[2] != hashlib.sha256(payload).hexdigest()
            or row[3] != "active"
        ):
            raise BudgetError("Grant is absent from the program or has changed")
        program = self._db.execute("SELECT cap FROM program").fetchone()
        allocated = self._db.execute(
            "SELECT COALESCE(SUM(CASE WHEN state='active' THEN cap ELSE settled END), 0) "
            "FROM grants"
        ).fetchone()[0]
        if program is None or program[0] != _money(GLOBAL_CAP_USD) or allocated > program[0]:
            raise BudgetError("Global program allocation is invalid")
        return grant

    def close_grant(self, path: str | Path) -> int:
        """Reclaim only proven unused cap after the lane owner has exited."""
        grant = self.validate(path)
        journal = Path(grant["journal_path"])
        owner_path = journal.with_suffix(journal.suffix + ".owner")
        with owner_path.open("a+b") as owner:
            _lock_owner_handle(owner)
            spent = 0
            if journal.exists():
                with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as lane:
                    assignment = lane.execute("SELECT id, cap FROM assignment").fetchone()
                    if assignment != (
                        grant["assignment_id"], _money(Decimal(str(grant["cap_usd"])))
                    ):
                        raise BudgetError("Lane journal has a different assignment")
                    spent, pending = lane.execute(
                        "SELECT COALESCE(SUM(COALESCE(charged, reserved)), 0), "
                        "SUM(CASE WHEN status='reserved' THEN 1 ELSE 0 END) "
                        "FROM attempts"
                    ).fetchone()
                    violations = lane.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
                    if pending or violations or spent > assignment[1]:
                        raise BudgetError("Lane must be fully reconciled before closing")
            self._db.execute("BEGIN IMMEDIATE")
            try:
                current = self._db.execute(
                    "SELECT state FROM grants WHERE assignment_id=?",
                    (grant["assignment_id"],),
                ).fetchone()
                if current != ("active",):
                    raise BudgetError("Grant is already closed")
                self._db.execute(
                    "UPDATE grants SET state='closed', settled=? WHERE assignment_id=?",
                    (spent, grant["assignment_id"]),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
        return spent

    def snapshot(self) -> dict[str, Any]:
        program = self._db.execute("SELECT id, cap FROM program").fetchone()
        if program is None:
            raise BudgetError("Program has not been initialized")
        allocated, settled, active, closed = self._db.execute(
            "SELECT COALESCE(SUM(CASE WHEN state='active' THEN cap ELSE settled END), 0), "
            "COALESCE(SUM(settled), 0), SUM(state='active'), SUM(state='closed') "
            "FROM grants"
        ).fetchone()
        return {
            "program_id": program[0],
            "cap_usd": program[1] / 1e9,
            "allocated_usd": allocated / 1e9,
            "unallocated_usd": (program[1] - allocated) / 1e9,
            "settled_usd": settled / 1e9,
            "active_grants": active or 0,
            "closed_grants": closed or 0,
        }


@contextmanager
def slot_scope(slot_id: str):
    """Attribute every call, auxiliary call, and retry in one semantic slot."""
    if not slot_id:
        raise BudgetError("Benchmark slot ID is required")
    token = _CURRENT_SLOT.set(slot_id)
    try:
        yield
    finally:
        _CURRENT_SLOT.reset(token)


def _slot_for(request: LLMCompletionRequest) -> str:
    slot = _CURRENT_SLOT.get() or request.metadata.get("benchmark_slot")
    if not isinstance(slot, str) or not slot:
        raise BudgetError("Benchmark slot ID is required before dispatch")
    return slot


def _money(value: Decimal) -> int:
    if not value.is_finite() or value < 0:
        raise BudgetError("Invalid non-negative monetary amount")
    return int((value * _NANODOLLARS).to_integral_value(rounding=ROUND_CEILING))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _lock_owner_handle(handle: Any) -> None:
    """Exclude a second lane process or a concurrent grant closure."""
    if os.name == "nt":
        import msvcrt

        try:
            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BudgetError("Another process owns this assignment journal") from exc
    else:
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BudgetError("Another process owns this assignment journal") from exc


def _safe_usage(usage: dict[str, Any]) -> tuple[int, int] | None:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
        return None
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    return input_tokens, output_tokens


def _openrouter_upstream(raw_response: dict[str, Any]) -> str | None:
    metadata = raw_response.get("openrouter_metadata")
    if not isinstance(metadata, dict):
        return None
    endpoints = metadata.get("endpoints")
    available = endpoints.get("available") if isinstance(endpoints, dict) else None
    if not isinstance(available, list):
        return None
    selected = [entry for entry in available if isinstance(entry, dict) and entry.get("selected") is True]
    if len(selected) != 1:
        return None
    provider = selected[0].get("provider")
    return provider if isinstance(provider, str) and provider else None


_CAPTURE_METADATA_KEYS = frozenset({
    "benchmark_slot", "purpose", "user_id", "source_message_id",
    "service_tier", "reasoning_effort",
})
_CAPTURE_USAGE_KEYS = frozenset({
    "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens",
    "total_tokens", "prompt_tokens_details", "completion_tokens_details",
    "cost", "cost_usd", "cost_details", "is_byok",
})


def _request_payload(request: LLMCompletionRequest) -> str:
    metadata = {
        key: value for key, value in request.metadata.items()
        if key in _CAPTURE_METADATA_KEYS
    }
    extra = request.metadata.get("provider_extra_body")
    if isinstance(extra, dict) and isinstance(extra.get("reasoning"), dict):
        metadata["profile_reasoning_effort"] = extra["reasoning"].get("effort")
    return json.dumps({
        "capture_format": "ordered_typesafe_v1",
        "model": request.model,
        "messages": [message.model_dump(mode="json") for message in request.messages],
        "choice_questions": {
            key: question.model_dump(mode="json")
            for key, question in (request.choice_questions or {}).items()
        },
        "score_questions": {
            key: question.model_dump(mode="json")
            for key, question in (getattr(request, "score_questions", None) or {}).items()
        },
        "response_schema": request.response_schema,
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "external_answer": request.external_answer,
        "metadata": metadata,
    }, ensure_ascii=False, sort_keys=False)


def _response_payload(response: LLMCompletionResponse) -> str:
    raw = response.raw_response
    return json.dumps({
        "provider": response.provider,
        "model": response.model,
        "output_text": response.output_text,
        "choice_answers": {
            key: answer.model_dump(mode="json")
            for key, answer in response.choice_answers.items()
        },
        "score_answers": {
            key: answer.model_dump(mode="json")
            for key, answer in (getattr(response, "score_answers", None) or {}).items()
        },
        "usage": {
            key: value for key, value in response.usage.items()
            if key in _CAPTURE_USAGE_KEYS
        },
        "response_id": raw.get("id"),
        "finish_reason": response.finish_reason,
        "service_tier": raw.get("service_tier"),
        "upstream_name": _openrouter_upstream(raw),
    }, ensure_ascii=False, sort_keys=True)


class LaneBudget:
    """One process owns one assignment and its SQLite journal until close."""

    @classmethod
    def from_grant(
        cls, grant_path: str | Path, journal_path: str | Path, *, expected_assignment_id: str
    ) -> "LaneBudget":
        """Read one coordinator-owned grant without editing other lanes."""
        raw_grant = json.loads(Path(grant_path).read_text(encoding="utf-8"))
        registry_path = raw_grant.get("registry_path")
        if not isinstance(registry_path, str):
            raise BudgetError("Grant has no program registry")
        registry = GlobalGrantRegistry(registry_path)
        try:
            grant = registry.validate(grant_path)
        finally:
            registry.close()
        if grant.get("assignment_id") != expected_assignment_id:
            raise BudgetError("Grant assignment ID does not match the task")
        canonical_journal = grant.get("journal_path")
        if not isinstance(canonical_journal, str) or not Path(canonical_journal).is_absolute():
            raise BudgetError("Grant needs an absolute canonical journal path")
        if os.path.normcase(str(Path(canonical_journal).resolve())) != os.path.normcase(
            str(Path(journal_path).resolve())
        ):
            raise BudgetError("Journal path does not match the coordinator grant")
        if not isinstance(grant.get("paid_dispatch_enabled"), bool):
            raise BudgetError("Grant needs an explicit paid-dispatch boolean")
        if not grant.get("price_verified_utc"):
            raise BudgetError("Grant has no price verification timestamp")
        entries = grant.get("prices")
        if not isinstance(entries, list) or not entries:
            raise BudgetError("Grant has no verified priced routes")
        prices = {}
        price_tiers = {}
        for entry in entries:
            key = (entry["provider"], entry["model"])
            if key in prices:
                raise BudgetError("Duplicate priced route")
            prices[key] = (
                str(entry["input_per_million"]),
                str(entry["output_per_million"]),
                int(entry["context_tokens"]),
            )
            price_tiers[key] = tuple(sorted(
                (
                    int(override["min_prompt_tokens"]),
                    Decimal(str(override["prompt"])) * _MILLION,
                    Decimal(str(override["completion"])) * _MILLION,
                )
                for override in entry.get("overrides", [])
            ))
        budget = cls(
            journal_path,
            assignment_id=grant["assignment_id"],
            cap_usd=str(grant["cap_usd"]),
            deadline_utc=grant["deadline_utc"],
            paid_dispatch_enabled=grant["paid_dispatch_enabled"],
            concurrency=grant["concurrency"],
            prices=prices,
            price_tiers=price_tiers,
        )
        try:
            registry = GlobalGrantRegistry(registry_path)
            try:
                registry.validate(grant_path)
            finally:
                registry.close()
        except BaseException:
            budget.close()
            raise
        return budget

    def __init__(
        self,
        path: str | Path,
        *,
        assignment_id: str,
        cap_usd: str,
        deadline_utc: str,
        paid_dispatch_enabled: bool,
        concurrency: dict[str, int],
        prices: dict[tuple[str, str], tuple[str, str, int]],
        price_tiers: dict[
            tuple[str, str], tuple[tuple[int, Decimal, Decimal], ...]
        ] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.assignment_id = assignment_id
        self.cap = _money(Decimal(cap_usd))
        self.deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
        if self.deadline.tzinfo is None or self.cap <= 0 or not assignment_id:
            raise BudgetError("Invalid assignment, deadline, or cap")
        self.enabled = paid_dispatch_enabled
        self.prices = prices
        self.price_tiers = price_tiers or {}
        if sum(concurrency.values()) > 4 or any(
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 2
            for limit in concurrency.values()
        ):
            raise BudgetError("Concurrency exceeds the program limits")
        self._thread_lock = threading.RLock()
        self._owner = self.path.with_suffix(self.path.suffix + ".owner")
        self._owner_handle = self._owner.open("a+b")
        try:
            self._lock_owner()
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS assignment "
                "(id TEXT PRIMARY KEY, cap INTEGER NOT NULL, deadline TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS attempts ("
                "id TEXT PRIMARY KEY, slot TEXT NOT NULL, provider TEXT NOT NULL, "
                "model TEXT NOT NULL, status TEXT NOT NULL, reserved INTEGER NOT NULL, "
                "charged INTEGER, started_utc TEXT NOT NULL, finished_utc TEXT, "
                "duration_ms REAL, input_tokens INTEGER, output_tokens INTEGER, "
                "response_model TEXT, error_type TEXT, cache_tokens INTEGER, "
                "reported_cost INTEGER, max_output INTEGER)"
            )
            columns = {
                row[1] for row in self._db.execute("PRAGMA table_info(attempts)")
            }
            for column, sql_type in (
                ("response_tier", "TEXT"),
                ("upstream_name", "TEXT"),
                ("is_byok", "INTEGER"),
            ):
                if column not in columns:
                    self._db.execute(f"ALTER TABLE attempts ADD COLUMN {column} {sql_type}")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS violations ("
                "attempt_id TEXT PRIMARY KEY, reason TEXT NOT NULL, "
                "observed_cost INTEGER NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS attempt_payloads ("
                "attempt_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, "
                "response_json TEXT, error_json TEXT)"
            )
            row = self._db.execute("SELECT id, cap, deadline FROM assignment").fetchone()
            identity = (assignment_id, self.cap, self.deadline.isoformat())
            if row is None:
                self._db.execute("INSERT INTO assignment VALUES (?, ?, ?)", identity)
            elif row != identity:
                raise BudgetError("Journal assignment does not match the current grant")
            self._db.commit()
            self._semaphores = {
                provider: asyncio.Semaphore(limit)
                for provider, limit in concurrency.items()
                if isinstance(limit, int) and limit > 0
            }
        except BaseException:
            if hasattr(self, "_db"):
                self._db.close()
            self._owner_handle.close()
            raise

    def _lock_owner(self) -> None:
        _lock_owner_handle(self._owner_handle)

    def close(self) -> None:
        with self._thread_lock:
            self._db.close()
            self._owner_handle.close()

    def _price(self, provider: str, model: str) -> tuple[Decimal, Decimal, int]:
        entry = self.prices.get((provider, model))
        if entry is None:
            raise BudgetError(f"No verified price for {provider}/{model}")
        input_rate, output_rate, context_tokens = entry
        input_rate, output_rate = Decimal(input_rate), Decimal(output_rate)
        if (
            context_tokens <= 0 or not input_rate.is_finite()
            or not output_rate.is_finite() or input_rate < 0 or output_rate < 0
        ):
            raise BudgetError("Model context bound must be positive")
        return input_rate, output_rate, context_tokens

    def _applicable_price(
        self, provider: str, model: str, input_tokens: int
    ) -> tuple[Decimal, Decimal, int]:
        input_rate, output_rate, context_tokens = self._price(provider, model)
        for threshold, tier_input, tier_output in self.price_tiers.get((provider, model), ()):
            if input_tokens >= threshold:
                input_rate, output_rate = tier_input, tier_output
        return input_rate, output_rate, context_tokens

    def upper_bound(self, provider: str, request: LLMCompletionRequest) -> int:
        input_rate, output_rate, context_tokens = self._price(provider, request.model)
        if request.tools:
            raise BudgetError("Tools have no verified cost bound")
        if provider == "openrouter":
            body = request.metadata.get("provider_extra_body") or {}
            if not isinstance(body, dict) or set(body) - {"reasoning"}:
                raise BudgetError("OpenRouter routing or extras are outside the priced baseline")
            reasoning = body.get("reasoning")
            if reasoning is not None and (
                not isinstance(reasoning, dict)
                or set(reasoning) != {"effort"}
                or reasoning["effort"] not in {
                    "none", "minimal", "low", "medium", "high", "xhigh", "max"
                }
            ):
                raise BudgetError("Unpriced OpenRouter reasoning options")
            if request.metadata.get("service_tier") not in (None, "default", "flex", "priority"):
                raise BudgetError("OpenRouter service tier is outside the priced catalogue")
        elif request.metadata.get("provider_extra_body"):
            raise BudgetError("Provider extra body has no verified cost bound")
        if provider == "openai" and request.metadata.get("service_tier") != "default":
            raise BudgetError("OpenAI needs the explicitly priced default service tier")
        if provider not in {"openai", "openrouter"} and request.metadata.get("service_tier") is not None:
            raise BudgetError("Unpriced service tier")
        if provider == "typesafe":
            if not (request.choice_questions or getattr(request, "score_questions", None)):
                raise BudgetError("TypeSafe requires explicit bounded questions")
            if output_rate != 0:
                raise BudgetError("TypeSafe output price needs a verified token bound")
            max_output = 0
        else:
            max_output = request.max_output_tokens
            if not isinstance(max_output, int) or max_output <= 0:
                raise BudgetError("A hard max_output_tokens is required")
            if max_output > context_tokens:
                raise BudgetError("Output bound exceeds model context")
        # The provider's context ceiling bounds every billable input token,
        # including framing, serialized questions, and hidden tokenization.
        tier_rates = [(input_rate, output_rate)]
        tier_rates.extend(
            (tier_input, tier_output)
            for threshold, tier_input, tier_output in self.price_tiers.get(
                (provider, request.model), ()
            )
            if threshold <= context_tokens
        )
        return max(
            _money(
                (Decimal(context_tokens) * tier_input
                 + Decimal(max_output) * tier_output) / _MILLION
            )
            for tier_input, tier_output in tier_rates
        )

    def reserve(self, *, slot: str, provider: str, request: LLMCompletionRequest) -> str:
        if not self.enabled:
            raise BudgetError("Paid dispatch is disabled")
        if _utc_now() >= self.deadline:
            raise BudgetError("Program deadline has passed")
        if provider not in self._semaphores:
            raise BudgetError(f"No concurrency grant for {provider}")
        reserved = self.upper_bound(provider, request)
        attempt_id = uuid4().hex
        with self._thread_lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if self._db.execute("SELECT 1 FROM violations LIMIT 1").fetchone():
                    raise BudgetError("Unreconciled cost-bound violation blocks dispatch")
                committed = self._db.execute(
                    "SELECT COALESCE(SUM(COALESCE(charged, reserved)), 0) FROM attempts"
                ).fetchone()[0]
                if committed + reserved > self.cap:
                    raise BudgetError("Assignment cap would be exceeded before dispatch")
                self._db.execute(
                    "INSERT INTO attempts "
                    "(id, slot, provider, model, status, reserved, started_utc, max_output) "
                    "VALUES (?, ?, ?, ?, 'reserved', ?, ?, ?)",
                    (attempt_id, slot, provider, request.model, reserved,
                     _utc_now().isoformat(), request.max_output_tokens),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
        return attempt_id

    def record_request(self, attempt_id: str, request: LLMCompletionRequest) -> None:
        """Persist the effective request before its provider call."""
        payload = _request_payload(request)
        with self._thread_lock:
            self._db.execute(
                "INSERT INTO attempt_payloads (attempt_id, request_json) VALUES (?, ?)",
                (attempt_id, payload),
            )
            self._db.commit()

    def finish(
        self,
        attempt_id: str,
        *,
        status: str,
        duration_ms: float,
        usage: dict[str, Any] | None = None,
        response_model: str | None = None,
        response_tier: str | None = None,
        response_upstream: str | None = None,
        response_payload: str | None = None,
        error_type: str | None = None,
        error_status: int | None = None,
    ) -> None:
        if status not in {"success", "error", "cancelled"}:
            raise BudgetError("Invalid attempt status")
        with self._thread_lock:
            row = self._db.execute(
                "SELECT provider, model, reserved, status, max_output FROM attempts WHERE id=?",
                (attempt_id,),
            ).fetchone()
            if row is None or row[3] != "reserved":
                raise BudgetError("Attempt is missing or already reconciled")
            provider, model, reserved, _, max_output = row
            observed = _safe_usage(usage or {})
            raw_reported_cost = (usage or {}).get("cost_usd")
            if provider == "openrouter":
                raw_reported_cost = (usage or {}).get("cost")
            reported_cost = None
            if raw_reported_cost is not None:
                try:
                    reported_cost = _money(Decimal(str(raw_reported_cost)))
                except Exception as exc:
                    raise BudgetError("Invalid provider-reported cost") from exc
            measured = None
            violation_reasons = []
            if observed is not None:
                input_rate, output_rate, context_tokens = self._applicable_price(
                    provider, model, observed[0]
                )
                output_within_limit = (
                    provider == "typesafe"
                    or (max_output is not None and observed[1] <= max_output)
                )
                measured = _money(
                    (Decimal(observed[0]) * input_rate + Decimal(observed[1]) * output_rate)
                    / _MILLION
                )
                if observed[0] > context_tokens or not output_within_limit:
                    violation_reasons.append("Provider usage exceeded a token bound")
            known_cost = max(
                value for value in (measured, reported_cost, 0) if value is not None
            )
            if known_cost > reserved:
                violation_reasons.append("Observed cost exceeded the reservation")
            if response_model is not None and response_model != model:
                violation_reasons.append("Provider returned an unexpected model")
            if status == "success" and provider == "openai" and response_tier != "default":
                violation_reasons.append("Provider did not confirm the default service tier")
            if status == "success" and provider == "openrouter":
                if response_tier not in (None, "default", "flex", "priority"):
                    violation_reasons.append("OpenRouter returned an unpriced tier")
                if raw_reported_cost is None:
                    violation_reasons.append("OpenRouter did not report billed cost")
                if (usage or {}).get("is_byok") is not False:
                    violation_reasons.append("OpenRouter BYOK status was not confirmed false")
            charged = reserved
            if status == "success" and observed is not None and not violation_reasons:
                charged = known_cost
            elif violation_reasons:
                charged = max(reserved, known_cost)
            cache = (usage or {}).get("prompt_tokens_details", {})
            cache_tokens = cache.get("cached_tokens") if isinstance(cache, dict) else None
            self._db.execute(
                "UPDATE attempts SET status=?, charged=?, finished_utc=?, "
                "duration_ms=?, input_tokens=?, output_tokens=?, response_model=?, "
                "error_type=?, cache_tokens=?, reported_cost=?, response_tier=?, "
                "upstream_name=?, is_byok=? WHERE id=?",
                (
                    status, charged, _utc_now().isoformat(), duration_ms,
                    observed[0] if observed else None,
                    observed[1] if observed else None,
                    response_model, error_type,
                    cache_tokens if isinstance(cache_tokens, int) else None,
                    reported_cost, response_tier, response_upstream,
                    int(usage["is_byok"]) if isinstance((usage or {}).get("is_byok"), bool) else None,
                    attempt_id,
                ),
            )
            self._db.execute(
                "UPDATE attempt_payloads SET response_json=?, error_json=? WHERE attempt_id=?",
                (
                    response_payload,
                    json.dumps({"type": error_type, "status_code": error_status})
                    if error_type else None,
                    attempt_id,
                ),
            )
            if violation_reasons:
                self._db.execute(
                    "INSERT INTO violations VALUES (?, ?, ?)",
                    (attempt_id, "; ".join(violation_reasons), known_cost),
                )
            self._db.commit()
            if violation_reasons:
                raise BudgetError("Cost-bound violation persisted; dispatch blocked")

    def snapshot(self) -> dict[str, Any]:
        with self._thread_lock:
            total, count = self._db.execute(
                "SELECT COALESCE(SUM(COALESCE(charged, reserved)), 0), COUNT(*) FROM attempts"
            ).fetchone()
            return {
                "assignment_id": self.assignment_id,
                "cap_usd": self.cap / 1e9,
                "committed_usd": total / 1e9,
                "remaining_usd": (self.cap - total) / 1e9,
                "attempts": count,
                "violations": self._db.execute("SELECT COUNT(*) FROM violations").fetchone()[0],
            }

    def slot_attempt_statuses(self, slot: str) -> list[str]:
        """Expose transport outcomes without reading private request payloads."""
        with self._thread_lock:
            return [
                str(row[0]) for row in self._db.execute(
                    "SELECT status FROM attempts WHERE slot=? ORDER BY started_utc, id",
                    (slot,),
                ).fetchall()
            ]


class _BudgetedProvider(LLMProvider):
    def __init__(self, inner: LLMProvider, budget: LaneBudget) -> None:
        self.inner = inner
        self.budget = budget
        self.name = inner.name
        self.supports_embeddings = inner.supports_embeddings
        self.supports_embedding_dimensions = inner.supports_embedding_dimensions
        self.supports_native_structured_output = inner.supports_native_structured_output
        self.supports_choices = inner.supports_choices
        self.supports_scores = getattr(inner, "supports_scores", False)

    def supports_native_structured_output_for(self, request: LLMCompletionRequest) -> bool:
        return self.inner.supports_native_structured_output_for(request)

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        semaphore = self.budget._semaphores.get(self.name)
        if semaphore is None:
            raise BudgetError(f"No concurrency grant for {self.name}")
        async with semaphore:
            attempt_id = self.budget.reserve(
                slot=_slot_for(request),
                provider=self.name,
                request=request,
            )
            self.budget.record_request(attempt_id, request)
            started = perf_counter()
            try:
                response = await self.inner.complete(request)
            except BaseException as exc:
                self.budget.finish(
                    attempt_id,
                    status="cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                    duration_ms=(perf_counter() - started) * 1000,
                    error_type=type(exc).__name__,
                    error_status=getattr(exc, "status_code", None),
                )
                raise
            self.budget.finish(
                attempt_id, status="success", duration_ms=(perf_counter() - started) * 1000,
                usage=response.usage, response_model=response.model,
                response_tier=response.raw_response.get("service_tier"),
                response_upstream=_openrouter_upstream(response.raw_response)
                if self.name == "openrouter" else None,
                response_payload=_response_payload(response),
            )
            return response

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise BudgetError("Embedding route has no verified reservation contract")

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        raise BudgetError("Streaming attempts have no complete evidence capture")
        yield  # pragma: no cover - async iterator contract


def install_budget(client: LLMClient[Any], budget: LaneBudget) -> None:
    """Wrap every currently registered provider once, before any benchmark call."""
    for name, provider in list(client._providers.items()):
        if isinstance(provider, _BudgetedProvider):
            raise BudgetError("Budget already installed")
        client._providers[name] = _BudgetedProvider(provider, budget)
