"""Offline admission and provider-attempt accounting checks."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from atagia.models.schemas_decisions import ChoiceQuestion
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMMessage,
    LLMProvider,
    RetryPolicy,
    TransientLLMError,
)
from benchmarks.jev_friendly_cards.budget import (
    BudgetError,
    GlobalGrantRegistry,
    LaneBudget,
    install_budget,
    slot_scope,
    _request_payload,
)


def _grant(tmp_path: Path, *, cap: str = "2", provider: str = "openrouter") -> tuple[Path, Path]:
    registry_path = tmp_path / "program.sqlite"
    registry = GlobalGrantRegistry.create(registry_path, program_id="fresh-program")
    path = tmp_path / "grant.json"
    model = "openai/test" if provider == "openrouter" else "jev-test"
    registry.issue(
        path,
        {
            "registry_path": str(registry_path.resolve()),
            "assignment_id": "new-assignment",
            "journal_path": str((tmp_path / "lane.sqlite").resolve()),
            "freeze_sha256": "f" * 64,
            "cap_usd": cap,
            "deadline_utc": "2099-01-01T00:00:00Z",
            "paid_dispatch_enabled": True,
            "price_verified_utc": "2026-09-26T00:00:00Z",
            "concurrency": {provider: 2},
            "prices": [
                {
                    "provider": provider,
                    "model": model,
                    "input_per_million": "1000",
                    "output_per_million": "0",
                    "context_tokens": 1000,
                    "source_url": "https://example.test/prices",
                    "verified_utc": "2026-09-26T00:00:00Z",
                }
            ],
        },
    )
    registry.close()
    return path, tmp_path / "lane.sqlite"


def _request(model: str = "openai/test") -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model=model,
        messages=[LLMMessage(role="user", content="A bounded offline test")],
        max_output_tokens=10,
        external_answer=True,
    )


def test_global_grants_do_not_overlap_under_race(tmp_path: Path) -> None:
    registry_path = tmp_path / "program.sqlite"
    registry = GlobalGrantRegistry.create(registry_path, program_id="fresh-program")
    registry.close()

    def allocate(number: int) -> str:
        current = GlobalGrantRegistry(registry_path)
        try:
            grant = {
                "registry_path": str(registry_path.resolve()),
                "assignment_id": f"assignment-{number}",
                "journal_path": str((tmp_path / f"lane-{number}.sqlite").resolve()),
                "freeze_sha256": "f" * 64,
                "cap_usd": "20",
                "deadline_utc": "2099-01-01T00:00:00Z",
                "paid_dispatch_enabled": True,
                "price_verified_utc": "2026-09-26T00:00:00Z",
                "concurrency": {"typesafe": 1},
                "prices": [{
                    "provider": "typesafe", "model": "jev-test",
                    "input_per_million": "1", "output_per_million": "0",
                    "context_tokens": 1000,
                    "source_url": "https://example.test/prices",
                    "verified_utc": "2026-09-26T00:00:00Z",
                }],
            }
            current.issue(tmp_path / f"grant-{number}.json", grant)
            return "issued"
        except BudgetError:
            return "blocked"
        finally:
            current.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(allocate, (1, 2)))
    assert sorted(outcomes) == ["blocked", "issued"]
    current = GlobalGrantRegistry(registry_path)
    assert current.snapshot() == {
        "program_id": "fresh-program", "cap_usd": 30.0,
        "allocated_usd": 20.0, "unallocated_usd": 10.0,
        "settled_usd": 0.0, "active_grants": 1, "closed_grants": 0,
    }
    current.close()


def test_attempts_keep_unknown_cost_and_block_budget(tmp_path: Path) -> None:
    grant, journal = _grant(tmp_path)
    budget = LaneBudget.from_grant(grant, journal, expected_assignment_id="new-assignment")
    try:
        first = budget.reserve(slot="slot-a", provider="openrouter", request=_request())
        second = budget.reserve(slot="slot-b", provider="openrouter", request=_request())
        budget.finish(first, status="error", duration_ms=50)
        budget.finish(second, status="cancelled", duration_ms=50)
        assert budget.snapshot()["committed_usd"] == 2.0
        with pytest.raises(BudgetError, match="cap"):
            budget.reserve(slot="slot-c", provider="openrouter", request=_request())
        with pytest.raises(BudgetError, match="verified price"):
            budget.reserve(slot="slot-c", provider="openrouter", request=_request("other"))
    finally:
        budget.close()


def test_high_context_price_tier_is_reserved_and_settled(tmp_path: Path) -> None:
    registry_path = tmp_path / "program.sqlite"
    registry = GlobalGrantRegistry.create(registry_path, program_id="tier-program")
    grant_path = tmp_path / "tier-grant.json"
    journal = tmp_path / "tier-lane.sqlite"
    registry.issue(grant_path, {
        "registry_path": str(registry_path.resolve()),
        "assignment_id": "tier-assignment",
        "journal_path": str(journal.resolve()),
        "freeze_sha256": "f" * 64,
        "cap_usd": "1",
        "deadline_utc": "2099-01-01T00:00:00Z",
        "paid_dispatch_enabled": True,
        "price_verified_utc": "2026-09-26T00:00:00Z",
        "concurrency": {"openrouter": 1},
        "prices": [{
            "provider": "openrouter", "model": "openai/test",
            "input_per_million": "0.1", "output_per_million": "0.5",
            "context_tokens": 1_000_000,
            "source_url": "https://example.test/prices",
            "verified_utc": "2026-09-26T00:00:00Z",
            "overrides": [{
                "min_prompt_tokens": 272_000,
                "prompt": "0.0000002", "completion": "0.00000075",
            }],
        }],
    })
    registry.close()
    budget = LaneBudget.from_grant(
        grant_path, journal, expected_assignment_id="tier-assignment"
    )
    try:
        assert budget.upper_bound("openrouter", _request()) == 200_007_500
        attempt = budget.reserve(
            slot="tier-slot", provider="openrouter", request=_request()
        )
        budget.finish(
            attempt, status="success", duration_ms=1,
            usage={
                "input_tokens": 300_000, "output_tokens": 10,
                "cost": 0.0600075, "is_byok": False,
            },
            response_model="openai/test", response_tier="default",
        )
        assert budget.snapshot()["committed_usd"] == pytest.approx(0.0600075)
    finally:
        budget.close()


def test_mutated_grant_is_rejected_before_journal_open(tmp_path: Path) -> None:
    grant, journal = _grant(tmp_path)
    data = json.loads(grant.read_text(encoding="utf-8"))
    data["cap_usd"] = "30"
    grant.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BudgetError, match="changed"):
        LaneBudget.from_grant(grant, journal, expected_assignment_id="new-assignment")
    assert not journal.exists()


def test_closed_grant_reclaims_only_reconciled_unused_cap(tmp_path: Path) -> None:
    grant_path, journal = _grant(tmp_path)
    budget = LaneBudget.from_grant(
        grant_path, journal, expected_assignment_id="new-assignment"
    )
    attempt = budget.reserve(slot="slot-a", provider="openrouter", request=_request())
    budget.finish(attempt, status="error", duration_ms=25)
    registry = GlobalGrantRegistry(tmp_path / "program.sqlite")
    try:
        with pytest.raises(BudgetError, match="owns"):
            registry.close_grant(grant_path)
        budget.close()
        assert registry.close_grant(grant_path) == 1_000_000_000
        assert registry.snapshot()["unallocated_usd"] == 29.0
        with pytest.raises(BudgetError, match="changed"):
            LaneBudget.from_grant(
                grant_path, journal, expected_assignment_id="new-assignment"
            )
        next_grant = json.loads(grant_path.read_text(encoding="utf-8"))
        next_grant["assignment_id"] = "next-assignment"
        next_grant["journal_path"] = str((tmp_path / "next-lane.sqlite").resolve())
        next_grant["cap_usd"] = "29"
        registry.issue(tmp_path / "next-grant.json", next_grant)
        assert registry.snapshot()["allocated_usd"] == 30.0
    finally:
        registry.close()


def test_grant_closed_between_validation_and_owner_lock_cannot_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grant_path, journal = _grant(tmp_path)
    original_init = LaneBudget.__init__

    def close_before_lock(self: LaneBudget, *args: object, **kwargs: object) -> None:
        registry = GlobalGrantRegistry(tmp_path / "program.sqlite")
        try:
            assert registry.close_grant(grant_path) == 0
        finally:
            registry.close()
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(LaneBudget, "__init__", close_before_lock)
    with pytest.raises(BudgetError, match="changed"):
        LaneBudget.from_grant(
            grant_path, journal, expected_assignment_id="new-assignment"
        )
    registry = GlobalGrantRegistry(tmp_path / "program.sqlite")
    assert registry.snapshot()["allocated_usd"] == 0.0
    registry.close()


class _RetryProvider(LLMProvider):
    name = "openrouter"
    supports_scores = True

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        if self.calls == 1:
            raise TransientLLMError("Simulated timeout")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text="ok",
            usage={"input_tokens": 1, "output_tokens": 1, "cost": 0.001, "is_byok": False},
            raw_response={"service_tier": "default"},
        )


@pytest.mark.asyncio
async def test_retry_reserves_each_provider_attempt(tmp_path: Path) -> None:
    grant, journal = _grant(tmp_path)
    budget = LaneBudget.from_grant(grant, journal, expected_assignment_id="new-assignment")
    provider = _RetryProvider()
    client = LLMClient(
        providers=[provider],
        retry_policy=RetryPolicy(attempts=2, base_delay_seconds=0, max_delay_seconds=0),
        extraction_retry_policy=RetryPolicy(attempts=2, base_delay_seconds=0, max_delay_seconds=0),
    )
    install_budget(client, budget)
    assert client._providers["openrouter"].supports_scores is True
    try:
        with slot_scope("slot-a"):
            result = await client.complete(_request("openrouter/openai/test"))
        assert result.output_text == "ok"
        assert provider.calls == 2
        assert budget.snapshot()["attempts"] == 2
        assert budget.snapshot()["committed_usd"] >= 1.0
    finally:
        await client.aclose()
        budget.close()


@pytest.mark.asyncio
async def test_cancelled_provider_keeps_reservation(tmp_path: Path) -> None:
    grant, journal = _grant(tmp_path, provider="typesafe")
    budget = LaneBudget.from_grant(grant, journal, expected_assignment_id="new-assignment")
    entered = asyncio.Event()

    class WaitingProvider(LLMProvider):
        name = "typesafe"
        supports_choices = True

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("Unreachable")

    client = LLMClient(providers=[WaitingProvider()], retry_policy=RetryPolicy(attempts=1))
    install_budget(client, budget)
    request = _request("typesafe/jev-test").model_copy(update={
        "choice_questions": {"q": ChoiceQuestion(instructions="Choose.", criteria={"a": "A", "b": "B"})}
    })
    try:
        with slot_scope("slot-a"):
            task = asyncio.create_task(client.complete(request))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        snapshot = budget.snapshot()
        assert snapshot["attempts"] == 1
        assert snapshot["committed_usd"] == 1.0
    finally:
        await client.aclose()
        budget.close()


def test_request_capture_preserves_question_and_option_order() -> None:
    request = _request("jev-test")
    request.choice_questions = {
        "q_z": ChoiceQuestion(instructions="Choose Z.", criteria={"z": None, "a": "A"}),
        "q_a": ChoiceQuestion(instructions="Choose A.", criteria={"no": None, "yes": None}),
    }
    captured = json.loads(_request_payload(request))
    assert captured["capture_format"] == "ordered_typesafe_v1"
    assert list(captured["choice_questions"]) == ["q_z", "q_a"]
    assert list(captured["choice_questions"]["q_z"]["criteria"]) == ["z", "a"]
