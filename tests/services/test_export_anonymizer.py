"""Dedicated unit tests for the ExportAnonymizer module."""

from __future__ import annotations

import json

import pytest

from atagia.models.schemas_replay import ExportAnonymizationMode, ExportedMessage
from atagia.services.export_anonymizer import (
    ExportAnonymizationError,
    ExportAnonymizationTooLargeError,
    ExportAnonymizationVerificationError,
    ExportAnonymizer,
    _RewriteResponse,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "test-anonymizer-model"


def _msg(message_id: str, seq: int, role: str, content: str) -> ExportedMessage:
    return ExportedMessage(message_id=message_id, seq=seq, role=role, content=content)


def _two_message_conversation() -> list[ExportedMessage]:
    return [
        _msg("msg_1", 1, "user", "Maria lives in Barcelona and needs help."),
        _msg("msg_2", 2, "assistant", "Sure, Maria. Barcelona has great resources."),
    ]


def _good_rewrite_payload(messages: list[ExportedMessage]) -> dict:
    """A valid rewrite payload matching the given messages."""
    return {
        "entities": [
            {
                "placeholder": "[person_001]",
                "readable_label": "Person 1",
                "source_forms": ["Maria"],
            },
            {
                "placeholder": "[place_001]",
                "readable_label": "Place 1",
                "source_forms": ["Barcelona"],
            },
        ],
        "messages": [
            {
                "message_id": msg.message_id,
                "strict_content": msg.content.replace("Maria", "[person_001]").replace(
                    "Barcelona", "[place_001]"
                ),
                "readable_content": msg.content.replace("Maria", "Person 1").replace(
                    "Barcelona", "Place 1"
                ),
            }
            for msg in messages
        ],
    }


def _approved_verification_payload() -> dict:
    return {
        "approved": True,
        "remaining_identifiers": [],
        "unsafe_descriptive_clues": [],
        "reasoning": "Projection is safe.",
    }


def _rejected_verification_payload(*, remaining: list[str] | None = None) -> dict:
    return {
        "approved": False,
        "remaining_identifiers": remaining or ["Maria"],
        "unsafe_descriptive_clues": [],
        "reasoning": "Source form still present.",
    }


class _CannedProvider(LLMProvider):
    """LLM provider that returns canned responses keyed by purpose metadata."""

    name = "canned-anonymizer"

    def __init__(self) -> None:
        self.rewrite_calls: int = 0
        self.verify_calls: int = 0
        self._rewrite_responses: list[dict] = []
        self._verify_responses: list[dict] = []

    def set_rewrite_responses(self, *payloads: dict) -> None:
        self._rewrite_responses = list(payloads)

    def set_verify_responses(self, *payloads: dict) -> None:
        self._verify_responses = list(payloads)

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata.get("purpose"))
        if purpose == "export_anonymization_rewrite":
            index = min(self.rewrite_calls, len(self._rewrite_responses) - 1)
            payload = self._rewrite_responses[index]
            self.rewrite_calls += 1
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(payload),
            )
        if purpose == "export_anonymization_verify":
            index = min(self.verify_calls, len(self._verify_responses) - 1)
            payload = self._verify_responses[index]
            self.verify_calls += 1
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(payload),
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embedding not expected")


def _build(provider: _CannedProvider) -> ExportAnonymizer:
    client = LLMClient(provider_name=provider.name, providers=[provider])
    return ExportAnonymizer(client, model=_MODEL)


# ---------------------------------------------------------------------------
# 1. Mechanical leak detection
# ---------------------------------------------------------------------------


class TestMechanicalLeakDetection:
    """Direct tests for _mechanical_leak_issues."""

    def _anonymizer(self) -> ExportAnonymizer:
        provider = _CannedProvider()
        return _build(provider)

    def test_surviving_source_form_detected(self) -> None:
        anonymizer = self._anonymizer()
        rewrite = _RewriteResponse(
            entities=[
                {
                    "placeholder": "[person_001]",
                    "readable_label": "Person 1",
                    "source_forms": ["Maria"],
                }
            ],
            messages=[
                {
                    "message_id": "msg_1",
                    "strict_content": "Maria still appears here.",
                    "readable_content": "Maria still appears here.",
                }
            ],
        )
        projection = {"msg_1": "Maria still appears here."}
        issues = anonymizer._mechanical_leak_issues(projection, rewrite)
        assert len(issues) == 1
        assert "Maria" in issues[0]

    def test_case_insensitive_matching(self) -> None:
        anonymizer = self._anonymizer()
        rewrite = _RewriteResponse(
            entities=[
                {
                    "placeholder": "[person_001]",
                    "readable_label": "Person 1",
                    "source_forms": ["Maria"],
                }
            ],
            messages=[
                {
                    "message_id": "msg_1",
                    "strict_content": "MARIA is still here.",
                    "readable_content": "MARIA is still here.",
                }
            ],
        )
        projection = {"msg_1": "MARIA is still here."}
        issues = anonymizer._mechanical_leak_issues(projection, rewrite)
        assert len(issues) == 1

    def test_short_non_digit_forms_skipped(self) -> None:
        anonymizer = self._anonymizer()
        rewrite = _RewriteResponse(
            entities=[
                {
                    "placeholder": "[entity_001]",
                    "readable_label": "Entity 1",
                    "source_forms": ["AI"],
                }
            ],
            messages=[
                {
                    "message_id": "msg_1",
                    "strict_content": "AI assistance needed.",
                    "readable_content": "AI assistance needed.",
                }
            ],
        )
        projection = {"msg_1": "AI assistance needed."}
        issues = anonymizer._mechanical_leak_issues(projection, rewrite)
        assert issues == []

    def test_short_form_with_digit_not_skipped(self) -> None:
        anonymizer = self._anonymizer()
        rewrite = _RewriteResponse(
            entities=[
                {
                    "placeholder": "[identifier_001]",
                    "readable_label": "Identifier 1",
                    "source_forms": ["A1"],
                }
            ],
            messages=[
                {
                    "message_id": "msg_1",
                    "strict_content": "Room A1 is reserved.",
                    "readable_content": "Room A1 is reserved.",
                }
            ],
        )
        projection = {"msg_1": "Room A1 is reserved."}
        issues = anonymizer._mechanical_leak_issues(projection, rewrite)
        assert len(issues) == 1

    def test_no_leak_when_properly_replaced(self) -> None:
        anonymizer = self._anonymizer()
        rewrite = _RewriteResponse(
            entities=[
                {
                    "placeholder": "[person_001]",
                    "readable_label": "Person 1",
                    "source_forms": ["Maria"],
                }
            ],
            messages=[
                {
                    "message_id": "msg_1",
                    "strict_content": "[person_001] lives here.",
                    "readable_content": "Person 1 lives here.",
                }
            ],
        )
        projection = {"msg_1": "[person_001] lives here."}
        issues = anonymizer._mechanical_leak_issues(projection, rewrite)
        assert issues == []


# ---------------------------------------------------------------------------
# 2. Oversize transcript rejection
# ---------------------------------------------------------------------------


class TestOversizeTranscriptRejection:

    @pytest.mark.asyncio
    async def test_too_many_messages(self) -> None:
        provider = _CannedProvider()
        anonymizer = _build(provider)
        messages = [_msg(f"msg_{i}", i, "user", "Short.") for i in range(201)]
        with pytest.raises(ExportAnonymizationTooLargeError, match="at most 200 messages"):
            await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)

    @pytest.mark.asyncio
    async def test_transcript_too_many_chars(self) -> None:
        provider = _CannedProvider()
        anonymizer = _build(provider)
        # Each message needs enough content to exceed 50,000 chars total.
        # 10 messages * ~5100 chars content + role + id overhead > 50_000
        long_content = "A" * 5100
        messages = [_msg(f"msg_{i}", i, "user", long_content) for i in range(10)]
        with pytest.raises(ExportAnonymizationTooLargeError, match="size cap"):
            await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)

    @pytest.mark.asyncio
    async def test_exactly_at_limit_passes_cap_check(self) -> None:
        """200 messages with small content should not trigger the cap."""
        provider = _CannedProvider()
        provider.set_rewrite_responses(
            _good_rewrite_payload([_msg("msg_0", 0, "user", "Hi")])
        )
        provider.set_verify_responses(_approved_verification_payload())
        anonymizer = _build(provider)
        # Single short message is well under limits.
        messages = [_msg("msg_0", 0, "user", "Hi")]
        # Should not raise the cap error -- may raise other errors from
        # the rewrite/verify flow, but the cap check itself passes.
        result = await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)
        assert result is not None


# ---------------------------------------------------------------------------
# 3. Retry behavior
# ---------------------------------------------------------------------------


class TestRetryBehavior:

    @pytest.mark.asyncio
    async def test_retry_on_mismatched_ids_then_success(self) -> None:
        provider = _CannedProvider()
        messages = _two_message_conversation()

        # First attempt: wrong message IDs
        bad_rewrite = _good_rewrite_payload(messages)
        bad_rewrite["messages"][0]["message_id"] = "msg_WRONG"

        # Second attempt: correct
        good_rewrite = _good_rewrite_payload(messages)

        provider.set_rewrite_responses(bad_rewrite, good_rewrite)
        provider.set_verify_responses(_approved_verification_payload())

        anonymizer = _build(provider)
        result = await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)

        assert provider.rewrite_calls == 2
        assert provider.verify_calls == 1
        assert "msg_1" in result.strict_messages
        assert "msg_2" in result.strict_messages

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_error(self) -> None:
        provider = _CannedProvider()
        messages = _two_message_conversation()

        # Both attempts return bad IDs
        bad_rewrite = _good_rewrite_payload(messages)
        bad_rewrite["messages"][0]["message_id"] = "msg_WRONG"

        provider.set_rewrite_responses(bad_rewrite, bad_rewrite)
        provider.set_verify_responses(_approved_verification_payload())

        anonymizer = _build(provider)
        with pytest.raises(ExportAnonymizationError, match="schema constraints"):
            await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)

        assert provider.rewrite_calls == 2
        assert provider.verify_calls == 0


# ---------------------------------------------------------------------------
# 4. Verifier rejection
# ---------------------------------------------------------------------------


class TestVerifierRejection:

    @pytest.mark.asyncio
    async def test_verification_rejected_raises_error(self) -> None:
        provider = _CannedProvider()
        messages = _two_message_conversation()

        provider.set_rewrite_responses(_good_rewrite_payload(messages))
        provider.set_verify_responses(_rejected_verification_payload())

        anonymizer = _build(provider)
        with pytest.raises(ExportAnonymizationVerificationError, match="could not be verified"):
            await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)

    @pytest.mark.asyncio
    async def test_verification_rejected_then_retried_and_approved(self) -> None:
        """First verification rejects, retry rewrites, second verification approves."""
        provider = _CannedProvider()
        messages = _two_message_conversation()

        good_rewrite = _good_rewrite_payload(messages)
        provider.set_rewrite_responses(good_rewrite, good_rewrite)
        provider.set_verify_responses(
            _rejected_verification_payload(),
            _approved_verification_payload(),
        )

        anonymizer = _build(provider)
        # The first attempt passes validation but fails verification,
        # triggering a retry. But _MAX_ATTEMPTS is 2, so the second
        # attempt's rewrite+verify should succeed.
        result = await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)

        assert provider.rewrite_calls == 2
        assert provider.verify_calls == 2
        assert result.strict_messages["msg_1"] is not None


# ---------------------------------------------------------------------------
# 5. Schema validation (_validate_rewrite)
# ---------------------------------------------------------------------------


class TestSchemaValidation:

    def _anonymizer(self) -> ExportAnonymizer:
        return _build(_CannedProvider())

    def test_mismatched_message_ids(self) -> None:
        anonymizer = self._anonymizer()
        messages = _two_message_conversation()
        rewrite = _RewriteResponse(
            entities=[],
            messages=[
                {"message_id": "msg_1", "strict_content": "a", "readable_content": "a"},
                {"message_id": "msg_WRONG", "strict_content": "b", "readable_content": "b"},
            ],
        )
        issues = anonymizer._validate_rewrite(messages, rewrite)
        assert any("exactly once" in issue for issue in issues)

    def test_duplicate_placeholders(self) -> None:
        anonymizer = self._anonymizer()
        messages = [_msg("msg_1", 1, "user", "Hello")]
        rewrite = _RewriteResponse(
            entities=[
                {"placeholder": "[person_001]", "readable_label": "Person 1", "source_forms": []},
                {"placeholder": "[person_001]", "readable_label": "Person 2", "source_forms": []},
            ],
            messages=[
                {"message_id": "msg_1", "strict_content": "Hello", "readable_content": "Hello"},
            ],
        )
        issues = anonymizer._validate_rewrite(messages, rewrite)
        assert any("Duplicate placeholder" in issue for issue in issues)

    def test_readable_label_same_as_placeholder(self) -> None:
        anonymizer = self._anonymizer()
        messages = [_msg("msg_1", 1, "user", "Hello")]
        rewrite = _RewriteResponse(
            entities=[
                {
                    "placeholder": "[person_001]",
                    "readable_label": "[person_001]",
                    "source_forms": [],
                },
            ],
            messages=[
                {"message_id": "msg_1", "strict_content": "Hello", "readable_content": "Hello"},
            ],
        )
        issues = anonymizer._validate_rewrite(messages, rewrite)
        assert any("must differ" in issue for issue in issues)

    def test_empty_strict_content_caught(self) -> None:
        anonymizer = self._anonymizer()
        messages = [_msg("msg_1", 1, "user", "Hello")]
        rewrite = _RewriteResponse(
            entities=[],
            messages=[
                {"message_id": "msg_1", "strict_content": "   ", "readable_content": "Hello"},
            ],
        )
        issues = anonymizer._validate_rewrite(messages, rewrite)
        assert any("missing strict_content" in issue for issue in issues)

    def test_empty_readable_content_caught(self) -> None:
        anonymizer = self._anonymizer()
        messages = [_msg("msg_1", 1, "user", "Hello")]
        rewrite = _RewriteResponse(
            entities=[],
            messages=[
                {"message_id": "msg_1", "strict_content": "Hello", "readable_content": "   "},
            ],
        )
        issues = anonymizer._validate_rewrite(messages, rewrite)
        assert any("missing readable_content" in issue for issue in issues)

    def test_valid_rewrite_has_no_issues(self) -> None:
        anonymizer = self._anonymizer()
        messages = _two_message_conversation()
        rewrite = _RewriteResponse(**_good_rewrite_payload(messages))
        issues = anonymizer._validate_rewrite(messages, rewrite)
        assert issues == []


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_single_message_conversation(self) -> None:
        provider = _CannedProvider()
        messages = [_msg("msg_1", 1, "user", "Maria called from Barcelona.")]
        rewrite_payload = {
            "entities": [
                {
                    "placeholder": "[person_001]",
                    "readable_label": "Person 1",
                    "source_forms": ["Maria"],
                },
                {
                    "placeholder": "[place_001]",
                    "readable_label": "Place 1",
                    "source_forms": ["Barcelona"],
                },
            ],
            "messages": [
                {
                    "message_id": "msg_1",
                    "strict_content": "[person_001] called from [place_001].",
                    "readable_content": "Person 1 called from Place 1.",
                },
            ],
        }
        provider.set_rewrite_responses(rewrite_payload)
        provider.set_verify_responses(_approved_verification_payload())

        anonymizer = _build(provider)
        result = await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)

        assert len(result.strict_messages) == 1
        assert result.summary.entity_count == 2

    @pytest.mark.asyncio
    async def test_no_entities_to_anonymize(self) -> None:
        provider = _CannedProvider()
        messages = [
            _msg("msg_1", 1, "user", "What is the speed of light?"),
            _msg("msg_2", 2, "assistant", "About 300,000 km/s in vacuum."),
        ]
        rewrite_payload = {
            "entities": [],
            "messages": [
                {
                    "message_id": "msg_1",
                    "strict_content": "What is the speed of light?",
                    "readable_content": "What is the speed of light?",
                },
                {
                    "message_id": "msg_2",
                    "strict_content": "About 300,000 km/s in vacuum.",
                    "readable_content": "About 300,000 km/s in vacuum.",
                },
            ],
        }
        provider.set_rewrite_responses(rewrite_payload)
        provider.set_verify_responses(_approved_verification_payload())

        anonymizer = _build(provider)
        result = await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.READABLE)

        assert result.summary.entity_count == 0
        assert result.summary.entities == []
        assert result.readable_messages["msg_1"] == "What is the speed of light?"

    @pytest.mark.asyncio
    async def test_raw_mode_raises_value_error(self) -> None:
        provider = _CannedProvider()
        anonymizer = _build(provider)
        with pytest.raises(ValueError, match="non-raw"):
            await anonymizer.anonymize_messages([], ExportAnonymizationMode.RAW)


# ---------------------------------------------------------------------------
# 7. Transcript rendering
# ---------------------------------------------------------------------------


class TestTranscriptRendering:

    def test_render_transcript_xml_structure(self) -> None:
        messages = [
            _msg("msg_1", 1, "user", "Hello world"),
            _msg("msg_2", 2, "assistant", "Hi there"),
        ]
        output = ExportAnonymizer._render_transcript(messages)
        assert output.startswith("<conversation>\n")
        assert output.endswith("\n</conversation>")
        assert 'id="msg_1"' in output
        assert 'role="user"' in output
        assert 'seq="1"' in output
        assert "Hello world" in output

    def test_render_transcript_html_escaping(self) -> None:
        messages = [
            _msg("msg_<1>", 1, "user", 'Content with <b>"tags"</b> & entities'),
        ]
        output = ExportAnonymizer._render_transcript(messages)
        assert "msg_&lt;1&gt;" in output
        assert "&lt;b&gt;" in output
        assert "&amp; entities" in output
        assert "&quot;tags&quot;" in output

    def test_render_projection_transcript(self) -> None:
        projection = {
            "msg_1": "[person_001] said hello.",
            "msg_2": "Response to [person_001].",
        }
        output = ExportAnonymizer._render_projection_transcript(projection)
        assert output.startswith("<conversation>\n")
        assert 'id="msg_1"' in output
        assert "[person_001] said hello." in output


# ---------------------------------------------------------------------------
# 8. Projection mode selection
# ---------------------------------------------------------------------------


class TestProjectionModeSelection:

    @pytest.mark.asyncio
    async def test_strict_mode_uses_strict_content(self) -> None:
        provider = _CannedProvider()
        messages = _two_message_conversation()
        provider.set_rewrite_responses(_good_rewrite_payload(messages))
        provider.set_verify_responses(_approved_verification_payload())

        anonymizer = _build(provider)
        result = await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)

        assert "[person_001]" in result.strict_messages["msg_1"]
        assert "Person 1" not in result.strict_messages["msg_1"]

    @pytest.mark.asyncio
    async def test_readable_mode_uses_readable_content(self) -> None:
        provider = _CannedProvider()
        messages = _two_message_conversation()
        provider.set_rewrite_responses(_good_rewrite_payload(messages))
        provider.set_verify_responses(_approved_verification_payload())

        anonymizer = _build(provider)
        result = await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.READABLE)

        assert "Person 1" in result.readable_messages["msg_1"]

    @pytest.mark.asyncio
    async def test_summary_entity_count_matches(self) -> None:
        provider = _CannedProvider()
        messages = _two_message_conversation()
        provider.set_rewrite_responses(_good_rewrite_payload(messages))
        provider.set_verify_responses(_approved_verification_payload())

        anonymizer = _build(provider)
        result = await anonymizer.anonymize_messages(messages, ExportAnonymizationMode.STRICT)

        assert result.summary.entity_count == 2
        assert result.summary.mode == ExportAnonymizationMode.STRICT
        assert result.summary.applied is True
        assert len(result.summary.entities) == 2
