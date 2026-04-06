"""Tests for transcript-window helpers used by chat orchestration."""

from __future__ import annotations

from pathlib import Path

from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.services.chat_support import (
    ChunkSummary,
    RawMessage,
    build_system_prompt,
    build_transcript_window,
    build_transcript_window_trace,
    estimate_tokens,
    format_chunk_summary,
    missing_uncovered_tail_start_seq,
    render_transcript_window,
)

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _message(seq: int, *, role: str | None = None, text: str) -> dict[str, object]:
    return {
        "seq": seq,
        "role": role or ("user" if seq % 2 else "assistant"),
        "text": text,
    }


def _chunk(summary_id: str, start_seq: int, end_seq: int, summary_text: str) -> dict[str, object]:
    return {
        "id": summary_id,
        "source_message_start_seq": start_seq,
        "source_message_end_seq": end_seq,
        "summary_text": summary_text,
    }


def _raw_seqs(entries: list[RawMessage | ChunkSummary]) -> list[int]:
    return [entry.seq for entry in entries if isinstance(entry, RawMessage)]


def _chunk_ids(entries: list[RawMessage | ChunkSummary]) -> list[str]:
    return [entry.chunk_id for entry in entries if isinstance(entry, ChunkSummary)]


def test_build_transcript_window_without_chunks_keeps_messages_raw() -> None:
    messages = [
        _message(1, text="alpha" * 10),
        _message(2, text="beta" * 10),
        _message(3, text="gamma" * 10),
        _message(4, text="delta" * 10),
    ]

    entries = build_transcript_window(messages, [], budget_tokens=200)
    rendered = render_transcript_window(entries)

    assert _raw_seqs(entries) == [1, 2, 3, 4]
    assert _chunk_ids(entries) == []
    assert [message["text"] for message in rendered] == [message["text"] for message in messages]


def test_build_transcript_window_uses_summary_for_complete_chunk_when_raw_does_not_fit() -> None:
    messages = [
        *[_message(seq, text=f"old-{seq}-" + ("x" * 1200)) for seq in range(1, 7)],
        *[_message(seq, text=f"recent-{seq}") for seq in range(7, 11)],
    ]
    chunk = _chunk("sum_old", 1, 6, "Earlier retry debugging decisions.")
    recency_tokens = sum(estimate_tokens(str(message["text"])) for message in messages[6:])
    budget = recency_tokens + estimate_tokens(format_chunk_summary(chunk))

    entries = build_transcript_window(messages, [chunk], budget_tokens=budget)
    rendered = render_transcript_window(entries)

    assert _chunk_ids(entries) == ["sum_old"]
    assert _raw_seqs(entries) == [7, 8, 9, 10]
    assert rendered[0]["role"] == "assistant"
    assert rendered[0]["text"].startswith("[Conversation summary | historical context only | turns 1-6]")


def test_build_transcript_window_handles_interior_chunk_gaps_as_uncovered_raw() -> None:
    messages = [_message(seq, text=f"msg-{seq}" * 10) for seq in range(1, 13)]
    chunks = [
        _chunk("sum_a", 1, 4, "Earlier design decisions."),
        _chunk("sum_b", 9, 12, "Recent compacted exchange."),
    ]
    raw_gap_and_floor_tokens = sum(
        estimate_tokens(str(message["text"]))
        for message in messages[4:]
    )
    budget = raw_gap_and_floor_tokens + estimate_tokens(format_chunk_summary(chunks[0]))

    entries = build_transcript_window(messages, chunks, budget_tokens=budget)

    assert _chunk_ids(entries) == ["sum_a"]
    assert _raw_seqs(entries) == [5, 6, 7, 8, 9, 10, 11, 12]


def test_build_transcript_window_uses_summary_for_incomplete_chunk_at_fetch_boundary() -> None:
    messages = [_message(seq, text=f"window-{seq}" * 8) for seq in range(501, 507)]
    chunk = _chunk("sum_boundary", 496, 502, "Chunk crossing the fetch boundary.")

    entries = build_transcript_window(messages, [chunk], budget_tokens=300)

    assert _chunk_ids(entries) == ["sum_boundary"]
    assert _raw_seqs(entries) == [503, 504, 505, 506]


def test_build_transcript_window_preserves_recency_floor_in_fully_compacted_conversation() -> None:
    messages = [_message(seq, text=f"turn-{seq}" * 8) for seq in range(1, 11)]
    chunk = _chunk("sum_all", 1, 10, "Fully compacted conversation summary.")
    recency_floor_tokens = sum(
        estimate_tokens(str(message["text"]))
        for message in messages[-4:]
    )
    budget = recency_floor_tokens + estimate_tokens(format_chunk_summary(chunk))

    entries = build_transcript_window(messages, [chunk], budget_tokens=budget)

    assert _chunk_ids(entries) == ["sum_all"]
    assert _raw_seqs(entries) == [7, 8, 9, 10]


def test_build_transcript_window_skips_oversized_uncovered_messages_without_chunk_fallback() -> None:
    messages = [
        _message(1, text="small-a" * 8),
        _message(2, text="huge-b" * 400),
        _message(3, text="small-c" * 8),
        _message(4, text="recent-4"),
        _message(5, text="recent-5"),
        _message(6, text="recent-6"),
        _message(7, text="recent-7"),
    ]
    floor_tokens = sum(estimate_tokens(str(message["text"])) for message in messages[-4:])
    budget = floor_tokens + estimate_tokens(str(messages[0]["text"])) + estimate_tokens(str(messages[2]["text"]))

    entries = build_transcript_window(messages, [], budget_tokens=budget)

    assert _raw_seqs(entries) == [1, 3, 4, 5, 6, 7]
    assert 2 not in _raw_seqs(entries)


def test_build_transcript_window_skips_newest_oversized_uncovered_message() -> None:
    messages = [
        _message(1, text="small-a" * 8),
        _message(2, text="huge-b" * 400),
        _message(3, text="recent-3"),
        _message(4, text="recent-4"),
        _message(5, text="recent-5"),
        _message(6, text="recent-6"),
    ]
    floor_tokens = sum(estimate_tokens(str(message["text"])) for message in messages[-4:])
    budget = floor_tokens + estimate_tokens(str(messages[0]["text"]))

    entries = build_transcript_window(messages, [], budget_tokens=budget)

    assert _raw_seqs(entries) == [1, 3, 4, 5, 6]
    assert 2 not in _raw_seqs(entries)


def test_build_transcript_window_trace_and_system_prompt_include_summary_boundary() -> None:
    messages = [
        *[_message(seq, text=f"older-{seq}" * 50) for seq in range(1, 7)],
        *[_message(seq, text=f"recent-{seq}") for seq in range(7, 11)],
    ]
    chunk = _chunk("sum_trace", 1, 6, "Summary text for traceability.")

    entries = build_transcript_window(messages, [chunk], budget_tokens=400)
    trace = build_transcript_window_trace(entries, 400)
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["general_qa"]
    resolved_policy = PolicyResolver().resolve(manifest, None, None)
    prompt = build_system_prompt(
        "general_qa",
        resolved_policy,
        "",
        "",
        "",
        "",
    )

    assert trace["chunk_ids"] == ["sum_trace"]
    assert trace["raw_message_seqs"] == [7, 8, 9, 10]
    assert trace["budget_tokens"] == 400
    assert trace["budget_used_tokens"] > 0
    assert "resolve them against that memory's source_window date" in prompt
    assert "last Saturday" in prompt
    assert "Calculate the actual calendar date when possible." in prompt
    assert "include all distinct items found across the retrieved memories" in prompt
    assert "[Conversation summary | historical context only | ...]" in prompt
    assert "[End of summary]" in prompt


def test_missing_uncovered_tail_start_seq_detects_gap_before_recent_window() -> None:
    messages = [_message(seq, text=f"msg-{seq}") for seq in range(21, 31)]
    chunks = [_chunk("sum_1", 1, 10, "Older compacted block.")]

    assert missing_uncovered_tail_start_seq(messages, chunks) == 11


def test_missing_uncovered_tail_start_seq_returns_none_when_recent_window_is_contiguous() -> None:
    messages = [_message(seq, text=f"msg-{seq}") for seq in range(21, 31)]
    chunks = [_chunk("sum_1", 1, 20, "Older compacted block.")]

    assert missing_uncovered_tail_start_seq(messages, chunks) is None
