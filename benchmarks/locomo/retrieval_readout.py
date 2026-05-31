"""Per-question retrieval readout artifacts for LoCoMo runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.base import BenchmarkReport, QuestionResult
from benchmarks.json_artifacts import write_json_atomic


def build_retrieval_readout(
    report: BenchmarkReport,
    *,
    source_report: str,
) -> dict[str, Any]:
    """Build a compact per-question retrieval diagnostic artifact."""
    questions: list[dict[str, Any]] = []
    for conversation in report.conversations:
        for result in conversation.results:
            questions.append(
                _question_readout(
                    result,
                    conversation_id=conversation.conversation_id,
                )
            )
    return {
        "artifact_kind": "locomo_retrieval_readout",
        "source_report": source_report,
        "benchmark_name": report.benchmark_name,
        "report_timestamp": report.timestamp,
        "overall_accuracy": report.overall_accuracy,
        "total_correct": report.total_correct,
        "total_questions": report.total_questions,
        "questions": questions,
    }


def save_retrieval_readout(readout: dict[str, Any], destination: str | Path) -> Path:
    """Persist a retrieval readout artifact."""
    return write_json_atomic(Path(destination).expanduser(), readout)


def _question_readout(
    result: QuestionResult,
    *,
    conversation_id: str,
) -> dict[str, Any]:
    trace = _dict(result.trace)
    selected_ids = _text_list(trace.get("selected_memory_ids"))
    selected_records = _selected_custody_records(trace, selected_ids)
    context = _dict(trace.get("context"))
    answer_evidence_items = [
        item for item in _list(context.get("answer_evidence_items")) if isinstance(item, dict)
    ]
    selected_evidence_memory_ids = _text_list(
        trace.get("selected_evidence_memory_ids")
    )
    answer_evidence_memory_ids = _text_list(
        context.get("answer_evidence_memory_ids")
    )
    answer_evidence_sufficiency = _dict(
        context.get("answer_evidence_sufficiency")
    )
    rendered_answer_evidence_items = [
        item
        for item in answer_evidence_items
        if item.get("selected_for_answer_pack") is True
    ]
    return {
        "conversation_id": conversation_id,
        "question_id": result.question.question_id,
        "question": result.question.question_text,
        "gold_answer": result.question.ground_truth,
        "model_answer": result.prediction,
        "score": result.score_result.score,
        "judge_reasoning": result.score_result.reasoning,
        "diagnosis_bucket": str(trace.get("diagnosis_bucket") or ""),
        "sufficiency_diagnostic": str(trace.get("sufficiency_diagnostic") or ""),
        "evidence_turn_ids": list(result.question.evidence_turn_ids),
        "evidence_message_ids": _text_list(trace.get("evidence_message_ids")),
        "selected_memory_ids": selected_ids,
        "selected_evidence_memory_ids": selected_evidence_memory_ids,
        "answer_evidence_memory_ids": answer_evidence_memory_ids,
        "answer_evidence_items": answer_evidence_items,
        "answer_evidence_sufficiency": answer_evidence_sufficiency,
        "selected_memory_summaries": _list(trace.get("selected_memory_summaries")),
        "selected_counts": _selected_counts(selected_records),
        "top_selected_item": _top_selected_item(selected_records),
        "exact_answer_quote_present": _exact_answer_quote_present(
            result.question.ground_truth,
            answer_evidence_items,
        ),
        "exact_answer_quote_rendered": _exact_answer_quote_present(
            result.question.ground_truth,
            rendered_answer_evidence_items,
        ),
        "context": {
            "items_included": context.get("items_included"),
            "items_dropped": context.get("items_dropped"),
            "budget_tokens": context.get("budget_tokens"),
            "total_tokens_estimate": context.get("total_tokens_estimate"),
            "memory_block_chars": context.get("memory_block_chars"),
        },
    }


def _selected_custody_records(
    trace: dict[str, Any],
    selected_ids: list[str],
) -> list[dict[str, Any]]:
    selected_id_set = set(selected_ids)
    records: list[dict[str, Any]] = []
    for record in _list(trace.get("retrieval_custody")):
        if not isinstance(record, dict):
            continue
        candidate_id = str(record.get("candidate_id") or "")
        if (
            record.get("selected") is True
            or record.get("composer_decision") == "selected"
            or candidate_id in selected_id_set
        ):
            records.append(record)
    return sorted(records, key=_selected_record_key)


def _selected_record_key(record: dict[str, Any]) -> tuple[int, str]:
    rank = _optional_int(record.get("selection_rank"))
    return (rank if rank is not None else 1_000_000, str(record.get("candidate_id") or ""))


def _selected_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_kinds: Counter[str] = Counter()
    source_kinds: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    for record in records:
        candidate_kinds[str(record.get("candidate_kind") or "unknown")] += 1
        source_kinds[str(record.get("source_kind") or "unknown")] += 1
        for channel in _text_list(record.get("channels") or record.get("retrieval_sources")):
            channels[channel] += 1
    return {
        "candidate_kinds": dict(sorted(candidate_kinds.items())),
        "source_kinds": dict(sorted(source_kinds.items())),
        "channels": dict(sorted(channels.items())),
    }


def _top_selected_item(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    record = records[0]
    return {
        "memory_id": str(record.get("candidate_id") or ""),
        "candidate_kind": str(record.get("candidate_kind") or ""),
        "source_kind": str(record.get("source_kind") or ""),
        "channels": _text_list(record.get("channels") or record.get("retrieval_sources")),
        "selection_rank": record.get("selection_rank"),
        "score_rank": record.get("score_rank"),
        "final_score": _dict(record.get("scorer")).get("final_score"),
    }


def _exact_answer_quote_present(
    ground_truth: str,
    answer_evidence_items: list[dict[str, Any]],
) -> bool:
    normalized_answer = _normalize_text(ground_truth)
    if not normalized_answer:
        return False
    for item in answer_evidence_items:
        quote = _normalize_text(str(item.get("supporting_quote") or ""))
        if normalized_answer and normalized_answer in quote:
            return True
    return False


def _normalize_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text_list(value: Any) -> list[str]:
    return [str(item) for item in _list(value)]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
