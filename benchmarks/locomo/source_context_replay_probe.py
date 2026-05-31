"""Fixed-context source-memory replay probe for LoCoMo failures.

This diagnostic builds answer contexts from already-retained DB evidence and
existing report custody. It does not change retrieval behavior and does not run
fresh LoCoMo ingestion.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Any, Iterable

from benchmarks.artifact_hash import sha256_file_if_exists
from benchmarks.json_artifacts import write_json_atomic
from benchmarks.llm_metrics import LLMCallRecorder, install_llm_call_recorder
from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.evidence_hydration_probe import (
    _build_db_index,
    _conversation_db_path,
    _dict,
    _list,
    _resolve_path,
    _trace_message_ids_by_turn,
    _trace_user_id,
)
from benchmarks.locomo.failure_ledger import ReportSpec, parse_report_spec
from benchmarks.locomo.source_memory_oracle_probe import (
    build_source_memory_oracle_probe,
)
from benchmarks.scorer import LLMJudgeScorer
from atagia.core.config import Settings
from atagia.core.llm_output_limits import ATAGIA_BENCH_ANSWER_MAX_OUTPUT_TOKENS
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage
from atagia.services.model_resolution import provider_qualified_model
from atagia.services.providers import build_llm_client


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "benchmarks" / "data" / "locomo10.json"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "docs" / "tmp"
_DEFAULT_ANSWER_MODEL = "openrouter/openai/gpt-chat-latest"
_DEFAULT_JUDGE_MODEL = "openrouter/anthropic/claude-opus-4.7"
_DEFAULT_VARIANTS = (
    "selected_current",
    "selected_plus_source_non_summary",
    "source_memory_all",
    "gold_source_window",
)
_CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "unscored",
}


def build_source_context_cases(
    report_specs: Iterable[ReportSpec | str | Path],
    *,
    data_path: str | Path = _DEFAULT_DATA_PATH,
    categories: set[str] | None = None,
    max_cases: int = 12,
    max_cases_per_bucket: int = 2,
) -> dict[str, Any]:
    """Build fixed-context replay cases from retained DB artifacts."""
    specs = [_normalize_report_spec(spec) for spec in report_specs]
    category_filter = categories or {"single-hop", "temporal"}
    dataset = LoCoMoAdapter(data_path).load()
    conversations_by_id = {
        conversation.conversation_id: conversation
        for conversation in dataset.conversations
    }
    report_index = _load_report_index(specs)
    oracle = build_source_memory_oracle_probe(
        specs,
        data_path=data_path,
        include_passing=False,
    )
    selected_items = _select_oracle_items(
        _list(oracle.get("items")),
        categories=category_filter,
        max_cases=max_cases,
        max_cases_per_bucket=max_cases_per_bucket,
    )

    db_cache: dict[tuple[str | None, str], Any] = {}
    cases: list[dict[str, Any]] = []
    for item in selected_items:
        source_report = str(item.get("source_report") or "")
        conversation_id = str(item.get("conversation_id") or "")
        question_id = str(item.get("question_id") or "")
        result = report_index["results"].get((source_report, conversation_id, question_id))
        conversation_report = report_index["conversations"].get((source_report, conversation_id))
        if result is None or conversation_report is None:
            continue
        dataset_conversation = conversations_by_id.get(conversation_id)
        db_path = _conversation_db_path(conversation_report, report_index["reports"][source_report])
        cache_key = (str(db_path) if db_path is not None else None, conversation_id)
        if cache_key not in db_cache:
            db_cache[cache_key] = _build_db_index(
                db_path=db_path,
                conversation_id=conversation_id,
                dataset_turns=list(getattr(dataset_conversation, "turns", [])),
            )
        db_index = db_cache[cache_key]
        if not getattr(db_index, "db_available", False):
            continue
        case = _build_case(
            item=item,
            result=result,
            db_path=db_index.db_path,
        )
        if case["contexts"]:
            cases.append(case)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "source_reports": [
            {
                "path": str(spec.path),
                "sha256": sha256_file_if_exists(spec.path),
                "conversation_filter": sorted(spec.conversation_ids)
                if spec.conversation_ids is not None
                else None,
            }
            for spec in specs
        ],
        "selection": {
            "categories": sorted(category_filter),
            "max_cases": max_cases,
            "max_cases_per_bucket": max_cases_per_bucket,
        },
        "summary": _case_summary(cases),
        "cases": cases,
    }


async def run_source_context_replay(
    cases_report: dict[str, Any],
    *,
    answer_model: str = _DEFAULT_ANSWER_MODEL,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    answer_max_output_tokens: int = ATAGIA_BENCH_ANSWER_MAX_OUTPUT_TOKENS,
    variants: Iterable[str] = _DEFAULT_VARIANTS,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Answer and judge fixed-context case variants with real models."""
    settings = Settings.from_env()
    if api_key:
        settings = replace(settings, openrouter_api_key=api_key)
    client = build_llm_client(settings)
    recorder = LLMCallRecorder()
    install_llm_call_recorder(client, recorder)
    judge = LLMJudgeScorer(client, judge_model)
    wanted_variants = tuple(variants)
    results: list[dict[str, Any]] = []

    for case in _list(cases_report.get("cases")):
        case_dict = _dict(case)
        contexts = _dict(case_dict.get("contexts"))
        for variant in wanted_variants:
            context = str(contexts.get(variant) or "").strip()
            if not context:
                continue
            started_at = perf_counter()
            answer = ""
            answer_error = None
            response_model = None
            usage: dict[str, Any] = {}
            try:
                response = await client.complete(
                    LLMCompletionRequest(
                        model=answer_model,
                        messages=_answer_messages(
                            question_text=str(case_dict.get("question_text") or ""),
                            context=context,
                        ),
                        max_output_tokens=answer_max_output_tokens,
                        metadata={
                            "purpose": "locomo_source_context_answer_probe",
                            "question": str(case_dict.get("question_text") or ""),
                            "context_variant": variant,
                        },
                    )
                )
                answer = response.output_text
                response_model = response.model
                usage = response.usage
            except Exception as exc:  # pragma: no cover - live API failure path.
                answer_error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (perf_counter() - started_at) * 1000.0

            score_result = None
            if answer:
                try:
                    score_result = await judge.score(
                        question=str(case_dict.get("question_text") or ""),
                        prediction=answer,
                        ground_truth=str(case_dict.get("ground_truth") or ""),
                    )
                except Exception as exc:  # pragma: no cover - live API failure path.
                    score_result = {
                        "score": 0,
                        "reasoning": f"{type(exc).__name__}: {exc}",
                        "judge_model": judge_model,
                    }

            results.append(
                {
                    "case_id": case_dict.get("case_id"),
                    "conversation_id": case_dict.get("conversation_id"),
                    "question_id": case_dict.get("question_id"),
                    "category_name": case_dict.get("category_name"),
                    "probe_bucket": case_dict.get("probe_bucket"),
                    "first_loss_stage": case_dict.get("first_loss_stage"),
                    "context_variant": variant,
                    "context_chars": len(context),
                    "answer": answer,
                    "answer_error": answer_error,
                    "answer_model": answer_model,
                    "response_model": response_model,
                    "answer_usage": usage,
                    "elapsed_ms": elapsed_ms,
                    "score_result": (
                        score_result.model_dump()
                        if hasattr(score_result, "model_dump")
                        else score_result
                    ),
                }
            )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "answer_model": answer_model,
        "judge_model": judge_model,
        "answer_max_output_tokens": answer_max_output_tokens,
        "source_case_summary": _dict(cases_report.get("summary")),
        "summary": _result_summary(results),
        "llm_call_summary": recorder.summary(),
        "cases": _list(cases_report.get("cases")),
        "results": results,
    }


def save_probe(report: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a source-context probe report."""
    return write_json_atomic(Path(output_path).expanduser(), report)


def format_case_summary(report: dict[str, Any]) -> str:
    """Return compact build-only summary text."""
    summary = _dict(report.get("summary"))
    return (
        "LoCoMo source-context cases: "
        f"cases={_int(summary.get('case_count'))} "
        f"buckets={_format_counts(_dict(summary.get('probe_bucket_counts')))} "
        f"categories={_format_counts(_dict(summary.get('category_counts')))} "
        f"variants={_format_counts(_dict(summary.get('context_variant_counts')))}"
    )


def format_replay_summary(report: dict[str, Any]) -> str:
    """Return compact replay summary text."""
    summary = _dict(report.get("summary"))
    return (
        "LoCoMo source-context replay: "
        f"generations={_int(summary.get('generation_count'))} "
        f"correct={_int(summary.get('correct_count'))} "
        f"accuracy={_float(summary.get('accuracy')):.3f} "
        f"by_variant={_format_variant_accuracy(_dict(summary.get('by_variant')))}"
    )


def _normalize_report_spec(spec: ReportSpec | str | Path) -> ReportSpec:
    if isinstance(spec, ReportSpec):
        return spec
    if isinstance(spec, Path):
        return ReportSpec(path=spec.expanduser())
    return parse_report_spec(spec)


def _load_report_index(specs: list[ReportSpec]) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    conversations: dict[tuple[str, str], dict[str, Any]] = {}
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for spec in specs:
        report = _read_json(spec.path)
        report_path = str(spec.path)
        reports[report_path] = report
        for conversation in _list(report.get("conversations")):
            conversation_dict = _dict(conversation)
            conversation_id = str(conversation_dict.get("conversation_id") or "")
            if spec.conversation_ids is not None and conversation_id not in spec.conversation_ids:
                continue
            conversations[(report_path, conversation_id)] = conversation_dict
            for result in _list(conversation_dict.get("results")):
                result_dict = _dict(result)
                question = _dict(result_dict.get("question"))
                question_id = str(question.get("question_id") or "")
                results[(report_path, conversation_id, question_id)] = result_dict
    return {
        "reports": reports,
        "conversations": conversations,
        "results": results,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in report: {path}")
    return value


def _select_oracle_items(
    items: list[Any],
    *,
    categories: set[str],
    max_cases: int,
    max_cases_per_bucket: int,
) -> list[dict[str, Any]]:
    bucket_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    candidates = sorted(
        (_dict(item) for item in items),
        key=lambda item: (
            _bucket_priority(_probe_bucket(item)),
            str(item.get("conversation_id") or ""),
            _question_number(str(item.get("question_id") or "")),
        ),
    )
    for item in candidates:
        if len(selected) >= max_cases:
            break
        category_name = str(item.get("category_name") or "")
        if category_name not in categories:
            continue
        bucket = _probe_bucket(item)
        if bucket_counts[bucket] >= max_cases_per_bucket:
            continue
        item = dict(item)
        item["probe_bucket"] = bucket
        selected.append(item)
        bucket_counts[bucket] += 1
    return sorted(
        selected,
        key=lambda item: (
            str(item.get("conversation_id") or ""),
            _question_number(str(item.get("question_id") or "")),
        ),
    )


def _probe_bucket(item: dict[str, Any]) -> str:
    category = str(item.get("category_name") or "unknown")
    labels = set(str(label) for label in _list(item.get("labels")))
    first_loss_stage = str(item.get("first_loss_stage") or "unknown")
    if "non_summary_available_but_not_raw" in labels:
        return f"{category}:non_summary_available_not_raw"
    if "non_summary_raw_but_not_selected" in labels:
        return f"{category}:non_summary_raw_not_selected"
    if "selected_source_memory_summary_only" in labels:
        return f"{category}:selected_summary_only"
    return f"{category}:{first_loss_stage}"


def _bucket_priority(bucket: str) -> int:
    if bucket.endswith(":non_summary_available_not_raw"):
        return 0
    if bucket.endswith(":non_summary_raw_not_selected"):
        return 1
    if bucket.endswith(":selected_summary_only"):
        return 2
    if bucket.endswith(":critical_selected_all"):
        return 3
    return 4


def _build_case(
    *,
    item: dict[str, Any],
    result: dict[str, Any],
    db_path: Path | None,
) -> dict[str, Any]:
    question = _dict(result.get("question"))
    trace = _dict(result.get("trace"))
    evidence_turn_ids = [str(turn_id) for turn_id in _list(question.get("evidence_turn_ids"))]
    user_id = _trace_user_id(trace)
    selected_memory_ids = [str(memory_id) for memory_id in _list(trace.get("selected_memory_ids"))]
    selected_rows = _memory_rows_by_ids(db_path, user_id, selected_memory_ids)
    trace_message_ids_by_turn = _trace_message_ids_by_turn(trace, evidence_turn_ids)
    source_message_ids = [
        str(message_id)
        for message_id in trace_message_ids_by_turn.values()
        if message_id
    ]
    source_rows = _memory_rows_for_source_messages(db_path, user_id, source_message_ids)
    source_non_summary_rows = [
        row for row in source_rows if not _is_summary_like(row)
    ]
    source_summary_rows = [
        row for row in source_rows if _is_summary_like(row)
    ]
    selected_ids = {str(row.get("id") or "") for row in selected_rows}
    selected_plus_non_summary = selected_rows + [
        row
        for row in source_non_summary_rows
        if str(row.get("id") or "") not in selected_ids
    ]
    source_messages = _messages_by_ids(db_path, source_message_ids)
    contexts = {
        "selected_current": _format_memory_context(selected_rows),
        "selected_plus_source_non_summary": _format_memory_context(
            selected_plus_non_summary
        ),
        "source_memory_all": _format_memory_context(source_rows),
        "source_memory_non_summary": _format_memory_context(source_non_summary_rows),
        "source_memory_summary_only": _format_memory_context(source_summary_rows),
        "gold_source_window": _format_message_context(
            source_messages,
            trace_message_ids_by_turn,
        ),
    }
    contexts = {
        name: value
        for name, value in contexts.items()
        if value.strip()
    }
    return {
        "case_id": (
            f"{item.get('conversation_id')}:{item.get('question_id')}:"
            f"{item.get('probe_bucket')}"
        ),
        "source_report": item.get("source_report"),
        "conversation_id": item.get("conversation_id"),
        "question_id": item.get("question_id"),
        "category": question.get("category"),
        "category_name": item.get("category_name"),
        "probe_bucket": item.get("probe_bucket"),
        "first_loss_stage": item.get("first_loss_stage"),
        "critical_custody_state": item.get("critical_custody_state"),
        "oracle_stage": item.get("oracle_stage"),
        "labels": _list(item.get("labels")),
        "question_text": str(question.get("question_text") or ""),
        "ground_truth": str(question.get("ground_truth") or ""),
        "baseline_prediction": str(result.get("prediction") or ""),
        "baseline_score": _int(_dict(result.get("score_result")).get("score")),
        "baseline_judge_reasoning": str(
            _dict(result.get("score_result")).get("reasoning") or ""
        ),
        "evidence_turn_ids": evidence_turn_ids,
        "source_message_ids": source_message_ids,
        "selected_memory_ids": selected_memory_ids,
        "source_memory_counts": {
            "selected": len(selected_rows),
            "source_all": len(source_rows),
            "source_non_summary": len(source_non_summary_rows),
            "source_summary_like": len(source_summary_rows),
        },
        "contexts": contexts,
        "context_chars": {
            name: len(value)
            for name, value in contexts.items()
        },
    }


def _memory_rows_by_ids(
    db_path: Path | None,
    user_id: str,
    memory_ids: list[str],
) -> list[dict[str, Any]]:
    if db_path is None or not memory_ids:
        return []
    placeholders = ",".join("?" for _ in memory_ids)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = [
            _decode_memory_row(dict(row))
            for row in connection.execute(
                f"""
                SELECT id, object_type, source_kind, status, scope,
                       privacy_level, canonical_text, payload_json
                FROM memory_objects
                WHERE user_id = ?
                  AND id IN ({placeholders})
                ORDER BY CASE id
                  {" ".join(f"WHEN ? THEN {index}" for index, _ in enumerate(memory_ids))}
                  ELSE {len(memory_ids)}
                END
                """,
                (user_id, *memory_ids, *memory_ids),
            )
        ]
    finally:
        connection.close()
    rows_by_id = {str(row.get("id") or ""): row for row in rows}
    return [rows_by_id[memory_id] for memory_id in memory_ids if memory_id in rows_by_id]


def _memory_rows_for_source_messages(
    db_path: Path | None,
    user_id: str,
    source_message_ids: list[str],
) -> list[dict[str, Any]]:
    if db_path is None or not source_message_ids:
        return []
    rows_by_id: dict[str, dict[str, Any]] = {}
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        for source_message_id in source_message_ids:
            rows = [
                _decode_memory_row(dict(row))
                for row in connection.execute(
                    """
                    SELECT DISTINCT mo.id, mo.object_type, mo.source_kind,
                                    mo.status, mo.scope, mo.privacy_level,
                                    mo.canonical_text, mo.payload_json
                    FROM memory_objects AS mo
                    JOIN json_each(mo.payload_json, '$.source_message_ids') AS source_ids
                    WHERE mo.user_id = ?
                      AND CAST(source_ids.value AS TEXT) = ?
                    ORDER BY mo.id
                    """,
                    (user_id, source_message_id),
                )
            ]
            for row in rows:
                memory_id = str(row.get("id") or "")
                if memory_id and memory_id not in rows_by_id:
                    rows_by_id[memory_id] = row
    finally:
        connection.close()
    return list(rows_by_id.values())


def _messages_by_ids(
    db_path: Path | None,
    message_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if db_path is None or not message_ids:
        return {}
    placeholders = ",".join("?" for _ in message_ids)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                f"""
                SELECT id, conversation_id, seq, text, occurred_at, content_kind
                FROM messages
                WHERE id IN ({placeholders})
                ORDER BY seq ASC
                """,
                tuple(message_ids),
            )
        ]
    finally:
        connection.close()
    return {str(row.get("id") or ""): row for row in rows}


def _decode_memory_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload_json")
    if isinstance(payload, str):
        try:
            row["payload_json"] = json.loads(payload)
        except json.JSONDecodeError:
            row["payload_json"] = {}
    elif not isinstance(payload, dict):
        row["payload_json"] = {}
    return row


def _is_summary_like(row: dict[str, Any]) -> bool:
    return (
        str(row.get("object_type") or "") == "summary_view"
        or str(row.get("source_kind") or "") == "summarized"
    )


def _format_memory_context(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get("canonical_text") or "").strip()
        if not text:
            continue
        parts.append(
            "\n".join(
                [
                    (
                        f"[{index}] memory_id={row.get('id')} "
                        f"type={row.get('object_type')} "
                        f"source_kind={row.get('source_kind')} "
                        f"status={row.get('status')}"
                    ),
                    text,
                ]
            )
        )
    return "\n\n".join(parts)


def _format_message_context(
    messages_by_id: dict[str, dict[str, Any]],
    trace_message_ids_by_turn: dict[str, str | None],
) -> str:
    parts: list[str] = []
    for turn_id, message_id in trace_message_ids_by_turn.items():
        if not message_id:
            continue
        message = messages_by_id.get(message_id)
        if not message:
            continue
        text = str(message.get("text") or "").strip()
        if not text:
            continue
        parts.append(
            "\n".join(
                [
                    (
                        f"[turn_id={turn_id} message_id={message_id} "
                        f"seq={message.get('seq')} occurred_at={message.get('occurred_at')}]"
                    ),
                    text,
                ]
            )
        )
    return "\n\n".join(parts)


def _answer_messages(*, question_text: str, context: str) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content=(
                "You are answering a fixed-context memory benchmark question. "
                "Use only the provided evidence. Answer directly and include all "
                "facts needed by the question. If the evidence is insufficient, say so."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"Evidence:\n{context}\n\n"
                f"Question: {question_text}\n"
                "Answer:"
            ),
        ),
    ]


def _case_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    variant_counts: Counter[str] = Counter()
    for case in cases:
        variant_counts.update(_dict(case.get("contexts")).keys())
    return {
        "case_count": len(cases),
        "category_counts": _counter_dict(case.get("category_name") for case in cases),
        "probe_bucket_counts": _counter_dict(case.get("probe_bucket") for case in cases),
        "first_loss_stage_counts": _counter_dict(case.get("first_loss_stage") for case in cases),
        "context_variant_counts": dict(sorted(variant_counts.items())),
    }


def _result_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    correct_count = sum(
        _int(_dict(result.get("score_result")).get("score"))
        for result in results
    )
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(result.get("context_variant") or "") for result in results}):
        group = [
            result
            for result in results
            if str(result.get("context_variant") or "") == variant
        ]
        correct = sum(_int(_dict(result.get("score_result")).get("score")) for result in group)
        by_variant[variant] = {
            "count": len(group),
            "correct": correct,
            "accuracy": correct / len(group) if group else None,
        }
    return {
        "generation_count": len(results),
        "correct_count": correct_count,
        "accuracy": correct_count / len(results) if results else None,
        "by_variant": by_variant,
        "by_category_name": _result_summary_by(results, "category_name"),
        "by_probe_bucket": _result_summary_by(results, "probe_bucket"),
    }


def _result_summary_by(results: list[dict[str, Any]], field_name: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result.get(field_name) or "unknown"), []).append(result)
    summary: dict[str, dict[str, Any]] = {}
    for key, group in sorted(grouped.items()):
        correct = sum(_int(_dict(result.get("score_result")).get("score")) for result in group)
        summary[key] = {
            "count": len(group),
            "correct": correct,
            "accuracy": correct / len(group) if group else None,
        }
    return summary


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        counter[str(value or "unknown")] += 1
    return dict(sorted(counter.items()))


def _format_counts(value: dict[str, Any]) -> str:
    parts = [
        f"{key}={_int(amount)}"
        for key, amount in sorted(value.items())
        if _int(amount)
    ]
    return " ".join(parts) if parts else "none"


def _format_variant_accuracy(value: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, item in sorted(value.items()):
        item_dict = _dict(item)
        count = _int(item_dict.get("count"))
        correct = _int(item_dict.get("correct"))
        accuracy = _float(item_dict.get("accuracy"))
        parts.append(f"{key}={correct}/{count}({accuracy:.2f})")
    return " ".join(parts) if parts else "none"


def _question_number(question_id: str) -> int:
    raw = question_id.rsplit(":q", 1)[-1]
    try:
        return int(raw)
    except ValueError:
        return 0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help=(
            "LoCoMo report JSON path. Optionally suffix with ::conv-id,conv-id "
            "to keep only those conversations from that report."
        ),
    )
    parser.add_argument(
        "--data-path",
        default=str(_DEFAULT_DATA_PATH),
        help="Path to locomo10.json.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=["single-hop", "temporal"],
        help="Category name to include. Repeatable.",
    )
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--max-cases-per-bucket", type=int, default=2)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Context variant to run. Repeatable. Defaults to core variants.",
    )
    parser.add_argument("--answer-model", default=_DEFAULT_ANSWER_MODEL)
    parser.add_argument("--judge-model", default=_DEFAULT_JUDGE_MODEL)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--answer-max-output-tokens",
        type=int,
        default=ATAGIA_BENCH_ANSWER_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build fixed-context cases; do not call answer or judge models.",
    )
    parser.add_argument(
        "--output",
        default=str(_default_output_path()),
        help="Output JSON path. Defaults to docs/tmp with a timestamp.",
    )
    args = parser.parse_args()
    if args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.max_cases_per_bucket < 1:
        parser.error("--max-cases-per-bucket must be positive")
    if args.answer_max_output_tokens < 1:
        parser.error("--answer-max-output-tokens must be positive")
    return args


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _DEFAULT_OUTPUT_DIR / f"locomo_source_context_replay_{timestamp}.json"


async def _async_main() -> None:
    args = _parse_args()
    answer_model = provider_qualified_model(args.provider, args.answer_model) or args.answer_model
    judge_model = provider_qualified_model(args.provider, args.judge_model) or args.judge_model
    case_report = build_source_context_cases(
        [parse_report_spec(value) for value in args.report],
        data_path=_resolve_path(Path(args.data_path)),
        categories={str(category) for category in args.category},
        max_cases=args.max_cases,
        max_cases_per_bucket=args.max_cases_per_bucket,
    )
    if args.build_only:
        output = save_probe(case_report, args.output)
        print(format_case_summary(case_report))
        print(f"Wrote {output}")
        return

    replay_report = await run_source_context_replay(
        case_report,
        answer_model=answer_model,
        judge_model=judge_model,
        answer_max_output_tokens=args.answer_max_output_tokens,
        variants=args.variant or _DEFAULT_VARIANTS,
        api_key=args.api_key,
    )
    output = save_probe(replay_report, args.output)
    print(format_replay_summary(replay_report))
    print(f"Wrote {output}")


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
