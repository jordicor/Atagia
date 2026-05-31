"""Tests for text-free initial-context package rollout artifacts."""

from __future__ import annotations

import json

from atagia.services.initial_context_package_validation import (
    build_initial_context_package_prompt_diff_artifact,
    write_initial_context_package_prompt_diff_artifact,
)


def test_prompt_diff_artifact_is_text_free_and_reports_rollout_metrics(tmp_path) -> None:
    private_fact = "Private snapshot fact that must not be serialized"
    prompt_without = "Core prompt\n[Retrieved Memories]\n- Existing evidence"
    prompt_with = (
        "Core prompt\n[Prepared Initial Context]\n"
        f"- {private_fact}\n[Retrieved Memories]\n- Existing evidence"
    )

    artifact = build_initial_context_package_prompt_diff_artifact(
        label="private retained replay",
        prompt_without_package=prompt_without,
        prompt_with_package=prompt_with,
        diagnostics_without_package={
            "enabled": False,
            "rendered": False,
            "raw_prompt": private_fact,
        },
        diagnostics_with_package={
            "enabled": True,
            "rendered": True,
            "read_ms": 1.25,
            "tokens_estimate": 42,
            "selected_profile_items": 1,
            "raw_prompt": private_fact,
            "packages": [
                {
                    "package_kind": "conversation",
                    "status": "hit",
                    "package_key_hash": "icp:v1:test",
                    "source_refs": [{"message_text": private_fact}],
                }
            ],
        },
        llm_calls_without_package=2,
        llm_calls_with_package=2,
        sql_statements_without_package=20,
        sql_statements_with_package=21,
    )

    serialized = json.dumps(artifact, ensure_ascii=False)
    assert private_fact not in serialized
    assert "private retained replay" not in serialized
    assert artifact["text_free"] is True
    assert artifact["label"]["chars"] == len("private retained replay")
    assert artifact["prompt"]["delta"]["prepared_context_marker_added"] is True
    assert artifact["request_path"]["llm_call_delta"] == 0
    assert artifact["request_path"]["sql_statement_delta"] == 1
    assert artifact["rollout_checks"]["package_read_added_llm_call"] is False
    assert artifact["rollout_checks"]["package_statuses"] == ["hit"]
    package_diag = artifact["initial_context_package"]["with_package"]["packages"][0]
    assert package_diag["package_kind"] == "conversation"
    assert package_diag["status"] == "hit"
    assert package_diag["package_key_hash"]["chars"] == len("icp:v1:test")
    assert "source_refs" not in package_diag

    artifact_path = write_initial_context_package_prompt_diff_artifact(
        artifact,
        tmp_path,
        filename="artifact.json",
    )
    assert artifact_path.read_text(encoding="utf-8").startswith("{")
