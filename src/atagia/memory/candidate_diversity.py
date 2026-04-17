"""Deterministic pre-scorer diversity selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from atagia.models.schemas_memory import QueryType

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_MIN_TOKEN_LENGTH = 3
_MAX_CLUSTER_SHARE = 0.60
_JACCARD_THRESHOLD = 0.30
_MAX_TARGET_CLUSTERS = 4


@dataclass(slots=True)
class _Cluster:
    representative: dict[str, Any]
    members: list[dict[str, Any]] = field(default_factory=list)
    representative_tokens: set[str] = field(default_factory=set)


def early_diversity_select(
    candidates: list[dict[str, Any]],
    *,
    query_type: QueryType,
    shortlist_k: int,
) -> list[dict[str, Any]]:
    """Select a broad-list shortlist with deterministic cluster coverage."""
    if shortlist_k <= 0:
        return []
    if query_type != "broad_list" or len(candidates) <= shortlist_k:
        return candidates[:shortlist_k]

    token_sets = _candidate_token_sets(candidates)
    clusters = _cluster_candidates(candidates, token_sets)
    if not clusters:
        return candidates[:shortlist_k]

    target_clusters = min(_MAX_TARGET_CLUSTERS, shortlist_k, len(clusters))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for cluster in clusters[:target_clusters]:
        candidate = cluster.members[0]
        candidate_id = str(candidate["id"])
        if candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)

    cluster_queues = [cluster.members[1:] for cluster in clusters]
    while len(selected) < shortlist_k:
        advanced = False
        for queue in cluster_queues:
            while queue:
                candidate = queue.pop(0)
                candidate_id = str(candidate["id"])
                if candidate_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate_id)
                advanced = True
                break
            if len(selected) >= shortlist_k:
                break
        if not advanced:
            break

    if len(selected) < shortlist_k:
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            if candidate_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate_id)
            if len(selected) >= shortlist_k:
                break

    return selected[:shortlist_k]


def _candidate_token_sets(candidates: list[dict[str, Any]]) -> list[set[str]]:
    token_sets = [_tokens_for_candidate(candidate) for candidate in candidates]
    if not token_sets:
        return []

    document_frequency: dict[str, int] = {}
    for token_set in token_sets:
        for token in token_set:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    max_frequency = max(1, int(len(candidates) * _MAX_CLUSTER_SHARE))
    filtered_sets: list[set[str]] = []
    for token_set in token_sets:
        filtered_sets.append(
            {
                token
                for token in token_set
                if document_frequency.get(token, 0) <= max_frequency
            }
        )
    return filtered_sets


def _tokens_for_candidate(candidate: dict[str, Any]) -> set[str]:
    text_parts = [
        str(candidate.get("canonical_text", "")),
        str(candidate.get("index_text", "")),
    ]
    tokens: set[str] = set()
    for text in text_parts:
        for match in _TOKEN_PATTERN.finditer(text.lower()):
            token = match.group(0).strip()
            if len(token) < _MIN_TOKEN_LENGTH:
                continue
            if not any(character.isalnum() for character in token):
                continue
            tokens.add(token)
    return tokens


def _cluster_candidates(
    candidates: list[dict[str, Any]],
    token_sets: list[set[str]],
) -> list[_Cluster]:
    clusters: list[_Cluster] = []
    for candidate, token_set in zip(candidates, token_sets, strict=True):
        best_cluster: _Cluster | None = None
        best_similarity = 0.0
        for cluster in clusters:
            similarity = _jaccard_similarity(token_set, cluster.representative_tokens)
            if similarity >= _JACCARD_THRESHOLD and similarity > best_similarity:
                best_cluster = cluster
                best_similarity = similarity
        if best_cluster is None:
            clusters.append(
                _Cluster(
                    representative=candidate,
                    members=[candidate],
                    representative_tokens=token_set,
                )
            )
            continue
        best_cluster.members.append(candidate)
    return clusters


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    union = left | right
    if not union:
        return 0.0
    return len(overlap) / len(union)
