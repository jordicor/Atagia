"""Async client for the OpenAI Privacy Filter sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from time import perf_counter
from typing import Any

import httpx

from atagia.core.config import Settings


class PrivacyFilterError(RuntimeError):
    """Base error for privacy filter calls."""


class PrivacyFilterUnavailable(PrivacyFilterError):
    """Raised when no configured OPF endpoint can answer."""


@dataclass(frozen=True, slots=True)
class PrivacyFilterSpan:
    """One detected OPF span without raw span text."""

    label: str
    start: int
    end: int
    text_sha256: str | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }
        if self.text_sha256 is not None:
            payload["text_sha256"] = self.text_sha256
        return payload


@dataclass(frozen=True, slots=True)
class PrivacyFilterDetection:
    """OPF detection result for one text field."""

    spans: list[PrivacyFilterSpan]
    endpoint_used: str | None
    latency_ms: float

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def labels(self) -> list[str]:
        seen: set[str] = set()
        labels: list[str] = []
        for span in self.spans:
            if span.label in seen:
                continue
            seen.add(span.label)
            labels.append(span.label)
        return labels


class OpenAIPrivacyFilterClient:
    """Async OPF sidecar client with primary/fallback failover."""

    def __init__(
        self,
        *,
        primary_url: str,
        fallback_url: str,
        timeout_seconds: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._primary_url = primary_url.rstrip("/")
        self._fallback_url = fallback_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAIPrivacyFilterClient":
        return cls(
            primary_url=settings.opf_primary_url,
            fallback_url=settings.opf_fallback_url,
            timeout_seconds=settings.opf_timeout_seconds,
        )

    @property
    def attempted_endpoints(self) -> tuple[str, ...]:
        """Return OPF endpoints attempted during failover."""
        return self._candidate_urls()

    async def health(self) -> tuple[str, dict[str, Any]]:
        """Return the first endpoint that responds to /health."""

        last_error: Exception | None = None
        for base_url in self._candidate_urls():
            try:
                payload = await self._request_json("GET", f"{base_url}/health")
                if not isinstance(payload, dict):
                    raise PrivacyFilterError(f"OPF /health returned non-object JSON from {base_url}")
            except Exception as exc:  # noqa: BLE001 - failover needs the last cause.
                last_error = exc
                continue
            return base_url, payload
        raise PrivacyFilterUnavailable(
            "OPF unreachable on primary and fallback endpoints"
        ) from last_error

    async def detect(self, text: str) -> PrivacyFilterDetection:
        """Run /detect and return spans without storing raw span text."""

        if not text.strip():
            return PrivacyFilterDetection(spans=[], endpoint_used=None, latency_ms=0.0)

        last_error: Exception | None = None
        for base_url in self._candidate_urls():
            start = perf_counter()
            try:
                payload = await self._request_json(
                    "POST",
                    f"{base_url}/detect",
                    json={"text": text},
                )
                spans = self._parse_spans(payload, text)
            except Exception as exc:  # noqa: BLE001 - failover needs the last cause.
                last_error = exc
                continue
            latency_ms = (perf_counter() - start) * 1000.0
            return PrivacyFilterDetection(
                spans=spans,
                endpoint_used=base_url,
                latency_ms=latency_ms,
            )
        raise PrivacyFilterUnavailable(
            "OPF /detect unreachable on primary and fallback endpoints"
        ) from last_error

    def _candidate_urls(self) -> tuple[str, ...]:
        if self._fallback_url == self._primary_url:
            return (self._primary_url,)
        return (self._primary_url, self._fallback_url)

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> Any:
        if self._http_client is not None:
            response = await self._http_client.request(
                method,
                url,
                timeout=self._timeout_seconds,
                **kwargs,
            )
            response.raise_for_status()
            return response.json()

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

    @classmethod
    def _parse_spans(cls, payload: Any, source_text: str) -> list[PrivacyFilterSpan]:
        if not isinstance(payload, dict):
            raise PrivacyFilterError("OPF /detect returned non-object JSON")
        raw_spans = payload.get("spans")
        if not isinstance(raw_spans, list):
            raise PrivacyFilterError("OPF /detect response missing spans list")

        spans: list[PrivacyFilterSpan] = []
        for raw_span in raw_spans:
            if not isinstance(raw_span, dict):
                raise PrivacyFilterError("OPF /detect span was not an object")
            label = str(raw_span.get("label") or "").strip()
            start = int(raw_span.get("start", 0))
            end = int(raw_span.get("end", 0))
            text_sha256 = cls._span_hash(source_text, start, end)
            spans.append(
                PrivacyFilterSpan(
                    label=label,
                    start=start,
                    end=end,
                    text_sha256=text_sha256,
                )
            )
        return spans

    @staticmethod
    def _span_hash(source_text: str, start: int, end: int) -> str | None:
        if start < 0 or end <= start or end > len(source_text):
            return None
        return hashlib.sha256(source_text[start:end].encode("utf-8")).hexdigest()
