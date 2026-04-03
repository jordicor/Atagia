"""Shared constants and patterns for chunking oversized extraction inputs."""

from __future__ import annotations

import re

CHUNKING_THRESHOLD_TOKENS = 2000
LEVEL0_MIN_SEGMENT_CHARS = 500
LEVEL1_MAX_CHUNK_TOKENS = 16000
LEVEL1_TARGET_CHUNK_TOKENS = 8000
LEVEL1_MIN_CHUNK_TOKENS = 4000
LEVEL1_MARKER_INTERVAL = 1000
LEVEL1_WINDOW_TOKENS = 30000
LEVEL1_OVERLAP_MARKERS = 3
PRIOR_CHUNK_CONTEXT_MAX_TOKENS = 2000

_LINE_PATTERNS = (
    r"(?P<whisper_ts>\[\d{2}:\d{2}:\d{2}(?:\.\d{3})?\s*-->\s*\d{2}:\d{2}:\d{2}(?:\.\d{3})?\])",
    r"(?P<timestamp>\[?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{3})?\]?(?:\s*(?:AM|PM))?)",
    r"(?P<date>\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})",
    r"(?P<speaker>(?:[A-Z][A-Za-z0-9 .'\-]{0,40}|[A-Z]{1,4}|Yo|Tu|Tú|Doctor|Paciente):)",
    r"(?P<numbered>(?:\(?\d+\)|\d+\.)|\[\d+\])\s+",
)

HORIZONTAL_RULE_PATTERN = re.compile(r"(?m)^\s*(?:[-=*]\s*){3,}$")
BLANK_BLOCK_PATTERN = re.compile(r"\n\s*\n+")
LINE_SEPARATOR_PATTERN = re.compile(
    r"(?m)^(?:"
    + "|".join(_LINE_PATTERNS)
    + r"|(?:\[[^\d\W][^\]\n]{1,40}\]))",
)
