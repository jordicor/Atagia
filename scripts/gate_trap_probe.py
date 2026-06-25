"""Ad-hoc probe: does the adaptive-gate classifier still SKIP when it should?

Runs the real parallel-card need detection (same cards + merge + Gemini model as
the benchmark) over trap/control questions and prints, per case, the resolved
``memory_dependence``, ``exact_recall_needed`` and whether the adaptive gate
WOULD skip memory retrieval (``memory_dependence in {world, conversation}``).

Trap cases mention a stored person name but ask general knowledge: the gate must
still skip them. Control cases verify pure world/conversation still skip and that
genuinely personal questions still retrieve. Not part of atagia-bench; no DB.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.memory.need_detector import NeedDetector
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExtractionContextMessage,
    ExtractionConversationContext,
    MemoryDependence,
)
from atagia.services.llm_client import LLMClient, RetryPolicy
from atagia.services.providers.openrouter import OpenRouterProvider

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
SKIP_CLASSES = {MemoryDependence.WORLD.value, MemoryDependence.CONVERSATION.value}

# Each case: (label, mode, question, recent_messages, expect_skip)
NEUTRAL = [
    ExtractionContextMessage(role="user", content="Hi, thanks for the help earlier."),
    ExtractionContextMessage(role="assistant", content="Happy to help. What next?"),
]
CASES: list[tuple[str, str, str, list[ExtractionContextMessage], bool]] = [
    # --- TRAPS: stored name mentioned, but the question is general knowledge -> SKIP ---
    ("trap-EN-caesar", "personal_assistant",
     "Ben told me you love the Roman Empire. Do you know when Julius Caesar died, and how?",
     NEUTRAL, True),
    ("trap-ES-luna", "personal_assistant",
     "Rosa me dijo que te gusta la astronomia. A que distancia esta la Luna de la Tierra?",
     NEUTRAL, True),
    ("trap-FR-hugo", "personal_assistant",
     "Claire dit que tu aimes l'histoire. Qui a ecrit Les Miserables?",
     NEUTRAL, True),
    # --- PURE WORLD: no personal anchor -> SKIP ---
    ("world-capital", "personal_assistant",
     "What is the capital of Australia?", NEUTRAL, True),
    ("world-joke", "coding_debug",
     "Tell me a joke about programmers.", NEUTRAL, True),
    # --- CONVERSATION: answerable from the visible turns -> SKIP ---
    ("conv-summarize", "personal_assistant",
     "Can you summarize what you just told me?",
     [ExtractionContextMessage(role="assistant",
        content="To deploy, run the migration, then restart the workers, then flip the flag.")],
     True),
    ("conv-lastmsg", "personal_assistant",
     "What did I ask in my previous message?",
     [ExtractionContextMessage(role="user", content="How do I reset my router?")],
     True),
    # --- PERSONAL / MIXED: genuinely need stored memory -> DO NOT skip ---
    ("personal-meds", "personal_assistant",
     "What's the dose of my thyroid medication now?", NEUTRAL, False),
    ("personal-code", "personal_assistant",
     "What's my building entry code?", NEUTRAL, False),
    ("mixed-trip", "personal_assistant",
     "We were discussing my upcoming move. What should I do first?",
     [ExtractionContextMessage(role="user", content="I'm moving next month and feeling overwhelmed.")],
     False),
]


async def main() -> int:
    settings = Settings.from_env()
    if not settings.openrouter_api_key:
        print("ERROR: ATAGIA_OPENROUTER_API_KEY not set")
        return 1
    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        site_url=settings.openrouter_site_url or "https://atagia.org",
        app_name=settings.openrouter_app_name or "Atagia gate probe",
    )
    client: LLMClient[Any] = LLMClient(
        providers=[provider],
        retry_policy=RetryPolicy(attempts=2, base_delay_seconds=0.5, max_delay_seconds=2.0),
        structured_output_retry_attempts=1,
    )
    clock = FrozenClock(datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc))
    detector = NeedDetector(client, clock, settings)
    loader = ManifestLoader(MANIFESTS_DIR)
    manifests = loader.load_all()
    resolver = PolicyResolver()

    hdr = (f"{'case':<18}{'mem_dep':<12}{'exact':<6}{'qtype':<11}{'skip':<6}"
           f"{'expect':<9}{'OK':<4}{'mem_raw':<14}{'shape_raw':<10}exact_raw")
    print(hdr)
    print("-" * len(hdr))
    ok_count = 0
    for label, mode, question, recent, expect_skip in CASES:
        policy = resolver.resolve(manifests[mode], None, None)
        ctx = ExtractionConversationContext(
            user_id="usr_probe", conversation_id="cnv_probe", source_message_id="msg_probe",
            workspace_id="wrk_probe", assistant_mode_id=mode, recent_messages=recent,
        )
        sink: list[Any] = []
        result = await detector.detect(
            message_text=question, role="user", conversation_context=ctx,
            resolved_policy=policy, content_language_profile=[],
            card_call_trace_sink=sink,
        )
        dep = result.memory_dependence.value
        skip = dep in SKIP_CLASSES
        ok = skip == expect_skip
        ok_count += ok
        def raw(name: str) -> str:
            return next((str(c.raw_output) for c in sink if c.card_name == name), "?").replace("\n", "|")
        print(f"{label:<18}{dep:<12}{str(result.exact_recall_needed):<6}{result.query_type:<11}"
              f"{str(skip):<6}{('skip' if expect_skip else 'retrieve'):<9}"
              f"{('OK' if ok else 'XX'):<4}{raw('memory')[:13]:<14}{raw('shape')[:9]:<10}{raw('exact')}")
    print("-" * len(hdr))
    print(f"matched expectation: {ok_count}/{len(CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
