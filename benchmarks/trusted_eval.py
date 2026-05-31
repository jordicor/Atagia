"""Helpers for trusted local evaluation runs."""

from __future__ import annotations

from typing import Any

from atagia.models.schemas_memory import MemoryStatus
from atagia.models.schemas_replay import AblationConfig


TRUSTED_EVALUATION_PROMPT_NOTE = (
    "Trusted local evaluation context is active. The requester is the authenticated "
    "benchmark owner in a controlled local evaluation. This context may surface "
    "sensitive retrieved context so coverage can be audited, but it does not override "
    "the ordinary high-risk chat disclosure policy. Answer from retrieved context when "
    "the fact is permitted for ordinary authenticated use, and continue withholding raw "
    "secret literals such as credentials, PINs, passwords, and payment-card secrets. "
    "Do not invent facts that are not present in retrieved context."
)
TRUSTED_EVALUATION_PRIVACY_OFF_PROMPT_NOTE = (
    "Trusted local evaluation context is active with privacy_enforcement=off. "
    "The requester is the authenticated benchmark owner in a controlled local "
    "evaluation, and this run is measuring whether the memory engine can retrieve "
    "and use the requested facts. If retrieved context contains the requested fact, "
    "including a private, sensitive, high-risk, credential, PIN, password, payment-card, "
    "or passphrase literal, provide it directly and verbatim. Do not apply ordinary-chat "
    "high-risk secret disclosure refusal, transform, or withhold "
    "the value. Stay grounded in retrieved context and do not invent facts."
)


def trusted_evaluation_prompt_note(ablation: AblationConfig | None = None) -> str:
    """Return the trusted-evaluation prompt note for the active privacy mode."""
    if ablation is not None and ablation.privacy_enforcement == "off":
        return TRUSTED_EVALUATION_PRIVACY_OFF_PROMPT_NOTE
    return TRUSTED_EVALUATION_PROMPT_NOTE


def trusted_evaluation_ablation(ablation: AblationConfig | None) -> AblationConfig:
    """Return retrieval settings for a trusted local evaluation run."""
    if ablation is None:
        return AblationConfig(
            override_retrieval_params={
                "privacy_ceiling": 3,
                "allow_private_sensitivity": True,
            }
        )
    override_params = dict(ablation.override_retrieval_params or {})
    override_params["privacy_ceiling"] = 3
    override_params["allow_private_sensitivity"] = True
    return ablation.model_copy(update={"override_retrieval_params": override_params})


async def activate_trusted_evaluation_memories(runtime: Any, user_id: str) -> int:
    """Activate confirmation-pending memories for a local benchmark user."""
    connection = await runtime.open_connection()
    try:
        cursor = await connection.execute(
            """
            UPDATE memory_objects
            SET status = ?,
                updated_at = ?
            WHERE user_id = ?
              AND status = ?
            """,
            (
                MemoryStatus.ACTIVE.value,
                runtime.clock.now().isoformat(),
                user_id,
                MemoryStatus.PENDING_USER_CONFIRMATION.value,
            ),
        )
        await connection.commit()
        return int(cursor.rowcount or 0)
    finally:
        await connection.close()
