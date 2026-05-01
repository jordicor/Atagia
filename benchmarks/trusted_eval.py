"""Helpers for trusted local benchmark evaluation."""

from __future__ import annotations

from typing import Any

from atagia.models.schemas_memory import MemoryStatus
from atagia.models.schemas_replay import AblationConfig


TRUSTED_EVALUATION_PROMPT_NOTE = (
    "Trusted benchmark evaluation mode is active. The requester is the authenticated "
    "benchmark owner in a controlled local evaluation. This mode may surface "
    "sensitive retrieved context so coverage can be audited, but it does not override "
    "the ordinary high-risk chat disclosure policy. Answer from retrieved context when "
    "the fact is permitted for ordinary authenticated use, and continue withholding raw "
    "secret literals such as credentials, PINs, passwords, and payment-card secrets. "
    "Do not invent facts that are not present in retrieved context."
)


def trusted_evaluation_ablation(ablation: AblationConfig | None) -> AblationConfig:
    """Return retrieval settings for a trusted benchmark run."""
    if ablation is None:
        return AblationConfig(override_retrieval_params={"privacy_ceiling": 3})
    override_params = dict(ablation.override_retrieval_params or {})
    override_params["privacy_ceiling"] = 3
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
