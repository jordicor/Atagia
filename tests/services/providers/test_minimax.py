"""Tests for the MiniMax provider adapter and provider factory wiring."""

from __future__ import annotations

from atagia.core.config import Settings
from atagia.services.providers import build_llm_client
from atagia.services.providers.minimax import MiniMaxProvider


def test_minimax_default_base_url_and_factory_wiring(monkeypatch) -> None:
    captured = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr("atagia.services.providers.openai.AsyncOpenAI", FakeAsyncOpenAI)

    provider = MiniMaxProvider(api_key="minimax-key")

    assert provider.name == "minimax"
    assert captured[-1]["base_url"] == "https://api.minimax.io/v1"

    settings = Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        minimax_api_key="minimax-key",
        openrouter_api_key=None,
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        llm_forced_global_model="minimax/MiniMax-M3",
    )
    client = build_llm_client(settings)

    assert client.provider_name is None
    assert client._provider("minimax").name == "minimax"
