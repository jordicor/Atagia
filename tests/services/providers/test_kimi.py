"""Tests for the Kimi provider adapter and provider factory wiring."""

from __future__ import annotations

from atagia.core.config import Settings
from atagia.services.providers import build_llm_client
from atagia.services.providers.kimi import KimiProvider


def test_kimi_default_base_url_and_factory_wiring(monkeypatch) -> None:
    captured = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr("atagia.services.providers.openai.AsyncOpenAI", FakeAsyncOpenAI)

    provider = KimiProvider(api_key="kimi-key")

    assert provider.name == "kimi"
    assert captured[-1]["base_url"] == "https://api.moonshot.ai/v1"

    settings = Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        kimi_api_key="kimi-key",
        openrouter_api_key=None,
        openrouter_site_url="https://atagia.org",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        llm_forced_global_model="kimi/kimi-k2.7-code",
    )
    client = build_llm_client(settings)

    assert client.provider_name is None
    assert client._provider("kimi").name == "kimi"
