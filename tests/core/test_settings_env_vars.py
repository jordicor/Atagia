"""Lockstep guard for the declared env var sources of every Settings field.

Provenance reporting reads ``SETTINGS_ENV_VARS`` to decide whether a field was
configured from the environment. If ``Settings.from_env`` starts reading a
variable that is not declared there, the report silently calls a configured
field ``default``. This test re-derives the mapping from the ``from_env`` source
and fails the build on any drift.

It also pins the purity of ``from_env``: an explicit environment mapping is read
exactly as given, without touching or depending on process-global state.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
from pathlib import Path

import pytest

from atagia.core import config as config_module
from atagia.core.config import Settings
from atagia.core.settings_env_vars import (
    COMPONENT_EXAMPLES_ENV_VARS,
    COMPONENT_MODEL_ENV_VARS,
    INTIMACY_COMPONENT_MODEL_ENV_VARS,
    SETTINGS_ENV_VARS,
)

_CONFIG_PATH = Path(inspect.getfile(config_module))

# Helpers that read a whole family of per-component variables from the mapping
# instead of one named variable.
_FAMILY_READERS = {
    "component_env_models_from_env": COMPONENT_MODEL_ENV_VARS,
    "component_env_examples_from_env": COMPONENT_EXAMPLES_ENV_VARS,
    "intimacy_component_env_models_from_env": INTIMACY_COMPONENT_MODEL_ENV_VARS,
}


def _env_vars_read_by(node: ast.AST) -> tuple[str, ...]:
    """Every env var name one ``from_env`` keyword argument reads, in order."""
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "env"
        ):
            assert child.args, f"env reader call without a name: {ast.dump(child)}"
            first = child.args[0]
            assert isinstance(first, ast.Constant) and isinstance(first.value, str), (
                f"env var name must be a literal: {ast.dump(child)}"
            )
            if first.value not in names:
                names.append(first.value)
        elif isinstance(func, ast.Name) and func.id in _FAMILY_READERS:
            names.extend(_FAMILY_READERS[func.id])
    return tuple(names)


def _declared_env_vars_from_source() -> dict[str, tuple[str, ...]]:
    tree = ast.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
    from_env = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "from_env"
    )
    settings_call = next(
        node
        for node in ast.walk(from_env)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cls"
    )
    derived: dict[str, tuple[str, ...]] = {}
    for keyword in settings_call.keywords:
        assert keyword.arg is not None, "from_env must not use **kwargs expansion"
        derived[keyword.arg] = _env_vars_read_by(keyword.value)
    return derived


def test_declared_env_vars_match_from_env() -> None:
    """``SETTINGS_ENV_VARS`` mirrors exactly what ``from_env`` reads per field."""
    derived = _declared_env_vars_from_source()

    assert derived == SETTINGS_ENV_VARS, (
        "SETTINGS_ENV_VARS drifted from Settings.from_env; update "
        "src/atagia/core/settings_env_vars.py to match the env vars each field "
        "is read from"
    )


def test_every_settings_field_declares_its_env_vars() -> None:
    """No field may be provenance-blind: each one declares where it is read from."""
    field_names = {field.name for field in dataclasses.fields(Settings)}

    assert set(SETTINGS_ENV_VARS) == field_names
    for name, env_vars in SETTINGS_ENV_VARS.items():
        assert env_vars, f"{name} declares no environment variable"


def test_from_env_reads_an_explicit_mapping_without_touching_the_process() -> None:
    """An explicit mapping is the whole environment for that call.

    The process environment is neither read nor written, which is what makes a
    baseline resolution safe: the previous implementation cleared and restored
    ``os.environ`` and could pull a developer's ``.env`` into the cleared window.
    """
    os.environ["ATAGIA_TEST_ENV_PURITY"] = "sentinel"
    try:
        before = dict(os.environ)
        settings = Settings.from_env({"ATAGIA_RRF_K": "123"})
        assert dict(os.environ) == before
    finally:
        del os.environ["ATAGIA_TEST_ENV_PURITY"]

    assert settings.rrf_k == 123
    # Nothing else leaked in from the real environment.
    assert settings.sqlite_path == Settings.from_env({}).sqlite_path


def test_from_env_with_explicit_mapping_never_loads_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``.env`` loader is reserved for the process-environment path."""

    def _fail() -> None:
        raise AssertionError("from_env(mapping) must not load .env")

    monkeypatch.setattr(config_module, "_load_dotenv_once", _fail)

    assert Settings.from_env({}).rrf_k == 60
