from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import py_compile
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def test_copyable_python_platform_scaffolds_compile(tmp_path: Path) -> None:
    for relative_path in (
        "integrations/aurvek/atagia_bridge.py",
        "integrations/open-webui/atagia_memory_filter.py",
        "integrations/openclaw/atagia_adapter.py",
        "integrations/hermes/atagia_provider.py",
    ):
        output = tmp_path / f"{Path(relative_path).stem}.pyc"
        py_compile.compile(str(ROOT / relative_path), cfile=str(output), doraise=True)


def test_copyable_python_platform_scaffolds_import() -> None:
    modules = {
        "aurvek_atagia_bridge": ROOT / "integrations/aurvek/atagia_bridge.py",
        "atagia_memory_filter": ROOT / "integrations/open-webui/atagia_memory_filter.py",
        "atagia_adapter": ROOT / "integrations/openclaw/atagia_adapter.py",
        "atagia_provider": ROOT / "integrations/hermes/atagia_provider.py",
    }

    loaded = {name: _load_module(name, path) for name, path in modules.items()}

    assert hasattr(loaded["aurvek_atagia_bridge"], "get_context_for_turn")
    assert hasattr(loaded["atagia_memory_filter"], "Filter")
    assert hasattr(loaded["atagia_adapter"], "AtagiaOpenClawAdapter")
    assert hasattr(loaded["atagia_provider"], "AtagiaHermesProvider")


def test_open_webui_filter_encodes_conversation_path_segments() -> None:
    module = _load_module(
        "atagia_memory_filter",
        ROOT / "integrations/open-webui/atagia_memory_filter.py",
    )

    encoded = module._path_segment("chat/with spaces?#")
    assert encoded.startswith("__atagia_b64_")
    assert "/" not in encoded
    assert module._path_segment("safe-chat_1") == "safe-chat_1"
    assert module._path_segment(".").startswith("__atagia_b64_")


def test_sillytavern_extension_scaffold_has_declared_assets() -> None:
    extension_dir = ROOT / "integrations" / "sillytavern" / "extension"
    manifest = json.loads((extension_dir / "manifest.json").read_text())

    assert manifest["display_name"] == "Atagia Memory"
    assert manifest["js"] == "index.js"
    assert manifest["css"] == "style.css"
    assert (extension_dir / manifest["js"]).is_file()
    assert (extension_dir / manifest["css"]).is_file()
    assert manifest["generate_interceptor"] == "atagiaMemoryInterceptor"


def test_sillytavern_extension_uses_injective_chat_id_encoding() -> None:
    source = (ROOT / "integrations/sillytavern/extension/index.js").read_text(
        encoding="utf-8"
    )

    assert "function transportId" in source
    assert "base64UrlEncode" in source
    assert "JSON.stringify" in source
    assert "atagia_conversation_id" in source
    assert "context.saveMetadata?.()" in source
    assert "replace(/[^A-Za-z0-9_.:-]/g, '_')" not in source
    assert "context.characterId" not in source
    assert "default-chat" not in source
    assert "${transportId(current.conversationPrefix || 'sillytavern')}-${transportId(raw)}" not in source


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
        sys.modules.pop(name, None)
    return module
