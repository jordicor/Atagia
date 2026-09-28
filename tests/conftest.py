"""Test bootstrap for src-layout imports."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Test settings must come only from each test's explicit environment. Loading
# a developer checkout's private .env makes resource and provider tests depend
# on machine-local configuration.
import dotenv  # noqa: E402

from atagia.core import config as config_module  # noqa: E402

config_module._DOTENV_LOADED = True


def _blocked_load_dotenv(*_args: object, **_kwargs: object) -> bool:
    """Refuse to import a private .env into os.environ during a test session.

    Blocking only the engine's own loader was not enough: several benchmark
    modules call ``load_dotenv()`` at import time, and tests import them for
    prompt-fidelity checks. One such import pollutes os.environ for the whole
    session, including every subprocess a later test spawns -- which is how a
    developer's ATAGIA_BASE_URL reached an integration smoke test running in
    node and pointed its plugin at a real server instead of the test stub.
    """
    # Deliberate: report "no .env loaded" so tests never read machine-local config.
    return False


dotenv.load_dotenv = _blocked_load_dotenv
for resource_variable in (
    "ATAGIA_MIGRATIONS_PATH",
    "ATAGIA_MANIFESTS_PATH",
    "ATAGIA_OPERATIONAL_PROFILES_PATH",
):
    os.environ.pop(resource_variable, None)
