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
from atagia.core import config as config_module  # noqa: E402

config_module._DOTENV_LOADED = True
for resource_variable in (
    "ATAGIA_MIGRATIONS_PATH",
    "ATAGIA_MANIFESTS_PATH",
    "ATAGIA_OPERATIONAL_PROFILES_PATH",
):
    os.environ.pop(resource_variable, None)
