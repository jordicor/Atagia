"""Command-line entry points for running Atagia services."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    """Run the Atagia FastAPI service."""
    parser = argparse.ArgumentParser(
        description="Run the Atagia API service.",
        epilog=(
            "Service callers must provide explicit memory identity: user_id, "
            "conversation_id, platform_id, optional user_persona_id, optional "
            "character_id, mode, and incognito. Legacy assistant_mode_id and "
            "workspace_id are compatibility aliases, not memory namespaces."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8100, type=int)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    uvicorn.run(
        "atagia.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
