"""Small status CLI for the copyable Atagia Hermes provider."""

from __future__ import annotations

import json

from .provider import AtagiaMemoryProvider


def main() -> None:
    provider = AtagiaMemoryProvider()
    print(json.dumps(provider.status(), indent=2, sort_keys=True))
    provider.shutdown()


if __name__ == "__main__":
    main()
