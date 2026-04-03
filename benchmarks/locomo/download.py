"""Download the LoCoMo dataset used by the benchmark harness."""

from __future__ import annotations

from pathlib import Path
import urllib.request

_RAW_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
_DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"


def download_locomo_dataset(output_path: str | Path | None = None) -> Path:
    """Download `locomo10.json` unless it already exists locally."""
    target_path = Path(output_path).expanduser() if output_path is not None else _DEFAULT_OUTPUT_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path
    with urllib.request.urlopen(_RAW_URL) as response:
        target_path.write_bytes(response.read())
    return target_path


def main() -> None:
    """Download the dataset and print the resulting file path."""
    path = download_locomo_dataset()
    print(path)


if __name__ == "__main__":
    main()
