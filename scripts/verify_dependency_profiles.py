#!/usr/bin/env python3
"""Lock, install, inventory, smoke-test, and audit supported dependency profiles."""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
import hashlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from typing import Sequence
import unicodedata
from zipfile import ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
LOCKS_DIR = ROOT / "requirements" / "locks"
BLOG_INPUT = ROOT / "requirements" / "profiles" / "blog.in"
BLOG_ENGINE_ROOT = ROOT / "_blog_engine"
BLOG_ARTIFACT_PATHS = (
    Path("blog_security.py"),
    Path("build.py"),
    Path("publication.py"),
    Path("release_provenance.py"),
    Path("content"),
    Path("templates"),
)
SUPPORTED_MINORS = {(3, 12), (3, 13)}
PYTHON_PROFILES = ("core", "mcp", "dev", "blog")
NODE_ARTIFACTS = (
    ROOT / "integrations" / "openclaw" / "plugin",
    ROOT / "integrations" / "sillytavern" / "server-plugin",
)
_PIN_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+==[^;\s]+(?:\s*;.*)?$")
DEPENDENCY_REPORT_SCHEMA = "atagia.dependency-verification-report.v1"
DEPENDENCY_PROFILE_SCHEMA = "atagia.dependency-profile-result.v1"
SDIST_SETUP_CFG = b"[egg_info]\ntag_build = \ntag_date = 0\n\n"
SDIST_GENERATED_FILES = frozenset(
    {
        "PKG-INFO",
        "setup.cfg",
        "src/atagia.egg-info/PKG-INFO",
        "src/atagia.egg-info/SOURCES.txt",
        "src/atagia.egg-info/dependency_links.txt",
        "src/atagia.egg-info/entry_points.txt",
        "src/atagia.egg-info/requires.txt",
        "src/atagia.egg-info/top_level.txt",
    }
)
SDIST_CANONICAL_ROOT_FILES = frozenset(
    {"LICENSE", "README.md", "pyproject.toml", "setup.py"}
)
SDIST_CANONICAL_TEST_FILES = frozenset(
    {
        "tests/test_client.py",
        "tests/test_embedding_backfill_cli.py",
        "tests/test_engine.py",
        "tests/test_mcp_server.py",
        "tests/test_retrieval_trace.py",
    }
)


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    name: str
    source: Path
    extras: tuple[str, ...]
    wheel_extras: tuple[str, ...]


PROFILE_SPECS = {
    "core": ProfileSpec("core", ROOT / "pyproject.toml", (), ()),
    "mcp": ProfileSpec("mcp", ROOT / "pyproject.toml", ("mcp",), ("mcp",)),
    "dev": ProfileSpec(
        "dev",
        ROOT / "pyproject.toml",
        ("dev", "mcp", "embeddings"),
        ("dev", "mcp", "embeddings"),
    ),
    "blog": ProfileSpec("blog", BLOG_INPUT, (), ()),
}


@dataclass(frozen=True, slots=True)
class ProfileResult:
    schema: str
    profile: str
    python: str
    profile_source: str
    profile_source_sha256: str
    lock_file: str
    lock_sha256: str
    inventory_file: str
    inventory_sha256: str
    audit: str
    audit_file: str | None
    audit_sha256: str | None
    smoke: str
    artifact_evidence: dict[str, object] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("lock", "check-locks"):
        child = subparsers.add_parser(command)
        child.add_argument("--python", default=sys.executable)
        child.add_argument("--profile", action="append", choices=PYTHON_PROFILES)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--python", default=sys.executable)
    verify.add_argument(
        "--profile",
        action="append",
        choices=(*PYTHON_PROFILES, "node"),
    )
    verify.add_argument("--artifacts-dir", default="build/dependency-audit")
    verify.add_argument("--no-audit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    python = _resolve_python(args.python)
    if args.command in {"lock", "check-locks"}:
        profiles = tuple(dict.fromkeys(args.profile or PYTHON_PROFILES))
        if args.command == "lock":
            _write_locks(python, profiles)
        else:
            _check_locks(python, profiles)
        return 0

    profiles = tuple(dict.fromkeys(args.profile or (*PYTHON_PROFILES, "node")))
    artifacts_dir = (ROOT / args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results = _verify_profiles(
        python,
        profiles,
        artifacts_dir=artifacts_dir,
        audit=not args.no_audit,
    )
    report = {
        "schema": DEPENDENCY_REPORT_SCHEMA,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "profiles": [asdict(result) for result in results],
    }
    (artifacts_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def _resolve_python(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        selected = candidate
    else:
        command = value if value.startswith("python") else f"python{value}"
        found = shutil.which(command)
        if found is None:
            raise RuntimeError(f"Python interpreter not found: {value}")
        selected = Path(found)
    selected = selected.absolute()
    if not selected.is_file():
        raise RuntimeError(f"Python interpreter not found: {value}")
    # Do not resolve a virtualenv's interpreter symlink: CPython uses the
    # invoked path to discover pyvenv.cfg and therefore the locked tool env.
    return selected


def _python_minor(python: Path) -> tuple[int, int]:
    output = _capture(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ]
    )
    major, minor = (int(part) for part in output.strip().split("."))
    if (major, minor) not in SUPPORTED_MINORS:
        raise RuntimeError(
            f"Unsupported dependency profile interpreter: Python {major}.{minor}"
        )
    return major, minor


def _lock_path(profile: str, minor: tuple[int, int]) -> Path:
    return LOCKS_DIR / f"{profile}-py{minor[0]}{minor[1]}.txt"


def _compile_profile(
    python: Path,
    spec: ProfileSpec,
    output: Path,
    *,
    constraint: Path | None = None,
) -> None:
    command = [
        str(python),
        "-m",
        "piptools",
        "compile",
        "--quiet",
        "--allow-unsafe",
        "--no-header",
        "--no-annotate",
        "--strip-extras",
        "--no-emit-index-url",
        f"--output-file={output}",
    ]
    if constraint is not None:
        command.append(f"--constraint={constraint}")
    for extra in spec.extras:
        command.append(f"--extra={extra}")
    command.append(str(spec.source))
    _run(command, cwd=ROOT)


def _lock_header(spec: ProfileSpec, minor: tuple[int, int]) -> str:
    source = spec.source.relative_to(ROOT)
    source_label = str(source)
    if spec.extras:
        source_label += f" (extras: {', '.join(spec.extras)})"
    return (
        f"# Generated by pip-tools with CPython {minor[0]}.{minor[1]}.\n"
        f"# Source: {source_label}\n"
        f"# Regenerate: python scripts/verify_dependency_profiles.py lock "
        f"--python {minor[0]}.{minor[1]} --profile {spec.name}\n"
    )


def _write_locks(python: Path, profiles: tuple[str, ...]) -> None:
    minor = _python_minor(python)
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="atagia-lock-") as temp_dir:
        for profile in profiles:
            spec = PROFILE_SPECS[profile]
            compiled = Path(temp_dir) / f"{profile}.txt"
            _compile_profile(python, spec, compiled)
            target = _lock_path(profile, minor)
            target.write_text(
                _lock_header(spec, minor) + compiled.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            _validate_lock(target)


def _check_locks(python: Path, profiles: tuple[str, ...]) -> None:
    minor = _python_minor(python)
    with tempfile.TemporaryDirectory(prefix="atagia-lock-check-") as temp_dir:
        for profile in profiles:
            expected = _lock_path(profile, minor)
            _validate_lock(expected)
            generated = Path(temp_dir) / f"{profile}.txt"
            _compile_profile(
                python,
                PROFILE_SPECS[profile],
                generated,
                constraint=expected,
            )
            if _lock_body(expected) != _lock_body(generated):
                raise RuntimeError(
                    f"Dependency lock is stale: {expected.relative_to(ROOT)}; run the lock command"
                )


def _validate_lock(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing dependency lock: {path.relative_to(ROOT)}")
    pins = [line for line in _lock_body(path).splitlines() if line]
    if not pins:
        raise RuntimeError(f"Dependency lock is empty: {path.relative_to(ROOT)}")
    invalid = [line for line in pins if _PIN_PATTERN.fullmatch(line) is None]
    if invalid:
        raise RuntimeError(f"Dependency lock contains non-exact entries: {invalid}")


def _lock_body(path: Path) -> str:
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _verify_profiles(
    python: Path,
    profiles: tuple[str, ...],
    *,
    artifacts_dir: Path,
    audit: bool,
) -> list[ProfileResult]:
    minor = _python_minor(python)
    if audit and any(profile in PYTHON_PROFILES for profile in profiles):
        _assert_outer_tool_versions(python, minor, ("pip-audit",))
    wheel: Path | None = None
    if any(profile in {"core", "mcp", "dev"} for profile in profiles):
        _assert_outer_tool_versions(
            python,
            minor,
            ("build", "pip", "setuptools", "wheel"),
        )
        wheel = _build_wheel(python, artifacts_dir)
        _record_wheel_metadata(wheel, artifacts_dir)
    results: list[ProfileResult] = []
    for profile in profiles:
        if profile == "node":
            results.extend(_verify_node_artifacts(artifacts_dir, audit=audit))
            continue
        lock = _lock_path(profile, minor)
        _validate_lock(lock)
        results.append(
            _verify_python_profile(
                python,
                profile,
                lock,
                wheel=wheel,
                artifacts_dir=artifacts_dir,
                audit=audit,
            )
        )
    return results


def _build_wheel(python: Path, artifacts_dir: Path) -> Path:
    wheel_dir = artifacts_dir / "wheel"
    if wheel_dir.exists():
        shutil.rmtree(wheel_dir)
    wheel_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="atagia-wheel-source-") as temp_dir:
        build_source = Path(temp_dir) / "source"
        build_source.mkdir()
        for filename in ("LICENSE", "README.md", "pyproject.toml", "setup.py"):
            shutil.copy2(ROOT / filename, build_source / filename)
        shutil.copytree(
            ROOT / "src",
            build_source / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
        _run(
            [
                str(python),
                "-m",
                "build",
                "--quiet",
                "--no-isolation",
                "--wheel",
                "--outdir",
                str(wheel_dir),
                str(build_source),
            ],
            cwd=build_source,
        )
    wheels = sorted(wheel_dir.glob("atagia-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one Atagia wheel, found {len(wheels)}")
    wheel = wheels[0]
    _assert_wheel_payload_matches_source(wheel)
    _write_build_report(wheel, artifacts_dir / "wheel-build-report.json")
    return wheel


def _write_build_report(wheel: Path, report_path: Path) -> None:
    report = {
        "version": "1.0",
        "artifacts": [
            {
                "name": wheel.name,
                "path": str(wheel.resolve()),
                "kind": "wheel",
                "size": wheel.stat().st_size,
                "hashes": {"sha256": _file_sha256(wheel)},
            }
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


def _assert_wheel_payload_matches_source(
    wheel: Path,
    *,
    source_root: Path | None = None,
) -> None:
    _assert_distribution_payload_matches_source(
        wheel,
        artifact_kind="wheel",
        source_root=source_root,
    )


def _assert_sdist_payload_matches_source(
    sdist: Path,
    *,
    source_root: Path | None = None,
) -> None:
    _assert_distribution_payload_matches_source(
        sdist,
        artifact_kind="sdist",
        source_root=source_root,
    )


def _assert_distribution_payload_matches_source(
    artifact: Path,
    *,
    artifact_kind: str,
    source_root: Path | None = None,
) -> None:
    canonical_source_root = source_root or ROOT / "src"
    canonical_project_root = canonical_source_root.parent
    expected = _source_package_payload(canonical_source_root)
    if artifact_kind == "wheel":
        actual = _wheel_package_payload(artifact, canonical_project_root)
    elif artifact_kind == "sdist":
        actual = _sdist_package_payload(artifact, canonical_project_root)
    else:
        raise ValueError(f"Unsupported distribution artifact kind: {artifact_kind}")
    expected_names = set(expected)
    actual_names = set(actual)
    changed = sorted(
        name for name in expected_names & actual_names if expected[name] != actual[name]
    )
    if actual_names != expected_names or changed:
        raise RuntimeError(
            f"Built {artifact_kind} payload differs from src/atagia: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}, "
            f"changed={changed}"
        )


def _source_package_payload(source_root: Path) -> dict[str, str]:
    return {
        path.relative_to(source_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in (source_root / "atagia").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _project_configuration(project_root: Path) -> dict[str, object] | None:
    path = project_root / "pyproject.toml"
    if not path.is_file():
        return None
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _distribution_stem(configuration: dict[str, object]) -> tuple[str, str]:
    project = configuration["project"]
    if not isinstance(project, dict):
        raise RuntimeError("pyproject.toml has no project table")
    name = re.sub(r"[-_.]+", "_", str(project["name"]))
    return name, str(project["version"])


def _normalize_requirement(requirement: str) -> str:
    dependency, separator, marker = requirement.partition(";")
    match = re.fullmatch(
        r"([A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?)(.*)",
        dependency.strip(),
    )
    if match is None:
        raise RuntimeError(f"Unsupported project requirement: {requirement!r}")
    name, specifiers = match.groups()
    ordered_specifiers = sorted(
        item.strip() for item in specifiers.split(",") if item.strip()
    )
    normalized = name + ",".join(ordered_specifiers)
    if separator:
        normalized += "; " + re.sub(r"\s+", " ", marker.strip())
    return normalized


def _expected_entry_points(configuration: dict[str, object]) -> bytes:
    project = configuration["project"]
    scripts = project.get("scripts", {})
    lines = ["[console_scripts]"]
    lines.extend(f"{name} = {scripts[name]}" for name in sorted(scripts))
    return ("\n".join(lines) + "\n").encode()


def _expected_requires_text(configuration: dict[str, object]) -> bytes:
    project = configuration["project"]
    lines = [
        _normalize_requirement(str(requirement))
        for requirement in project.get("dependencies", [])
    ]
    optional = project.get("optional-dependencies", {})
    for extra in sorted(optional):
        lines.extend(
            [
                "",
                f"[{extra}]",
                *(
                    _normalize_requirement(str(requirement))
                    for requirement in optional[extra]
                ),
            ]
        )
    return ("\n".join(lines) + "\n").encode()


def _expected_requires_dist(configuration: dict[str, object]) -> list[str]:
    project = configuration["project"]
    requirements = [
        _normalize_requirement(str(requirement))
        for requirement in project.get("dependencies", [])
    ]
    for extra, dependencies in project.get("optional-dependencies", {}).items():
        requirements.extend(
            f'{_normalize_requirement(str(requirement))}; extra == "{extra}"'
            for requirement in dependencies
        )
    return requirements


def _validate_project_metadata(
    content: bytes,
    *,
    configuration: dict[str, object],
    project_root: Path,
    label: str,
) -> None:
    message = BytesParser(policy=policy.default).parsebytes(content)
    project = configuration["project"]
    allowed_headers = {
        "Classifier",
        "Description-Content-Type",
        "Dynamic",
        "License-Expression",
        "License-File",
        "Metadata-Version",
        "Name",
        "Provides-Extra",
        "Requires-Dist",
        "Requires-Python",
        "Summary",
        "Version",
    }
    unexpected_headers = set(message.keys()) - allowed_headers
    if unexpected_headers:
        raise RuntimeError(
            f"Built {label} metadata contains unexpected headers: "
            f"{sorted(unexpected_headers)}"
        )
    expected_scalars = {
        "Description-Content-Type": "text/markdown",
        "Metadata-Version": "2.4",
        "Name": str(project["name"]),
        "Version": str(project["version"]),
        "Summary": str(project["description"]),
        "Requires-Python": str(project["requires-python"]),
    }
    license_expression = project.get("license")
    if isinstance(license_expression, str):
        expected_scalars["License-Expression"] = license_expression
    mismatches = {
        name: {"expected": expected, "actual": message.get(name)}
        for name, expected in expected_scalars.items()
        if message.get(name) != expected
    }
    duplicate_scalars = [
        name for name in expected_scalars if len(message.get_all(name, [])) != 1
    ]
    if duplicate_scalars:
        mismatches["duplicate scalar headers"] = duplicate_scalars
    actual_requirements = [
        _normalize_requirement(value) for value in message.get_all("Requires-Dist", [])
    ]
    expected_requirements = _expected_requires_dist(configuration)
    expected_extras = list(project.get("optional-dependencies", {}))
    if actual_requirements != expected_requirements:
        mismatches["Requires-Dist"] = {
            "expected": expected_requirements,
            "actual": actual_requirements,
        }
    if message.get_all("Provides-Extra", []) != expected_extras:
        mismatches["Provides-Extra"] = {
            "expected": expected_extras,
            "actual": message.get_all("Provides-Extra", []),
        }
    if message.get_all("Classifier", []) != list(project.get("classifiers", [])):
        mismatches["Classifier"] = "does not match pyproject.toml"
    if message.get_all("License-File", []) != ["LICENSE"]:
        mismatches["License-File"] = {
            "expected": ["LICENSE"],
            "actual": message.get_all("License-File", []),
        }
    if message.get_all("Dynamic", []) != ["license-file"]:
        mismatches["Dynamic"] = {
            "expected": ["license-file"],
            "actual": message.get_all("Dynamic", []),
        }
    _headers, separator, description = content.partition(b"\n\n")
    if not separator or description != (project_root / "README.md").read_bytes():
        mismatches["Description"] = "does not match README.md"
    if mismatches:
        raise RuntimeError(f"Built {label} metadata differs from project: {mismatches}")


def _validate_member_tree(
    names: set[str],
    file_names: set[str],
    *,
    artifact_kind: str,
) -> None:
    for name in names:
        for parent in PurePosixPath(name).parents:
            rendered = parent.as_posix()
            if rendered == ".":
                break
            if rendered in file_names:
                raise RuntimeError(
                    f"Built {artifact_kind} contains a file/ancestor collision: {name}"
                )
    for directory in names - file_names:
        prefix = f"{directory}/"
        if not any(name.startswith(prefix) for name in names):
            raise RuntimeError(
                f"Built {artifact_kind} contains an unexpected empty directory: {directory}"
            )


def _validate_explicit_directory_tree(names: set[str], file_names: set[str]) -> None:
    expected_directories: set[str] = set()
    for name in file_names:
        expected_directories.update(
            parent.as_posix()
            for parent in PurePosixPath(name).parents
            if parent.as_posix() != "."
        )
    actual_directories = names - file_names
    if actual_directories != expected_directories:
        raise RuntimeError(
            "Built sdist directory inventory differs from member parent closure: "
            f"missing={sorted(expected_directories - actual_directories)}, "
            f"unexpected={sorted(actual_directories - expected_directories)}"
        )


def _validate_wheel_record(
    content: bytes,
    *,
    files: dict[str, bytes],
    record_name: str,
) -> None:
    rows = list(csv.reader(StringIO(content.decode("utf-8"), newline="")))
    if any(len(row) != 3 for row in rows):
        raise RuntimeError("Built wheel RECORD contains a malformed row")
    recorded: dict[str, tuple[str, str]] = {}
    for name, digest, size in rows:
        if name in recorded:
            raise RuntimeError(f"Built wheel RECORD repeats a member: {name}")
        recorded[name] = (digest, size)
    if set(recorded) != set(files):
        raise RuntimeError("Built wheel RECORD inventory differs from archive members")
    for name, data in files.items():
        digest, size = recorded[name]
        if name == record_name:
            if digest or size:
                raise RuntimeError(
                    "Built wheel RECORD self-row must omit hash and size"
                )
            continue
        expected_digest = base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()
        ).rstrip(b"=")
        if digest != f"sha256={expected_digest.decode()}" or size != str(len(data)):
            raise RuntimeError(f"Built wheel RECORD integrity mismatch for {name}")


def _validate_wheel_metadata(
    files: dict[str, bytes],
    *,
    dist_info_root: str,
    configuration: dict[str, object],
    project_root: Path,
) -> None:
    expected_members = {
        f"{dist_info_root}/licenses/LICENSE",
        f"{dist_info_root}/METADATA",
        f"{dist_info_root}/WHEEL",
        f"{dist_info_root}/entry_points.txt",
        f"{dist_info_root}/top_level.txt",
        f"{dist_info_root}/RECORD",
    }
    actual_members = {name for name in files if name.startswith(f"{dist_info_root}/")}
    if actual_members != expected_members:
        raise RuntimeError(
            "Built wheel dist-info inventory differs from the canonical manifest: "
            f"missing={sorted(expected_members - actual_members)}, "
            f"unexpected={sorted(actual_members - expected_members)}"
        )
    license_name = f"{dist_info_root}/licenses/LICENSE"
    if files[license_name] != (project_root / "LICENSE").read_bytes():
        raise RuntimeError("Built wheel license differs from LICENSE")
    metadata_name = f"{dist_info_root}/METADATA"
    _validate_project_metadata(
        files[metadata_name],
        configuration=configuration,
        project_root=project_root,
        label="wheel",
    )
    if files[f"{dist_info_root}/entry_points.txt"] != _expected_entry_points(
        configuration
    ):
        raise RuntimeError("Built wheel entry points differ from pyproject.toml")
    if files[f"{dist_info_root}/top_level.txt"] != b"atagia\n":
        raise RuntimeError("Built wheel top-level package metadata is invalid")
    wheel_metadata = BytesParser(policy=policy.default).parsebytes(
        files[f"{dist_info_root}/WHEEL"]
    )
    if set(wheel_metadata.keys()) != {
        "Generator",
        "Root-Is-Purelib",
        "Tag",
        "Wheel-Version",
    } or any(
        len(wheel_metadata.get_all(name, [])) != 1
        for name in ("Generator", "Root-Is-Purelib", "Wheel-Version")
    ):
        raise RuntimeError("Built wheel compatibility metadata has invalid headers")
    if (
        wheel_metadata.get("Wheel-Version") != "1.0"
        or wheel_metadata.get("Root-Is-Purelib") != "true"
        or wheel_metadata.get_all("Tag", []) != ["py3-none-any"]
    ):
        raise RuntimeError("Built wheel compatibility metadata is invalid")
    record_name = f"{dist_info_root}/RECORD"
    _validate_wheel_record(files[record_name], files=files, record_name=record_name)


def _wheel_package_payload(
    wheel: Path,
    project_root: Path,
) -> dict[str, str]:
    configuration = _project_configuration(project_root)
    with ZipFile(wheel) as archive:
        names: set[str] = set()
        portable_names: dict[str, str] = {}
        file_names: set[str] = set()
        files: dict[str, bytes] = {}
        for member in archive.infolist():
            _validate_archive_member_name(member.filename, artifact_kind="wheel")
            name = member.filename.removesuffix("/")
            if name in names:
                raise RuntimeError(f"Built wheel contains a duplicate member: {name}")
            portable_name = _portable_archive_name(name)
            if portable_name in portable_names:
                raise RuntimeError(
                    "Built wheel contains a portable-name collision: "
                    f"{portable_names[portable_name]!r}, {name!r}"
                )
            portable_names[portable_name] = name
            names.add(name)
            _validate_wheel_member_type(member)
            if member.is_dir():
                raise RuntimeError(
                    f"Built wheel contains an explicit directory member: {name}"
                )
            file_names.add(name)
            files[name] = archive.read(member)
        _validate_member_tree(names, file_names, artifact_kind="wheel")

    roots = {name.split("/", 1)[0] for name in names}
    dist_info_roots = {root for root in roots if root.endswith(".dist-info")}
    if configuration is not None:
        distribution, version = _distribution_stem(configuration)
        expected_dist_info = f"{distribution}-{version}.dist-info"
        expected_filename = f"{distribution}-{version}-py3-none-any.whl"
        if wheel.name != expected_filename:
            raise RuntimeError(
                f"Built wheel filename {wheel.name!r} does not match {expected_filename!r}"
            )
        if roots != {"atagia", expected_dist_info}:
            raise RuntimeError(
                f"Built wheel contains unexpected top-level payload: {sorted(roots)}"
            )
        _validate_wheel_metadata(
            files,
            dist_info_root=expected_dist_info,
            configuration=configuration,
            project_root=project_root,
        )
    elif roots - {"atagia"} - dist_info_roots or len(dist_info_roots) > 1:
        raise RuntimeError(
            f"Built wheel contains unexpected top-level payload: {sorted(roots)}"
        )

    return {
        name: hashlib.sha256(content).hexdigest()
        for name, content in files.items()
        if name.startswith("atagia/")
    }


def _validate_sdist_manifest(
    files: dict[str, bytes],
    *,
    configuration: dict[str, object],
    project_root: Path,
) -> None:
    expected_source_files = set(SDIST_CANONICAL_ROOT_FILES)
    expected_source_files.update(
        name for name in SDIST_CANONICAL_TEST_FILES if (project_root / name).is_file()
    )
    missing_source = expected_source_files - set(files)
    missing_generated = set(SDIST_GENERATED_FILES) - set(files)
    if missing_source or missing_generated:
        raise RuntimeError(
            "Built sdist canonical manifest is incomplete: "
            f"missing_source={sorted(missing_source)}, "
            f"missing_generated={sorted(missing_generated)}"
        )
    for name, content in files.items():
        source_path = project_root / name
        if name in expected_source_files:
            if not source_path.is_file() or content != source_path.read_bytes():
                raise RuntimeError(
                    f"Built sdist source member differs from checkout: {name}"
                )
        elif name.startswith("src/atagia/") or name in SDIST_GENERATED_FILES:
            continue
        else:
            raise RuntimeError(
                f"Built sdist contains unexpected source payload: {name}"
            )

    if files["setup.cfg"] != SDIST_SETUP_CFG:
        raise RuntimeError("Built sdist generated setup.cfg is invalid")
    if files["PKG-INFO"] != files["src/atagia.egg-info/PKG-INFO"]:
        raise RuntimeError("Built sdist PKG-INFO copies differ")
    _validate_project_metadata(
        files["PKG-INFO"],
        configuration=configuration,
        project_root=project_root,
        label="sdist",
    )
    if files["src/atagia.egg-info/entry_points.txt"] != _expected_entry_points(
        configuration
    ):
        raise RuntimeError("Built sdist entry points differ from pyproject.toml")
    if files["src/atagia.egg-info/requires.txt"] != _expected_requires_text(
        configuration
    ):
        raise RuntimeError("Built sdist requirements differ from pyproject.toml")
    if files["src/atagia.egg-info/top_level.txt"] != b"atagia\n":
        raise RuntimeError("Built sdist top-level package metadata is invalid")
    if files["src/atagia.egg-info/dependency_links.txt"].strip():
        raise RuntimeError("Built sdist contains unexpected dependency links")
    sources_content = files["src/atagia.egg-info/SOURCES.txt"]
    if b"\r" in sources_content or sources_content.endswith(b"\n"):
        raise RuntimeError("Built sdist SOURCES.txt must use canonical LF text")
    source_rows = sources_content.decode("utf-8").splitlines()
    if not source_rows or any(not row for row in source_rows):
        raise RuntimeError("Built sdist SOURCES.txt contains an empty row")
    for row in source_rows:
        _validate_archive_member_name(row, artifact_kind="sdist SOURCES.txt")
    if len(source_rows) != len(set(source_rows)):
        raise RuntimeError("Built sdist SOURCES.txt contains duplicates")
    expected_rows = set(files) - {"PKG-INFO", "setup.cfg"}
    if set(source_rows) != expected_rows:
        raise RuntimeError("Built sdist SOURCES.txt differs from archive inventory")


def _sdist_package_payload(
    sdist: Path,
    project_root: Path,
) -> dict[str, str]:
    configuration = _project_configuration(project_root)
    with tarfile.open(sdist, mode="r:gz") as archive:
        names: set[str] = set()
        portable_names: dict[str, str] = {}
        file_names: set[str] = set()
        roots: set[str] = set()
        archived_files: dict[str, bytes] = {}
        for member in archive.getmembers():
            _validate_archive_member_name(member.name, artifact_kind="sdist")
            if member.name.endswith("/"):
                raise RuntimeError(
                    f"Built sdist contains a trailing-slash member: {member.name}"
                )
            name = member.name.removesuffix("/")
            if name in names:
                raise RuntimeError(f"Built sdist contains a duplicate member: {name}")
            portable_name = _portable_archive_name(name)
            if portable_name in portable_names:
                raise RuntimeError(
                    "Built sdist contains a portable-name collision: "
                    f"{portable_names[portable_name]!r}, {name!r}"
                )
            portable_names[portable_name] = name
            names.add(name)
            root, separator, _relative_name = name.partition("/")
            roots.add(root)
            if not separator and member.type != tarfile.DIRTYPE:
                raise RuntimeError(
                    "Built sdist contains a root-level file outside its root"
                )
            if member.type == tarfile.DIRTYPE:
                if member.mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                    raise RuntimeError(
                        f"Built sdist contains unsafe mode bits: {member.name}"
                    )
                continue
            if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                raise RuntimeError(
                    f"Built sdist contains a link or special member: {member.name}"
                )
            if member.mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                raise RuntimeError(
                    f"Built sdist contains unsafe mode bits: {member.name}"
                )
            file_names.add(name)
            member_file = archive.extractfile(member)
            if member_file is None:
                raise RuntimeError(f"Unable to read sdist member: {member.name}")
            archived_files[name] = member_file.read()
        if len(roots) != 1:
            raise RuntimeError("Built sdist package payload has multiple roots")
        _validate_member_tree(names, file_names, artifact_kind="sdist")
        _validate_explicit_directory_tree(names, file_names)

    root = next(iter(roots))
    if configuration is not None:
        distribution, version = _distribution_stem(configuration)
        expected_root = f"{distribution}-{version}"
        expected_filename = f"{expected_root}.tar.gz"
        if sdist.name != expected_filename:
            raise RuntimeError(
                f"Built sdist filename {sdist.name!r} does not match "
                f"{expected_filename!r}"
            )
        if root != expected_root:
            raise RuntimeError(
                f"Built sdist root {root!r} does not match {expected_root!r}"
            )
    relative_files = {
        name.removeprefix(f"{root}/"): content
        for name, content in archived_files.items()
    }
    if configuration is not None:
        _validate_sdist_manifest(
            relative_files,
            configuration=configuration,
            project_root=project_root,
        )
    else:
        unexpected = [
            name for name in relative_files if not name.startswith("src/atagia/")
        ]
        if unexpected:
            raise RuntimeError(
                f"Built sdist contains unexpected source payload: {sorted(unexpected)}"
            )
    return {
        name.removeprefix("src/"): hashlib.sha256(content).hexdigest()
        for name, content in relative_files.items()
        if name.startswith("src/atagia/")
    }


def _validate_archive_member_name(name: str, *, artifact_kind: str) -> None:
    rendered = name[:-1] if name.endswith("/") else name
    parts = rendered.split("/")
    windows_reserved = {"CON", "PRN", "AUX", "NUL"}
    windows_reserved.update(f"COM{index}" for index in range(1, 10))
    windows_reserved.update(f"LPT{index}" for index in range(1, 10))
    if (
        not rendered
        or rendered.startswith("/")
        or "\\" in rendered
        or "\x00" in rendered
        or any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(rendered).as_posix() != rendered
        or bool(PureWindowsPath(rendered).drive)
        or bool(PureWindowsPath(rendered).root)
        or any(
            ":" in part
            or part.rstrip(" .") != part
            or part.split(".", 1)[0].upper() in windows_reserved
            for part in parts
        )
    ):
        raise RuntimeError(
            f"Built {artifact_kind} contains an unsafe member name: {name!r}"
        )


def _portable_archive_name(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _validate_wheel_member_type(member: ZipInfo) -> None:
    unix_mode = (member.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    allowed_type = stat.S_IFDIR if member.is_dir() else stat.S_IFREG
    if file_type not in {0, allowed_type}:
        raise RuntimeError(
            f"Built wheel contains a link or special member: {member.filename}"
        )
    if unix_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        raise RuntimeError(f"Built wheel contains unsafe mode bits: {member.filename}")


def _record_wheel_metadata(wheel: Path, artifacts_dir: Path) -> None:
    with ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError("Built wheel has no unique METADATA file")
        metadata = archive.read(metadata_names[0])
    (artifacts_dir / "wheel-METADATA.txt").write_bytes(metadata)


def _assert_outer_tool_versions(
    python: Path,
    minor: tuple[int, int],
    package_names: tuple[str, ...],
) -> None:
    expected = _locked_versions(LOCKS_DIR / "bootstrap.txt")
    expected.update(_locked_versions(_lock_path("dev", minor)))
    requested = {_normalize_package_name(name) for name in package_names}
    query = (
        "import importlib.metadata, json; "
        f"names = {sorted(requested)!r}; "
        "print(json.dumps({name: importlib.metadata.version(name) for name in names}))"
    )
    installed = json.loads(_capture([str(python), "-c", query]))
    mismatches = {
        name: {"expected": expected.get(name), "installed": installed.get(name)}
        for name in requested
        if expected.get(name) != installed.get(name)
    }
    if mismatches:
        raise RuntimeError(
            "Dependency gate tooling does not match the tested locks: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def _verify_python_profile(
    python: Path,
    profile: str,
    lock: Path,
    *,
    wheel: Path | None,
    artifacts_dir: Path,
    audit: bool,
) -> ProfileResult:
    with tempfile.TemporaryDirectory(prefix=f"atagia-{profile}-") as temp_dir:
        venv = Path(temp_dir) / "venv"
        _run([str(python), "-m", "venv", str(venv)])
        venv_python = _venv_python(venv)
        _run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--requirement",
                str(LOCKS_DIR / "bootstrap.txt"),
            ]
        )
        if profile == "blog":
            install_target = ["--requirement", str(lock)]
        else:
            if wheel is None:
                raise RuntimeError("Wheel is required for an Atagia runtime profile")
            extras = PROFILE_SPECS[profile].wheel_extras
            target = str(wheel)
            if extras:
                target += f"[{','.join(extras)}]"
            install_target = ["--constraint", str(lock), target]
        _run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                *install_target,
            ]
        )
        _run([str(venv_python), "-m", "pip", "check"])
        inventory = _capture([str(venv_python), "-m", "pip", "list", "--format=json"])
        inventory_path = (
            artifacts_dir / f"{profile}-{_minor_label(python)}-inventory.json"
        )
        inventory_path.write_text(inventory, encoding="utf-8")
        _assert_locked_versions(inventory, lock)
        bootstrap_lock = LOCKS_DIR / "bootstrap.txt"
        _assert_locked_versions(inventory, bootstrap_lock)
        _assert_no_unlocked_packages(
            inventory,
            locks=(lock, bootstrap_lock),
            allowed=("atagia",) if profile != "blog" else (),
        )
        smoke_status, artifact_evidence = _run_python_smoke(
            venv_python,
            profile,
            artifacts_dir=artifacts_dir,
        )
        audit_status = "not_requested"
        audit_path: Path | None = None
        if audit:
            audit_path = artifacts_dir / f"{profile}-{_minor_label(python)}-audit.json"
            _run(
                [
                    str(python),
                    "-m",
                    "pip_audit",
                    "--strict",
                    "--progress-spinner=off",
                    "--vulnerability-service=osv",
                    "--format=json",
                    "--output",
                    str(audit_path),
                    "--requirement",
                    str(lock),
                    "--no-deps",
                    "--disable-pip",
                ]
            )
            audit_status = "passed"
        return ProfileResult(
            schema=DEPENDENCY_PROFILE_SCHEMA,
            profile=profile,
            python=_minor_label(python),
            profile_source=str(PROFILE_SPECS[profile].source.relative_to(ROOT)),
            profile_source_sha256=_file_sha256(PROFILE_SPECS[profile].source),
            lock_file=str(lock.relative_to(ROOT)),
            lock_sha256=_file_sha256(lock),
            inventory_file=_artifact_path_label(inventory_path),
            inventory_sha256=_file_sha256(inventory_path),
            audit=audit_status,
            audit_file=(
                _artifact_path_label(audit_path) if audit_path is not None else None
            ),
            audit_sha256=(_file_sha256(audit_path) if audit_path is not None else None),
            smoke=smoke_status,
            artifact_evidence=artifact_evidence,
        )


def _assert_locked_versions(inventory_json: str, lock: Path) -> None:
    installed = {
        _normalize_package_name(str(item["name"])): str(item["version"])
        for item in json.loads(inventory_json)
    }
    for line in _lock_body(lock).splitlines():
        requirement = line.split(";", 1)[0].strip()
        name, version = requirement.split("==", 1)
        normalized = _normalize_package_name(name)
        actual = installed.get(normalized)
        if actual != version:
            raise RuntimeError(
                f"Installed {name} version {actual!r} does not match lock {version!r}"
            )


def _locked_versions(lock: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in _lock_body(lock).splitlines():
        requirement = line.split(";", 1)[0].strip()
        name, version = requirement.split("==", 1)
        versions[_normalize_package_name(name)] = version
    return versions


def _assert_no_unlocked_packages(
    inventory_json: str,
    *,
    locks: tuple[Path, ...],
    allowed: tuple[str, ...],
) -> None:
    installed = {
        _normalize_package_name(str(item["name"]))
        for item in json.loads(inventory_json)
    }
    expected = {_normalize_package_name(name) for name in allowed}
    for lock in locks:
        expected.update(_locked_versions(lock))
    unexpected = sorted(installed - expected)
    if unexpected:
        raise RuntimeError(
            f"Profile contains packages absent from its locks: {unexpected}"
        )


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _run_python_smoke(
    python: Path,
    profile: str,
    *,
    artifacts_dir: Path,
) -> tuple[str, dict[str, object] | None]:
    if profile == "blog":
        return _run_blog_artifact_smoke(python, artifacts_dir=artifacts_dir)
    smoke = (
        "import asyncio\n"
        "from atagia.core.config import default_resource_path\n"
        "from atagia.core.db_sqlite import close_connection, initialize_database\n"
        "async def smoke():\n"
        "    connection = await initialize_database(\n"
        "        ':memory:', default_resource_path('migrations')\n"
        "    )\n"
        "    await close_connection(connection)\n"
        "asyncio.run(smoke())\n"
    )
    if profile in {"mcp", "dev"}:
        smoke += "import atagia.mcp_server, mcp\n"
    if profile == "dev":
        smoke += "import pip_audit, pytest, ruff, sqlite_vec\n"
    _run([str(python), "-c", smoke], cwd=ROOT)
    return "passed", None


def _run_blog_artifact_smoke(
    python: Path,
    *,
    artifacts_dir: Path,
) -> tuple[str, dict[str, object] | None]:
    missing = [
        path
        for relative in BLOG_ARTIFACT_PATHS
        if not (path := BLOG_ENGINE_ROOT / relative).exists()
    ]
    if missing:
        if os.getenv("ATAGIA_ALLOW_MISSING_PRIVATE_BLOG_ARTIFACT") != "1":
            rendered = ", ".join(str(path.relative_to(ROOT)) for path in missing)
            raise RuntimeError(f"Private blog artifact is unavailable: {rendered}")
        dependency_smoke = (
            "from markdown_it import MarkdownIt\n"
            "import nh3, yaml\n"
            "rendered = MarkdownIt('commonmark', {'html': False}).render('# title\\n<script>x</script>')\n"
            "assert '<script>' not in nh3.clean(rendered)\n"
            "assert yaml.safe_load('title: safe')['title'] == 'safe'\n"
        )
        _run([str(python), "-c", dependency_smoke], cwd=ROOT)
        return "dependency-only-private-artifact-unavailable", None

    source_manifest = _blog_source_manifest(BLOG_ENGINE_ROOT)
    source_digest = _canonical_sha256(source_manifest)
    source_manifest_with_digest = {
        **source_manifest,
        "source_digest": source_digest,
    }
    label = _minor_label(python)
    source_manifest_path = artifacts_dir / f"blog-{label}-source-manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest_with_digest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory(prefix="atagia-blog-artifact-") as temp_dir:
        workspace = Path(temp_dir) / "blog"
        workspace.mkdir()
        for file_record in source_manifest["files"]:
            relative = Path(str(file_record["path"]))
            source = BLOG_ENGINE_ROOT / relative
            destination = workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        releases = Path(temp_dir) / "releases"
        public_parent = Path(temp_dir) / "public"
        public_parent.mkdir()
        release = releases / "release-profile-smoke"
        _run(
            [
                str(python),
                str(workspace / "build.py"),
                "--release-id",
                "profile-smoke",
                "--releases-root",
                str(releases),
                "--json",
            ],
            cwd=workspace,
        )
        _run(
            [
                str(python),
                str(workspace / "build.py"),
                "--validate-release",
                str(release),
                "--json",
            ],
            cwd=workspace,
        )
        _run(
            [
                str(python),
                str(workspace / "publication.py"),
                "--releases-root",
                str(releases),
                "--public-entrypoint",
                str(public_parent / "blog"),
                "status",
                "--allow-missing",
            ],
            cwd=workspace,
        )
        if (workspace / "run_pipeline.py").is_file():
            _run(
                [str(python), str(workspace / "run_pipeline.py"), "--status"],
                cwd=workspace,
            )
        manifest_path = release / ".atagia-release.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        payload = _load_blog_provenance().release_payload(manifest)
        release_files = _tree_file_records(release)
        routes = list(manifest["routes"])
        provenance = _load_blog_provenance()
        evidence: dict[str, object] = {
            "schema": provenance.BLOG_DEPENDENCY_EVIDENCE_SCHEMA,
            "source_manifest": source_manifest,
            "source_manifest_file": _artifact_path_label(source_manifest_path),
            "source_digest": source_digest,
            "release_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "release_payload_digest": _canonical_sha256(payload),
            "release_root_digest": _canonical_sha256(
                {"schema": "atagia.blog-release-root.v1", "files": release_files}
            ),
            "route_count": len(routes),
            "post_route_count": sum(
                1
                for route in routes
                if route.startswith("/blog/")
                and route.endswith("/")
                and route != "/blog/"
                and not route.startswith(("/blog/category/", "/blog/tag/"))
            ),
            "file_count": len(release_files),
        }
        evidence_path = artifacts_dir / f"blog-{label}-release-evidence.json"
        evidence["evidence_file"] = _artifact_path_label(evidence_path)
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if _canonical_sha256(_blog_source_manifest(BLOG_ENGINE_ROOT)) != source_digest:
        raise RuntimeError("Private blog source changed while its profile was verified")
    return "passed", evidence


def _blog_source_manifest(root: Path) -> dict[str, object]:
    return _load_blog_provenance().build_source_manifest(root)


def _load_blog_provenance():
    path = BLOG_ENGINE_ROOT / "release_provenance.py"
    if not path.is_file():
        raise RuntimeError(f"Private blog provenance module is unavailable: {path}")
    spec = importlib.util.spec_from_file_location(
        "_atagia_blog_release_provenance", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Private blog provenance module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree_file_records(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if not path.is_file() or path.is_symlink():
            continue
        content = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_size": len(content),
            }
        )
    return records


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _verify_node_artifacts(
    artifacts_dir: Path,
    *,
    audit: bool,
) -> list[ProfileResult]:
    results: list[ProfileResult] = []
    for artifact in NODE_ARTIFACTS:
        lock = artifact / "package-lock.json"
        if not lock.is_file():
            raise RuntimeError(f"Missing Node lockfile: {lock.relative_to(ROOT)}")
        with tempfile.TemporaryDirectory(
            prefix=f"atagia-node-{artifact.name}-"
        ) as temp_dir:
            workspace = Path(temp_dir) / artifact.name
            shutil.copytree(
                artifact,
                workspace,
                ignore=shutil.ignore_patterns("node_modules"),
            )
            _run(
                ["npm", "ci", "--ignore-scripts", "--engine-strict"],
                cwd=workspace,
            )
            audit_path: Path | None = None
            if audit:
                audit_path = artifacts_dir / f"node-{artifact.name}-audit.json"
                audit_path.write_text(
                    _capture(
                        ["npm", "audit", "--audit-level=low", "--json"],
                        cwd=workspace,
                    ),
                    encoding="utf-8",
                )
            _run(["npm", "test"], cwd=workspace)
            inventory_path = artifacts_dir / f"node-{artifact.name}-inventory.json"
            inventory_path.write_text(
                _capture(["npm", "ls", "--all", "--json"], cwd=workspace),
                encoding="utf-8",
            )
        results.append(
            ProfileResult(
                schema=DEPENDENCY_PROFILE_SCHEMA,
                profile=f"node:{artifact.relative_to(ROOT)}",
                python="n/a",
                profile_source=str((artifact / "package.json").relative_to(ROOT)),
                profile_source_sha256=_file_sha256(artifact / "package.json"),
                lock_file=str(lock.relative_to(ROOT)),
                lock_sha256=_file_sha256(lock),
                inventory_file=_artifact_path_label(inventory_path),
                inventory_sha256=_file_sha256(inventory_path),
                audit="passed" if audit else "not_requested",
                audit_file=(
                    _artifact_path_label(audit_path) if audit_path is not None else None
                ),
                audit_sha256=(
                    _file_sha256(audit_path) if audit_path is not None else None
                ),
                smoke="passed",
            )
        )
    return results


def _minor_label(python: Path) -> str:
    major, minor = _python_minor(python)
    return f"py{major}{minor}"


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _capture(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


if __name__ == "__main__":
    raise SystemExit(main())
