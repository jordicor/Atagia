"""Setuptools command customizations for clean package builds."""

from __future__ import annotations

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.errors import SetupError


class CleanBuildPy(build_py):
    """Replace the generated package tree instead of merging into stale output."""

    def run(self) -> None:
        project_root = Path(__file__).resolve().parent
        source_root = (project_root / "src").resolve()
        build_command = self.get_finalized_command("build")
        build_base = _resolve_output_path(project_root, Path(build_command.build_base))
        build_lib = _resolve_output_path(project_root, Path(self.build_lib))
        package_output = build_lib / "atagia"
        if not build_lib.is_relative_to(build_base) or build_lib.is_relative_to(
            source_root
        ):
            raise SetupError(
                "Refusing to clean package output outside the generated build base"
            )
        if not self.dry_run:
            if package_output.is_symlink():
                package_output.unlink()
            elif package_output.exists():
                shutil.rmtree(package_output)
        super().run()


def _resolve_output_path(project_root: Path, path: Path) -> Path:
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


setup(cmdclass={"build_py": CleanBuildPy})
