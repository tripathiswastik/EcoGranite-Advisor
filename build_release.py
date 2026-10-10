"""
Release Packaging Tool for EcoGranite-Advisor.

Packages a clean release distribution zip file strictly excluding:
- .git / repository metadata
- .env / secrets (except .env.example)
- __pycache__ / compiled python bytecode (*.pyc, *.pyo)
- .pytest_cache, .mypy_cache, .ruff_cache
- .venv / virtual environments
- release artifacts / dist / build
"""

import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist"
RELEASE_ZIP_NAME = "EcoGranite-Advisor-v4.2.zip"

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "ENV",
    "env",
    "dist",
    "build",
    ".vscode",
    ".idea",
    "scratch",
}

EXCLUDED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".zip",
    ".swp",
    ".swo",
}


def is_excluded(rel_path: Path) -> bool:
    parts = rel_path.parts
    # Exclude any folder in EXCLUDED_DIRS
    for part in parts:
        if part in EXCLUDED_DIRS:
            return True

    filename = rel_path.name
    # Exclude .env files except .env.example
    if filename == ".env" or (filename.startswith(".env.") and filename != ".env.example") or filename.startswith("_env"):
        return True

    # Exclude bytecode and zip files
    for suffix in EXCLUDED_FILE_SUFFIXES:
        if filename.endswith(suffix):
            return True

    return False


def build_release_zip() -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DIST_DIR / RELEASE_ZIP_NAME

    print(f"Building clean release archive: {zip_path.name}...")
    included_files = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)
            rel_dir = root_path.relative_to(PROJECT_ROOT)

            # Skip excluded directories
            if any(part in EXCLUDED_DIRS for part in rel_dir.parts):
                continue

            for file in files:
                file_path = root_path / file
                rel_file = file_path.relative_to(PROJECT_ROOT)

                if is_excluded(rel_file):
                    continue

                arcname = f"EcoGranite-Advisor/{rel_file.as_posix()}"
                zf.write(file_path, arcname)
                included_files.append(rel_file.as_posix())

    print(f"Packaged {len(included_files)} files into {zip_path}.")

    # Integrity verification
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for name in namelist:
            parts = name.split("/")
            assert ".git" not in parts, f"Security violation: .git found in {name}"
            assert "__pycache__" not in name, f"Packaging error: __pycache__ found in {name}"
            assert not name.endswith(".pyc"), f"Packaging error: bytecode found in {name}"
            base = Path(name).name
            if base.startswith(".env") and base != ".env.example":
                raise AssertionError(f"Security violation: {base} bundled in release package!")
            if base.startswith("_env"):
                raise AssertionError(f"Security violation: {base} bundled in release package!")

    print("Verification PASSED: Archive is free of .git, bytecode, and unapproved .env secrets.")
    return zip_path


if __name__ == "__main__":
    build_release_zip()
