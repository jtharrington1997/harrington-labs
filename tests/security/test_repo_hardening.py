from __future__ import annotations

import subprocess
from pathlib import Path

FORBIDDEN_TRACKED_PATTERNS = [
    ".env",
    ".streamlit/secrets.toml",
    "data/manual/secrets.json",
    "data/manual/access.json",
]
REQUIRED_GITIGNORE_ENTRIES = [
    ".streamlit/secrets.toml",
    "data/manual/access.json",
]
FORBIDDEN_SUFFIXES = (".pem", ".key")


def _tracked_files() -> list[str]:
    proc = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def test_forbidden_secret_files_are_not_tracked() -> None:
    tracked = _tracked_files()
    for path in tracked:
        assert path not in FORBIDDEN_TRACKED_PATTERNS, f"tracked secret-like file: {path}"
        assert not path.endswith(FORBIDDEN_SUFFIXES), f"tracked key-like file: {path}"


def test_gitignore_covers_secret_entrypoints() -> None:
    gitignore = Path(".gitignore").read_text()
    for entry in REQUIRED_GITIGNORE_ENTRIES:
        assert entry in gitignore, f"missing .gitignore entry: {entry}"
