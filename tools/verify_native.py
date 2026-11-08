"""Native build, compiler checks and isolated Python/JS contract suite."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from build_native import build

root = Path(__file__).resolve().parents[1]
build()
cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
for command in (
    [cargo, "fmt", "--check"],
    [cargo, "clippy", "--locked", "--all-targets", "--", "-D", "warnings"],
    [cargo, "test", "--locked"],
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
):
    subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "REMOTEWORKER_CONFIG": "", "PYTHONPATH": str(root)},
        check=True,
    )
