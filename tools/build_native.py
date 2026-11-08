"""Build and atomically install the native module into this checkout."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build(*, release: bool = True) -> Path:
    root = Path(__file__).resolve().parents[1]
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
    target = root / "target"
    target.mkdir(exist_ok=True)
    with (target / "build-native.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        command = [cargo, "build", "--locked", "--features", "extension-module"]
        if release:
            command.append("--release")
        subprocess.run(
            command,
            cwd=root,
            env={**os.environ, "PYO3_PYTHON": sys.executable},
            check=True,
        )
        source = target / ("release" if release else "debug") / "lib_native.so"
        destination = root / "remoteworker" / "_native.abi3.so"
        descriptor, temporary = tempfile.mkstemp(prefix=".native-", suffix=".so", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream, source.open("rb") as binary:
                shutil.copyfileobj(binary, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from remoteworker import _native; assert _native.ARCHITECTURE == 'modular-monolith'",
            ],
            cwd=root,
            check=True,
        )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(build(release=not args.debug))
