from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_SOURCE = ROOT / "web" / "remote_conditioning.js"
QWEN_WEB_SOURCE = ROOT / "web" / "remote_qwen_image21.js"


def write_widget_inputs_shim(temp: Path) -> None:
    widget_inputs = temp / "extensions" / "core"
    widget_inputs.mkdir(parents=True)
    (widget_inputs / "widgetInputs.js").write_text(
        textwrap.dedent(
            """
            export function setWidgetConfig(input, config) {
              input.widget = { ...(input.widget || {}), config };
            }
            """
        ),
        encoding="utf-8",
    )


def run_web(
    source,
    *,
    app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
    api="export const api = {};",
    args=(),
):
    """Run the shipped ESM with only the Comfy host boundary substituted."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shutil.copytree(ROOT / "web", root / "custom/web")
        (root / "scripts").mkdir()
        (root / "scripts/app.js").write_text(app, encoding="utf-8")
        (root / "scripts/api.js").write_text(api, encoding="utf-8")
        (root / "package.json").write_text('{"type":"module"}', encoding="utf-8")
        write_widget_inputs_shim(root)
        runner = root / "contract.mjs"
        runner.write_text(textwrap.dedent(source), encoding="utf-8")
        return subprocess.run(
            [shutil.which("node"), str(runner), *args],
            check=True,
            capture_output=True,
            text=True,
        )

from remoteworker.minimax import image_to_video as REMOTE_MINIMAX_H3_IMAGE_TO_VIDEO
from remoteworker.minimax import reference as REMOTE_MINIMAX_H3

__all__ = ["run_web", "REMOTE_MINIMAX_H3", "REMOTE_MINIMAX_H3_IMAGE_TO_VIDEO"]
