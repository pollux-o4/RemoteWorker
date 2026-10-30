"""Load a packaged native extension when the checkout has no local build."""
from importlib import metadata, util
from pathlib import Path
import sys

try:
    distribution = metadata.distribution("remoteworker-native")
except metadata.PackageNotFoundError as error:
    raise ImportError(
        "RemoteWorker Rust extension missing: run this runtime's Python with install.py, "
        "or install its remoteworker-native wheel."
    ) from error

candidates = [distribution.locate_file(file) for file in distribution.files or ()
              if file.parts[:1] == ("remoteworker",) and file.name.startswith("_native.")
              and file.name.endswith((".so", ".pyd"))]
if len(candidates) != 1:
    raise ImportError("RemoteWorker wheel does not contain exactly one native extension")
location = Path(candidates[0])
specification = util.spec_from_file_location(__name__, location)
if specification is None or specification.loader is None:
    raise ImportError(f"RemoteWorker native extension cannot be loaded: {location}")
module = util.module_from_spec(specification)
specification.loader.exec_module(module)
sys.modules[__name__] = module
