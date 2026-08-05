"""Compatibility launcher for the restored custom GenDeg runner bytecode."""

import importlib.util
from pathlib import Path


_compiled = Path(__file__).with_suffix(".pyc")
_spec = importlib.util.spec_from_file_location("__main__", _compiled)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load restored runner: {_compiled}")
_module = importlib.util.module_from_spec(_spec)
_module.__file__ = str(_compiled)
_spec.loader.exec_module(_module)

