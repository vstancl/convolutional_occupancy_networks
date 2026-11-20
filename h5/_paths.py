"""Makes csgbox and dexio importable for h5 code, without adding them as dependencies.

``add_external_paths()`` puts the two source trees on ``sys.path``, imports the modules
that h5 code needs, and checks that no mesh library was pulled in. Nothing is installed,
and ``pyproject.toml`` and ``uv.lock`` of ConvONet are not changed. Call it before any
``csgbox`` or ``dexio`` import in h5 code.

The default locations are the local checkouts of the user's machine. Set the environment
variables ``CSGBOX_ROOT`` and ``DEXIO_DIR`` to override them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_CSGBOX_ROOT: Path = Path.home() / "PhD" / "csgbox"  # the csgbox package is CSGBOX_ROOT / csgbox
DEFAULT_DEXIO_DIR: Path = Path.home() / "PhD" / "dexelize" / "python"  # dexio.py lives here

# Modules whose presence in sys.modules would mean the decoder path pulled in a mesh library.
FORBIDDEN_MODULES: tuple[str, ...] = ("manifold3d", "manifold")


def add_external_paths() -> str:
    """Put csgbox and dexio on sys.path, import them, and return ``numpy.__version__``.

    Raises ``RuntimeError`` if a mesh library (manifold) is loaded by the import: the decoder
    path must not depend on it.
    """
    csgbox_root = Path(os.environ.get("CSGBOX_ROOT", DEFAULT_CSGBOX_ROOT)).expanduser()
    dexio_dir = Path(os.environ.get("DEXIO_DIR", DEFAULT_DEXIO_DIR)).expanduser()
    for p in (csgbox_root, dexio_dir):
        if not p.is_dir():
            raise RuntimeError(f"external path does not exist: {p}")
        s = str(p)
        if s not in sys.path:
            sys.path.append(s)

    import numpy  # noqa: F401
    import csgbox.ceilings  # noqa: F401
    import dexio  # noqa: F401

    loaded = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    if loaded:
        raise RuntimeError(f"mesh library loaded by the decoder import path: {loaded}")
    return numpy.__version__
