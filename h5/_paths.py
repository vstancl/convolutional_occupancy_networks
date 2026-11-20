"""Makes csgbox and dexio importable for h5 code, without adding them as dependencies.

Importing this module puts the two source trees on ``sys.path``. Nothing is installed,
and ``pyproject.toml`` and ``uv.lock`` of ConvONet are not changed. Import it before
any ``csgbox`` or ``dexio`` import in h5 code.

Both locations are the local checkouts of the user's machine; they are read-only for h5 code.
"""

from __future__ import annotations

import sys
from pathlib import Path

CSGBOX_ROOT: Path = Path.home() / "PhD" / "csgbox"  # csgbox package lives in CSGBOX_ROOT / csgbox
DEXIO_DIR: Path = Path.home() / "PhD" / "dexelize" / "python"  # dexio.py lives here

for _p in (CSGBOX_ROOT, DEXIO_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
