"""[phd] Run an upstream script with the dexel registration active.

Usage:
    uv run python phd_work/run.py <script.py> [args...]
e.g. uv run python phd_work/run.py train.py phd_work/configs/dexel_3plane_3cls_L32.yaml

Calls phd_work/dexel_register.register(), then runs the script with runpy as __main__,
with sys.argv = [script, *args]. The script's own imports (src.*) are unchanged.
"""
import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# sys.path[0] is phd_work/ when this file is run directly; put the repo root first so the
# upstream packages (src, configs) resolve as they do when run from the repo root.
sys.path[0] = REPO
sys.path.insert(1, os.path.join(REPO, 'phd_work'))

from phd_work.dexel_register import register  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    register()
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
