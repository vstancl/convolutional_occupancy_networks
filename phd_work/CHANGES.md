# Changes to upstream files

All changes to files that exist in upstream since the baseline tag `upstream-838bea5`
(`git diff upstream-838bea5 main`). Scope: every tracked upstream file, including root-level scripts.

Behaviour impact:
- **identical**: the same computation by construction (same builtin, same default, or the crash path was unreachable before).
- **parity-verified**: checked by `phd_work/parity.py` against a reference captured from the old code.
- **build or summary only**: changes build artefacts or post-processing tables, not model outputs.

## Modified upstream files

| File | Change | Reason | Commit | Behaviour impact |
| ---- | ------ | ------ | ------ | ---------------- |
| `configs/voxel/shapenet_1plane.yaml`, `configs/voxel/shapenet_3plane.yaml`, `configs/voxel/shapenet_grid32.yaml` | `method: onet` → `method: conv_onet` | Upstream bug: `method_dict` only has `conv_onet`, so the config raised `KeyError: 'onet'` | 9b16c89 | identical (the config could not run before) |
| `src/config.py`, `src/data/core.py` | `yaml.load(f)` → `yaml.load(f, Loader=yaml.Loader)` (3 call sites) | PyYAML ≥ 6 requires `Loader` | a15f931 | identical (`yaml.Loader` is the default PyYAML 3.13 used) |
| `src/utils/binvox_rw.py`, `src/utils/libmesh/inside_mesh.py`, `src/utils/voxels.py` | `np.bool` → `bool`, `np.int` → `int` | NumPy ≥ 1.24 removed the aliases | 95eba60 | identical (the aliases were the builtins themselves) |
| `setup.py` | pykdtree extension built from `kdtree.pyx` instead of `kdtree.c` | Upstream's Cython 0.27 C output does not compile on Python ≥ 3.11 (`longintrepr.h`) | 10e384a | build only |
| `src/utils/libkdtree/pykdtree/kdtree.c` | file removed from git (Cython-generated) | Regenerated from `kdtree.pyx` at build time | 2cf37e0 | build only |
| `src/encoder/voxels.py` | `torch_scatter.scatter_mean` → local `scatter_mean` using `scatter_reduce_(..., reduce='mean', include_self=False)` | Removes `torch_scatter` from the voxel encoder path | ca728fb | parity-verified: forward (CUDA) and gradients (CPU) bit-identical to the old code; GPU backward not verified |
| `generate.py`, `eval_meshes.py` | `.groupby(...).mean()` → `.mean(numeric_only=True)` in summary tables | pandas ≥ 2 raises on string columns | 969976e | build or summary only (post-processing, no model output) |
| `.gitignore` | adds rules for `src/utils/libkdtree/pykdtree/kdtree.c` (generated), `.venv/`, `phd_work/parity/` | Generated and local files | 60da320, 69736aa, 0dda9cb, d4789b2 | none (repository hygiene) |

Note on `phd_work/parity/`: this rule ignores the parity reference files. They are tracked anyway, added with `git add -f`.

## Added top-level files (not in upstream)

| File | Added in | Purpose |
| ---- | -------- | ------- |
| `pyproject.toml`, `uv.lock`, `.python-version` | 423d979 | Pinned environment (uv, Python 3.12, torch 2.12.0 cu130). `pyproject.toml` disables setuptools auto-discovery in 2cf37e0 |
| `CLAUDE.md` | 69736aa | Project rules for this fork |
| `README.md` banner | see `git log -- README.md` | Three-line pointer to `phd_work/README.md` and this file. Rest of the README is upstream |
| `phd_work/` | 69736aa and later | Port code, logs, parity harness, references, port log |

## Not changed

Model code (`src/conv_onet`, `src/encoder/pointnet.py`, `src/encoder/unet*.py`, decoders), the training loop, and the checkpoint loader are byte-identical to upstream. `src/encoder/pointnet.py` still imports `torch_scatter` (see `phd_work/README.md`, known limitations).
