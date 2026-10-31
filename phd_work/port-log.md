# Port log

Chronological log of every problem hit while porting ConvONet to a current stack,
with cause, fix and impact on model behaviour. Upstream baseline: tag `upstream-838bea5`.

## Environment

| Item         | Value                                                                                                      |
| ------------ | ---------------------------------------------------------------------------------------------------------- |
| OS / GPU     | Fedora 44, RTX 4070 12 GB (sm_89), driver 615.78.08 (CUDA ≤ 13.4)                                          |
| Compiler     | GCC 16.2.1                                                                                                 |
| Python       | 3.12.15 (uv-managed; system Python 3.14 not used)                                                          |
| Key packages | torch 2.12.0+cu130, torchvision 0.27.0+cu130, torch-scatter 2.1.2+pt212cu130, numpy 1.26.4, Cython 0.29.37 |
| Full pin     | `uv.lock`                                                                                                  |

**Why torch 2.12 and not the newest 2.14:** 2.12 is the newest torch for which PyG
publishes prebuilt `torch_scatter` wheels (checked 2026-10-18: 2.10–2.12 yes, 2.13/2.14 only `pyg_lib`).
Plan: run the original code with `torch_scatter` first (reference), then replace it with native
`scatter_reduce_` under a parity test, then upgrade torch separately.

**Why `numpy<2`, `Cython<3`:** minimal source changes first; each upgrade later as its own change with a parity test.

---

## 2026-10-18 — #1 `uv add`: download timeouts

- **Symptom:** `Failed to download torch==2.12.0+cu130 … network timeout (UV_HTTP_TIMEOUT 30s)`; same for `fonttools` metadata.
- **Cause:** dataset download (aria2c, 16 connections) saturated the network link.
- **Fix:** paused the dataset download; `export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2`.
- **Model impact:** none.

## 2026-10-18 — #2 `uv add torch-scatter`: tried to build from source

- **Symptom:** `Failed to build torch-scatter==2.1.2 … ModuleNotFoundError: No module named 'torch'`.
- **Cause:** PyG wheels were configured via `find-links`, which uv does not prefer over PyPI → uv picked the PyPI sdist.
- **Fix:** PyG page as a flat index (`format = "flat"`, `explicit = true`), `torch-scatter` pinned to it in `[tool.uv.sources]` (Linux only).
  Lock restricted to `linux x86_64` and `darwin arm64` via `[tool.uv] environments`.
- **Verify:** `uv tree --depth 1` shows `torch-scatter v2.1.2+pt212cu130` (no "Building").
- **Model impact:** none (same upstream library, prebuilt for this torch).

## 2026-10-18 — #3 Sanity check

- `torch 2.12.0+cu130`, CUDA runtime 13.0, `cuda.is_available() == True`, device RTX 4070.
- `torch_scatter 2.1.2+pt212cu130` imports.

## 2026-10-18 — #4 `build_ext`: `longintrepr.h: No such file or directory`

- **Symptom:** `src/utils/libkdtree/pykdtree/kdtree.c:181: fatal error: longintrepr.h`.
- **Cause:** `kdtree.c` is a checked-in Cython **0.27.3** output; such old generated C includes
  `longintrepr.h`, which Python 3.11 moved to `cpython/`. The source `kdtree.pyx` is in the repo.
- **Fix:** `setup.py` builds from `kdtree.pyx` instead of `kdtree.c` (one line);
  stale generated `kdtree.c` removed from git; generated `.c/.cpp` added to `.gitignore`.
- **Model impact:** none. KD-tree is used only in evaluation (`src/eval.py`, `src/common.py`),
  same algorithm source; exact nearest-neighbour search.

## 2026-10-18 — #5 `build_ext`: copy to `src/src/...` fails

- **Symptom:** `could not create 'src/src/utils/libkdtree/pykdtree/kdtree…so'` (compilation itself succeeded).
- **Cause:** our new `pyproject.toml` has a `[project]` table → setuptools enables package auto-discovery,
  sees `src/` and assumes *src-layout*, mapping package `src.*` to `src/src/*`. Upstream had no `pyproject.toml`.
  (Here `src` itself is the package: `from src import config`.)
- **Fix:** `[tool.setuptools] packages = []` in `pyproject.toml` (disables auto-discovery; extensions stay defined in `setup.py`).
- **Model impact:** none.

## 2026-10-18 — #6 `build_ext` result

- TODO: OK / next error; list of built `.so` files.

---

## 2026-10-19 — #7 `smoke.py`: `np.int` removed in numpy 1.26 (upstream code)

- **Symptom:** `AttributeError: module 'numpy' has no attribute 'int'` at `src/utils/binvox_rw.py:206`
  (`def dense_to_sparse(voxel_data, dtype=np.int):`), raised on `from src import config` → `src.data` → `src.data.fields` → `src.utils.binvox_rw`.
- **Cause:** `np.int` was removed in NumPy 1.24; upstream code predates that. Pinned numpy is 1.26.4 (`numpy<2` pin).
- **Fix:** replaced `np.bool` → `bool` and `np.int` → `int` (word-bounded, so `np.bool_`, `np.int32`, `np.intp`
  untouched) in all upstream `.py` files: `src/utils/binvox_rw.py` (9 sites incl. `np.int` at line 206),
  `src/utils/libmesh/inside_mesh.py` (2), `src/utils/voxels.py` (2). Commit `[port] numpy: replace removed aliases`.
- **Model impact:** none, no parity test needed. The removed aliases were the builtins themselves
  (`np.bool is bool`, `np.int is int`), so the resulting dtypes are bit-identical.

## 2026-10-19 — #8 `smoke.py`: `yaml.load()` needs `Loader` (upstream code)

- **Symptom:** `TypeError: load() missing 1 required positional argument: 'Loader'` at `src/config.py:22` (`cfg_special = yaml.load(f)`), raised from `config.load_config`.
- **Cause:** PyYAML 6.0 made `Loader` a required argument of `yaml.load`. Upstream code predates that.
- **Fix:** `yaml.load(f)` → `yaml.load(f, Loader=yaml.Loader)` at all 3 sites: `src/config.py` (2×, lines 22 and 33),
  `src/data/core.py` (1×, line 71). `yaml.Loader` is the default PyYAML 3.13 used when `Loader` was omitted
  (deliberately not `safe_load`, which would accept fewer tags). Commit `[port] yaml: pass Loader=yaml.Loader explicitly`.
- **Model impact:** none, no parity test needed. Making the old implicit default explicit gives identical parsing.

## 2026-10-19 — #9 `smoke.py`: `KeyError: 'onet'` in `config.get_model` (upstream config)

- **Symptom:** `KeyError: 'onet'` at `src/config.py:70` (`method_dict[method]`); `method_dict` only has `'conv_onet'`.
- **Cause:** `configs/voxel/shapenet_3plane.yaml` (inherited by `configs/voxel/pretrained/shapenet_3plane.yaml`)
  sets `method: onet`, overriding `conv_onet` from `configs/default.yaml`. Configs are identical to upstream
  (`git diff upstream-838bea5 -- configs` is empty), so this is an upstream bug. 3 of 16 configs have `method: onet`.
- **Fix:** `method: onet` → `method: conv_onet` in the 3 affected voxel configs only
  (`configs/voxel/shapenet_1plane.yaml`, `shapenet_grid32.yaml`, `shapenet_3plane.yaml`). Commit `[cfg] voxel configs: method onet -> conv_onet`.
- **Model impact:** none, no parity test needed. The `method` key only selects the module;
  `conv_onet` is the only implementation in `method_dict`.

## 2026-10-19 — #10 `generate.py`: `TypeError: dtype 'str' does not support operation 'mean'` (upstream code)

- **Symptom:** after all 26 smoke objects were generated, `generate.py:210`
  (`time_df.groupby(by=['class name']).mean()`) raised `TypeError: dtype 'str' does not support operation 'mean'`.
- **Cause:** `time_df` has a non-numeric column (`name`); pandas ≥ 2.0 no longer silently drops non-numeric
  columns in `groupby().mean()` (default `numeric_only=False`). Installed pandas 3.0.6. Upstream predates this.
- **Impact:** only the final timing summary. Meshes and `time_generation_full.pkl` were written before the crash.
  `time_generation.pkl` (per-class) was not.
- **Fix:** `.groupby(by=['class name']).mean()` → `.mean(numeric_only=True)` in `generate.py:210` and `eval_meshes.py:160`
  (`eval_meshes.py` has the same pattern on a frame with string columns). The follow-up `time_df_class.mean()` /
  `eval_df_class.mean()` calls need no change: after the groupby the frames are purely numeric.
  Commit `[port] pandas: numeric_only=True in groupby/mean summaries`.
- **Model impact:** none, no parity test needed. Post-processing of timings/metrics only; `numeric_only=True`
  reproduces pandas<2 behaviour (non-numeric columns dropped), so the printed tables have the same columns.

## Warnings kept on purpose

- **Cython `language_level not set, using 2`:** kept. Upstream was built with Py2 semantics
  (e.g. `/` on ints = integer division). Cython 3 defaults to level 3 → semantic change → parity risk.
  Must be handled explicitly when upgrading Cython.
- **`Not all members given for struct 'Voxel'/'Vertex'/'Triangle'`:** present upstream, harmless.
- **`ninja not found`, `compiler (c++) not compatible … g++`:** from torch `BuildExtension`, which `setup.py`
  uses although there are no CUDA/torch extensions; `c++` is g++ on Fedora. Harmless.

## Open items

- macOS: `pykdtree` is compiled with `-fopenmp`, which Apple clang does not support → Mac build will need handling (Mac is read/debug only for now).
- Later upgrades, each with parity test: native `scatter_reduce_` instead of `torch_scatter` → torch 2.14 → numpy 2 → Cython 3.

- OK. All 6 extensions built: kdtree, mcubes, triangle_hash, mise, simplify_mesh, voxelize.
- New warning: `-std=c++20 is valid for C++ but not for C` on C sources; torch `BuildExtension`
  adds the flag to every extension. Harmless (gcc ignores it for C).

## 2026-10-19 — Dataset
- dataset_small_v1.1.zip, 73 419 791 657 B, `unzip -tq` OK; extracted 80 GB to ~/PhD/data/onet/ShapeNet,
  symlinked as `data` in repo root (ignored by upstream `/data`).
- 13 classes, 8751 test objects (`wc -l` shows 8738: test.lst files lack a trailing newline).
- Object dir: model.binvox, points.npz, pointcloud.npz, img_choy2016/.

## Full test-set eval (2026-10-19)
- 8751/8751 meshes generated and evaluated, 0 missing.
- 2× upstream warning `contains1 != contains2` (benign).
- Class-mean results vs `paper_reference.md` (Table 2, Ours-2D 3×64²):

  | Metric              | Ours   | Paper |
  | ------------------- | ------ | ----- |
  | IoU                 | 0.7522 | 0.752 |
  | Chamfer-L1 (raw)    | 0.0086 | 0.0092 (paper ×10 = 0.092) |
  | Chamfer-L1 (×10)    | 0.086  | 0.092 |
  | Normal C.           | 0.9086 | 0.905 |
  | F-Score             | 0.7299 | 0.735 |

- **Known deviation:** Chamfer-L1 is 7% lower than the paper; cause unknown. Candidates: trimesh 4.x surface
  sampling; paper evaluation run vs released checkpoint.
- **Decisive test deferred:** evaluate the same meshes in the legacy env. Not needed for relative comparisons.

## Training step benchmark (2026-10-26)
- Config: `phd_work/configs/voxel_3plane_bench.yaml`; `train.py --exit-after 600`; RTX 4070; batch 64 × 1024 points.
- 5100 iterations in 600 s.
- Median 0.1052 s/it (49 intervals from it=300, range 0.102–0.113); includes data loading.
- GPU memory (`nvidia-smi`): peak 5835 MiB, idle 2348 MiB, delta ~3.49 GB. Upper bound: includes CUDA context and
  allocator cache.
- Training path ran with no new port errors.

## 2026-10-31 — Parity harness: GPU results not reproducible by default

- **Symptom:** `phd_work/parity.py compare` on unchanged code failed: encoder `xz`/`yz` and logits differed by
  ~0.02 between two identical runs on the RTX 4070 (`xy` identical).
- **Cause:** non-deterministic GPU kernels in the voxel encoder path (scatter-add via atomics in `torch_scatter`,
  and/or cuDNN algorithm choice). Fixed by `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. With these, repeated GPU runs are bit-identical;
  CPU runs are bit-identical too.
- **Fix:** set in `phd_work/parity.py` (new, `[phd]`). Reference captured in that mode; `.gitignore` excludes
  `phd_work/parity/` (the reference tensors are regenerated, not committed).
- **Cross-device:** CUDA vs CPU outputs differ by 0.026 in logits for the same weights and inputs (floating point
  summation order and different kernels). References are therefore device-specific; `compare` refuses a device mismatch.
- **Model impact:** none. This is a test-harness issue. But it means any GPU run is only reproducible to ~1e-2 in
  encoder features *unless* deterministic mode is on. Relevant when reading small metric differences, e.g. the
  Chamfer-L1 deviation above (not yet checked against this noise).
- **Reference:** `phd_work/parity/ref_voxel3plane.pt`, sha256 `40fca362a48ef195ed1f42bc5504478a017a532b1c57f8a74876c1683a812f7e`,
  CUDA, torch 2.12.0, seed 0, 2 synthetic inputs (32³ occupancy, 20000 query points). Scope: `voxel_simple_local`
  encoder + `simple_local` decoder only (the path used by the pretrained 3-plane model).

## 2026-10-31 — [port] voxel encoder: `torch_scatter.scatter_mean` → native `scatter_reduce_`

- **Change:** `src/encoder/voxels.py` drops `from torch_scatter import scatter_mean`; a local `scatter_mean(src, index, out)`
  calls `out.scatter_reduce_(-1, index.expand_as(src), src, reduce='mean', include_self=False)`. Valid only for a
  zero-initialised `out`, which is how both call sites use it (empty bins stay 0, as with torch_scatter).
- **Scope:** voxel encoder only (`voxel_simple_local`, used by the pretrained 3-plane model). `pointnet.py` still uses
  `torch_scatter` (`scatter_mean`, `scatter_max`); `torch_scatter` stays in the environment until that is ported.
- **Parity:** `uv run python phd_work/parity.py compare` → **PARITY OK**, max|diff| = 0 for all three planes and logits
  (CUDA, deterministic mode, reference captured on the unchanged code).
- **Sensitivity check:** a deliberately wrong variant (`include_self=True`) gives max|diff| = 0.39 in logits, so the
  harness detects real differences.
- **Unit check:** on random data (incl. empty bins, shapes of plane and grid), native vs torch_scatter differ by ≤ 2.4e-7
  (float32 summation order). The model forward was bit-identical, which is within that tolerance.
- **Model impact:** none expected; verified by parity above.

## 2026-10-31 — Parity: provenance, reference from the old code, gradients

- **References captured from d4789b2** (last commit before the swap, still `torch_scatter`), in a temporary
  `git worktree` at `../convonet-ref` with the updated `parity.py` copied in. Compared on `ca728fb` (native scatter).
  The worktree was built with the same compiled extensions (`.so`) copied from `main`; `src/utils` sources are unchanged
  between the two commits.
- **Provenance:** each reference stores the commit, a dirty flag for `src/` and `configs/`, the harness sha256, torch
  version and device. `capture` refuses to run when model code has uncommitted changes.
- **Forward, CUDA** (`ref_voxel3plane.pt`, sha256 `f936ca836ff64ef5367e31b65df7023c06360dffd81da0e94a47501ada2307cb`):
  max|diff| = 0 for `xz`, `xy`, `yz` and logits (max|ref| ≈ 31).
- **Gradients, CPU** (`ref_voxel3plane_grad_cpu.pt`, sha256 `922c722f46afec35d16ae4a15905baebf5e2f9c67b3d7d48152018175d2ae049`):
  loss 4.07933664 (diff 0); max|diff| = 0 for `encoder.conv_in.weight` (max|ref| 2.4e-1), `encoder.conv_in.bias` (4.1e-1),
  `decoder.fc_p.weight` (9.2e-3), `decoder.fc_p.bias` (3.6e-1). Loss is BCE-with-logits against fixed seeded random
  targets, one backward, model in eval mode.
- **Sensitivity:** the wrong variant (`include_self=True`) moves the gradients by 3.9e-4 to 4.8e-2, so the gradient
  check detects a real change.
- **Gradient mode is CPU-only:** `grid_sampler_2d_backward_cuda` has no deterministic implementation; with
  `use_deterministic_algorithms(True)` torch raises on backward. So gradients are compared on CPU, which is deterministic.
- **Still required: `torch_scatter`.** `src/encoder/__init__.py` imports `pointnet` unconditionally, and `pointnet.py`
  imports `torch_scatter` at module level. So importing `src` fails without `torch_scatter`, including on Mac. The
  voxel encoder no longer uses it, but the dependency stays until `pointnet.py` is ported. Not done here, and it needs its
  own parity reference.
- **Not run:** full test-set evaluation (not needed for this check).
