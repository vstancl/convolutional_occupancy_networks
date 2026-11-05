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

- OK. All 6 extensions built: kdtree, mcubes, triangle_hash, mise, simplify_mesh, voxelize.
- New warning: `-std=c++20 is valid for C++ but not for C` on C sources; torch `BuildExtension`
  adds the flag to every extension. Harmless (gcc ignores it for C).

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
- `torch_scatter` is still required: `src/encoder/__init__.py` imports `pointnet` unconditionally, and `pointnet.py`
  imports `torch_scatter` at module level, so `import src` fails without it (on Mac too). The voxel encoder no longer uses
  it. Porting `pointnet.py` needs its own parity reference. Not done.
- Later upgrades, each with parity test: torch 2.14 → numpy 2 → Cython 3. (Native `scatter_reduce_` is done for the voxel path, see the swap entry.)
- **Known gap: GPU backward of `scatter_reduce_` is not parity-tested.** The gradient check runs on CPU (CPU gradients
  match the old code exactly). CUDA backward of `grid_sampler_2d` has no deterministic implementation, so it cannot be
  compared exactly under deterministic mode.

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

- **Known deviation:** Chamfer-L1 is about 6.5% lower than the paper (0.0086 vs 0.0092 raw; the 7% in earlier notes was a
  rounding of this). Cause unknown. Candidates: trimesh surface sampling (the locked and installed trimesh is 5.1.1; the earlier "trimesh 4.x" note
  was wrong; the eval-time version was not recorded); paper evaluation run vs released checkpoint.
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

## 2026-10-31 — Fresh-clone check: pretrained checkpoint does not load without CUDA (upstream code)

- **Symptom:** `phd_work/parity.py compare --grad --device cpu` with the GPU hidden (`CUDA_VISIBLE_DEVICES=`) fails in
  model loading: `Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`.
- **Cause:** the pretrained checkpoint was saved from CUDA tensors. `src/checkpoints.py` `load_url` calls
  `model_zoo.load_url(url, progress=True)` without `map_location`, so the tensors are restored on CUDA.
- **Fix:** `map_location='cpu'` in `load_url`. Commit `1fc3353` (`[port]`, one line).
- **Model impact:** identical. `load_state_dict` copies the values into the module, so the weights do not change; only the
  intermediate storage device changes. Verified: `parity.py compare` (CUDA) and `compare --grad --device cpu` pass with
  max|diff| = 0 on every tensor, both with and without a visible GPU.
- **Not fixed (latent):** `CheckpointIO.load_file` (local files) calls `torch.load(filename)` without `map_location`; the same
  problem would occur for a CUDA-saved local file. Not on the documented path, so left unchanged.
- **Also found:** the README claimed the `--grad` check runs without a GPU; it did not, until this fix.

## 2026-10-31 — Fresh-clone check of the reproduction README

- Steps run literally from a fresh clone of the committed repository: `uv sync` (exit 0, versions as documented);
  `setup.py build_ext --inplace` (exit 0, six `.so` files); `ln -s` of the existing dataset to `data`; `phd_work/smoke.py`
  (`OK: all logits finite`); `parity.py compare --device cuda` and `compare --grad --device cpu` (`PARITY OK`);
  quick generate and eval on the 26-object smoke set (both exit 0).
- **README errors found and fixed:**
  - Build warnings list named `ninja not found`, `compiler (c++) not compatible` and `-std=c++20`. The fresh build did not
    print the last two; it printed the ninja fallback message plus signed/unsigned comparisons, NumPy deprecated-API
    `#warning` and narrowing conversions, which were missing. List corrected.
  - "Runs without a GPU" for the `--grad` check: false until the checkpoint fix above; now verified.
- **Dataset:** size check passed (73419791657 B); full `unzip -tq dataset_small_v1.1.zip` passed (`No errors detected`, exit 0).
- **Not re-run:** download of the dataset zip, extraction, and the full
  generate/eval (outputs are from the port).

## 2026-10-31 — Resume from a local checkpoint crashes: `torch.load` rejects numpy.float64 (upstream code)

- **Symptom:** resuming from `model.pt` or `model_best.pt` written after a validation pass fails with
  `_pickle.UnpicklingError: Weights only load failed` (`torch.load` with `weights_only=True`).
  Reproduced with a minimal script: `torch.save({'loss_val_best': np.mean([...])})` loads with
  `weights_only=False` and fails with the default.
- **Cause:** `BaseTrainer.evaluate` returns `np.mean(...)` (numpy.float64). `train.py` stores it as `loss_val_best`
  after the first validation. torch ≥ 2.6 defaults `torch.load` to `weights_only=True`, which does not allow numpy
  globals. `train.py` only catches `FileExistsError` around `checkpoint_io.load`, so the resume crashes.
- **Why the other tests did not show it:**
  - The pretrained URL load (`load_url`, `model_zoo`) is a different path and the pretrained file contains only tensors.
  - The 10-min benchmark checkpoint (`out/phd_bench`, it = 5121) was written before the first validation
    (`validate_every: 10000`), so `loss_val_best` is still the Python float `-inf`.
- **Fix:** `torch.load(filename)` → `torch.load(filename, weights_only=False)` in `src/checkpoints.py` `load_file`
  (one line, commit `94ab38b`, `[fix]`). `map_location` and `train.py` unchanged.
- **Model impact:** none, no parity run. `weights_only=False` is the behaviour of torch < 2.6, which this file was
  written for. The loaded objects are the same; only the unpickler's allowed-globals check is disabled.
  The checkpoints are our own local files, not untrusted input.
- **Alternatives rejected:**
  - `float(metric_val_best)` in `train.py`: fixes this one scalar, but `train.py` is kept byte-identical to upstream,
    and the next numpy scalar saved would break the same way.
  - `torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, ...])`: needs an exact list of numpy
    internals that depends on the numpy version. Fragile.
- **Still open:** `load_file` has no `map_location`, so a CUDA-saved local file still needs a GPU to load (latent, see the
  earlier entry).

## 2026-10-31 — Resume check: `phd_work/resume_check.py` and the resume config

- `phd_work/configs/voxel_3plane_resume.yaml` inherits `configs/voxel/shapenet_3plane.yaml`, sets `out/phd_resume`,
  `print_every: 20`, `validate_every: 500`, `checkpoint_every: 500`, `visualize_every: 0`.
- `phd_work/resume_check.py <ckpt> [--min-it N]`: loads through `CheckpointIO` with no modules registered (returns all
  entries), prints scalars and the type of `loss_val_best`, and checks that every Adam per-parameter `step` equals `it`.
  This holds after a resume only if the optimizer state was restored. `--min-it N` requires `it > N`.
- Check on the existing benchmark checkpoint (`out/phd_bench/model.pt`, it = 5121): `PASS`, Adam step = 5121.
- Missing file: `FAIL`, exit 1.
- The result of the two-run resume test is in the next entry.

## 2026-10-31 — Resume test result (two runs, GPU, `configs/voxel_3plane_resume.yaml`)

Logs: `phd_work/logs/resume_run1.log`, `resume_check_run1.log`, `resume_run2.log`, `resume_check_run2.log`.

- **Run 1** (`--exit-after 300`, fresh `out/phd_resume`): exit 3, wall 5m58.96s (user CPU 17m35s; GPU utilisation not sampled).
  Validations: it=500 `Validation metric (iou): 0.5943`, `New best`; it=1000 `0.5986`, `New best`, then `Time limit reached. Exiting.`
  Last printed loss it=1000: 41.4415 (it=980: 30.5736; it=20: 514.6266).
- **Checkpoint after run 1:** `resume_check` PASS, `it = 1000`, `epoch_it = 3`, Adam step = 1000 for all parameters.
  `loss_val_best` = 0.5986212 (saved as `numpy.float32`, see deviation below).
- **Run 2** (`--exit-after 120`, resumed from `out/phd_resume/model.pt`, no `UnpicklingError`): exit 3, wall 2m03.44s.
  First line: `Current best validation metric (iou): 0.59862119` (= run 1's last `New best`, 0.5986 at 4 decimals).
  First printed loss it=1020: 31.9023 (continues from ~30–42 range of run 1; no jump back toward 514). Run 2 exited at it=1427.
- **Resume check after run 2:** `resume_check --min-it 1000` PASS, `it = 1427`, Adam step = 1427 for all parameters.

**Deviation from the test brief (criterion a):** the brief expected `loss_val_best` to be `float64`. It is `numpy.float32`.
Cause: `evaluate` takes `np.mean` of float32 values, so the result is float32. The unpickling mechanism is the same
(a numpy scalar global, rejected by `weights_only=True`), so the fix covers it; the dtype in the brief was wrong.
The fix was not changed.

**Correction of the brief's float64 claim:** the brief said `loss_val_best` would be `float64`. That was wrong: `np.mean` over
`numpy.float32` values returns `numpy.float32`. The float32 result was accepted; no float64 test was run.
**Any numpy scalar triggers the same rejection.** Checked with default `torch.load` (weights_only=True) on a saved dict
containing one scalar each: `np.float32`, `np.float64` and `np.int64` all raise `UnpicklingError`. So the fix covers any numpy scalar
that train.py saves, not only the dtype seen here.

**Timings:** run 1 ≈ 6 min, of which the 2 validations cost roughly 65 s each (it=500: checkpoint at 147 s, next print at 213 s).

## 2026-10-31 — 3-class voxel baseline: class IDs and object counts

Synset check (`data/ShapeNet/metadata.yaml`): `03001627` = chair, `04379243` = table, `03636649` = lamp. All three folders exist.

Object counts per split. Source: non-empty lines of `<synset>/{train,val,test}.lst` (`grep -c .`). The loader reads these files,
so these are the sizes it will use. `wc -l` gives one less, because the last line has no trailing newline.

| Class | train | val | test | total | object folders on disk |
| ----- | ----- | --- | ---- | ----- | ---------------------- |
| chair (03001627) | 4746 | 677 | 1355 | 6778 | 6778 |
| table (04379243) | 5958 | 850 | 1701 | 8509 | 8509 |
| lamp (03636649)  | 1624 | 231 | 463  | 2318 | 2318 |
| **3 classes** | **12328** | **1758** | **3519** | **17605** | |

Each class's split total equals its folder count, so no object is missing from the lists.

## 2026-10-31 — 3-class smoke run (`voxel_3plane_3cls.yaml`, `--exit-after 600`)

Log: `phd_work/logs/3cls_smoke.log`. Started 17:14:30, exited with `Time limit reached. Exiting.` at it=3500 (time 594.26 s in the log, exit code 0).

- Classes: 3 (`03001627`, `04379243`, `03636649`). The validation set has 1758 objects, which matches the 3-class val total in the counts entry above.
  The upstream `train.py` does not print dataset sizes, so the 1758 comes from the validation progress bar.
- Throughput: about 6.5 it/s (it=100 at 39 s, it=3500 at 594 s).
- Validation at it=2000: `Validation metric (iou): 0.6152`, `New best model (loss 0.6152)`. Validation pass: 24 s (`1758/1758 [00:24, 71.80it/s]`).
- Losses ranged 32–55 over it=100–3500, with no divergence.
- Checkpoints in the smoke run: `out/phd_3cls` was deleted after the run, as the brief asks.

## 2026-10-31 — Full 3-class run started (`3cls_run1`)

- Command: `uv run python train.py phd_work/configs/voxel_3plane_3cls.yaml --exit-after 12600`, with `PYTHONUNBUFFERED=1`,
  stdout and stderr tee'd to `phd_work/logs/3cls_run1.log`. Started 2026-10-31 17:25:15 CET under `nohup` (not tmux).
  `--exit-after` is in seconds (12600 s = 3.5 h).
- Expected end: about 20:55 CET.
- Check progress: `tail -f phd_work/logs/3cls_run1.log` (or `grep -a "Validation metric" phd_work/logs/3cls_run1.log`).
- Stop early: `pkill -f "train.py phd_work/configs/voxel_3plane_3cls.yaml"` (the run saves `out/phd_3cls/model.pt` on a clean
  exit only; a kill may leave the last checkpoint at the previous `checkpoint_every` multiple).
- Parser: `phd_work/val_curve.py <log>` writes `<log>`-named CSV (it, iou) and PNG and prints the best IoU, its
  iteration, and the gain over the last 20k iterations. Tested on the smoke log (1 validation, it=2000, IoU 0.6152).

## 2026-11-01 — 3-class run1 result, evaluation, speed check, stopping rule

**Status:** steps 1–3 done. Step 4 (resume to +50k in tmux) **not started**: the resume point differs from the brief (see Finding A).

### Run1 outcome

- `phd_work/logs/3cls_run1.log`: 37 validations, it=2000..74000. Best val IoU **0.6868 at it=70000** (`out/phd_3cls/model_best.pt`).
- The log's last validation is at it=74000 (0.6849). The run exited with `Time limit reached` after 38 more iterations (see A).
- Spike: 0.6822 at it=38000 (the old-rule baseline).

### Reference values on the 3-class validation set (1758 objects)

Command: `uv run python phd_work/val_eval.py phd_work/configs/voxel_3plane_3cls.yaml [--model URL]`. Logs: `phd_work/logs/val_eval_pretrained.log`, `phd_work/logs/val_eval_3cls_best.log`.

| Weights | IoU | iou_voxels |
| ------- | --- | ---------- |
| Pretrained, `voxel_shapenet_3plane.pt` (URL from `configs/voxel/pretrained/shapenet_3plane.yaml`; 13-class model) | 0.690104 | 0.732813 |
| `out/phd_3cls/model_best.pt` (it=70000) | 0.686806 | 0.729516 |

- The 3cls model_best value reproduces the logged 0.6868 with a diff of 6e-6, within the 4-decimal rounding of the log. Only one evaluation run was made, so run-to-run nondeterminism was not measured.
- These are **not** paper-reference numbers. The paper's IoU 0.752 (`phd_work/paper_reference.md`) is for the 13-class test set. The pretrained model is trained on all classes and scores 0.6901 here on the 3-class val split.

### Speed check

Commands: `uv run python train.py phd_work/configs/voxel_3plane_bench.yaml --exit-after 300` (log `phd_work/logs/bench_phd.log`) and the same 300 s with `out_dir` set to a scratch dir (config in the session scratchpad, which inherits `phd_work/configs/voxel_3plane_3cls.yaml`; log `phd_work/logs/speed_3cls_scratch.log`). GPU logged with `nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu,memory.used --format=csv -l 5` (`phd_work/logs/gpu_bench_phd.csv`, `phd_work/logs/gpu_speed_3cls_scratch.csv`).

| Run | Start it | Windows used | Median s/it | Mean s/it | Mean GPU util | Peak GPU mem |
| --- | -------- | ------------ | ----------- | --------- | ------------- | ------------ |
| bench (13 classes) | 5200 (resumed) | it 5200–6700 | 0.1538 | 0.1548 | 77% | 4.1 GB |
| 3cls scratch | 0 | it 300–1700 | 0.1556 | 0.1559 | 81% | 4.1 GB |

- Each window is one 100-iteration print interval, so the median is over 15 (bench) and 14 (3cls) windows.
- Discrepancy, explained: the bench did **not** start fresh. `out/phd_bench/model.pt` was left over from an earlier bench run, so it resumed at it=5200 and its 300 s covered it 5200–6700. The "exclude the first 300 it" rule therefore did not apply the way it would on a fresh run. The 3cls scratch run was fresh.
- Run1 average check: 12565 s at it=74000 includes 37 validations of about 27 s each (about 1000 s), so (12565 − 1000) / 74000 ≈ 0.156 s/it, in line with the medians.
- Other GPU users: the only processes on the GPU during the runs were desktop graphics processes (`kwin_wayland`, `plasmashell`, `firefox`, …, type G). No other compute process appeared. This is noted from the process list, not from a sampled per-process trace.
- `out/phd_bench` was removed with `rm -rf` after the bench, as the brief asked. `out/phd_speed3cls_scratch` (the 3cls scratch run) is kept, not deleted.

### Stopping rule: changed

- **Old rule:** stop when the best IoU in the last 20k iterations exceeds the best at or before (last − 20k) by ≤ 0.005. On run1 this is best 0.6868 (it=70000) against best 0.6822 (it≤38000): gain 0.0046, which would have stopped the run. The 38k spike sets the baseline, so the statistic depends on one validation point.
- **New rule** (`phd_work/val_curve.py`): least-squares slope of val IoU over the validations with it ≥ last − 20k, in IoU per 20k iterations. Continue if slope ≥ 0.005 per 20k.
- Run1 result: slope **+0.00904 IoU per 20k** (n = 11 validations, it 54000–74000), residual SD 0.00291. The standard error of the slope is about 0.0028 (residual SD / √Sxx with Sxx = 1.1 in 20k units; hand-calculated, not printed by the script). The rule says continue.
- The slope is driven by the final validations, so a single new peak would move it. It is still above the threshold.

### Finding A: the resume point is 74038, not 74000

- `out/phd_3cls/model.pt` (written on exit) has `it = 74038` (`resume_check.py`: `it = 74038`, Adam step 74038). The log's last validation is at it=74000.
- Cause: the validation at it=74000 took about 27 s after the 12565 s print, which put the time just under `--exit-after 12600`. The exit check then fired 38 iterations later, at it=74038. No `it=` print occurs between 74000 and 74038 because prints are every 100 iterations.
- Consequence: a resume will start at 74038 and report `Current best` 0.6868. The brief expected it=74000.
- Decision: not resumed. The brief says to stop and report on any difference. Run2 is one command away, once the resume point is accepted.

### Errors and fixes

- `pkill -f "nvidia-smi --query-gpu"` matched the shell that was running it, which killed the command (exit 144). Fix: stop the logger by PID (`kill $GPID`) instead of matching the command line. The bench itself had already finished (exit 3 = `Time limit reached`).
- `bench` resumed from a leftover checkpoint (see Speed check). Fix: remove `out/phd_bench` after each bench run, and start a bench from an empty out dir.

### Commands run

- `uv run python phd_work/val_curve.py phd_work/logs/3cls_run1.log --out-prefix phd_work/logs/3cls_run1_val`
- `uv run python phd_work/val_eval.py phd_work/configs/voxel_3plane_3cls.yaml --model https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/models/voxel/voxel_shapenet_3plane.pt`
- `uv run python phd_work/val_eval.py phd_work/configs/voxel_3plane_3cls.yaml`
- `uv run python train.py phd_work/configs/voxel_3plane_bench.yaml --exit-after 300`
- `uv run python train.py <scratch>/speed_3cls.yaml --exit-after 300` (`out_dir: out/phd_speed3cls_scratch`)
- `uv run python phd_work/resume_check.py out/phd_3cls/model.pt` → `it = 74038`

## 2026-11-01 — Run2 started (resume from it=74038, +9000 s)

- Commit `36b2e7b` (`[phd] logs: ...`) also added `phd_work/val_curve.py` and `phd_work/logs/3cls_run1_val.csv`. Its message does not mention them. History was not rewritten.
- Run2: `tmux new-session -d -s phd_3cls_run2 "... train.py phd_work/configs/voxel_3plane_3cls.yaml --exit-after 9000 > phd_work/logs/3cls_run2.log 2>&1"`, started 13:32:18 CET.
- Start check: `Loading checkpoint from local file`, `Current best validation metric (iou): 0.68680882`, first print `it=74100`. The resume point is 74038 (see the entry above).

## 2026-11-02 — Run2 finished; plateau check; `max_it` option; test-set eval and variance run

### Run2 result and stopping decision

- Run2 ended at `Time limit reached. Exiting.` (`--exit-after 9000`), last print it=125100, last validation it=124000. Combined curve (run1 + run2, 62 validations, it 2000..124000): `phd_work/logs/3cls_combined.csv`, `.png`.
- Command: `uv run python phd_work/val_curve.py phd_work/logs/3cls_run1.log phd_work/logs/3cls_run2.log --out-prefix phd_work/logs/3cls_combined`.
- Best val IoU **0.6899 at it=106000** (run2 territory; run1 ended at 74k with best 0.6868).
- Slope over it ≥ 104000 (n=11 validations): **−0.00006 IoU per 20k** (printed to 5 decimals; −0.0001 at 4 decimals), residual SD 0.00227, SE of slope 0.0022. Threshold for continuing is ≥ 0.005 per 20k, so the rule says **stop**.
- Other segments (hand-computed from the CSV with `fit_line`, SE = residual SD / √Sxx in 20k units):
  - 54k–74k: +0.0090 per 20k, SE 0.0028 (n=11)
  - 76k–124k: +0.0027 per 20k, SE 0.0006 (n=25)
- Mean and SD (ddof=1) of the last 10 validations (it 106k–124k): **0.6859, SD 0.0022**. The brief expected ~0.6855. This differs by 0.0004 (about half an SE). The number logged here is the computed one.
- Old max-gain statistic (for comparison only): 0.0022 (best 0.6899 at 106k vs 0.6877 at ≤102k). It is driven by the single peak at 106k.
- Pretrained reference (validation, `phd_work/logs/val_eval_pretrained.log`): 0.6901. Our plateau sits about 0.0045 below it.
- **Possible post-resume dip, unproven:** mean 76k–90k = 0.6812 (n=8) vs 64k–74k = 0.6846 (n=6), difference −0.0034. The Welch z is about −3.4 (SE 0.0010) when the two windows are treated as independent. The windows were chosen after looking at the data, and consecutive validations are correlated, so this z overstates the evidence. The brief's "~2σ" estimate does not match this calculation. Treat the dip as a hypothesis for the variance run (step below) to test, not as a finding.
- Plateau level: the last-10 mean (0.6859) is the number to quote, with SD 0.0022 as the noise scale.

### `val_curve.py`: last-10 statistics

- Added to the stdout report: mean and SD (ddof=1) of the last 10 validations. Commit `e5fbdce`.

### `max_it` option in `train.py` ([cfg], commit `5836842`)

- New line: `max_it = cfg['training'].get('max_it', -1)`. After the validation block and before the `exit_after` check: if `max_it > 0 and it >= max_it`, print `Max iterations reached. Exiting.`, save `model.pt` with the same fields as the `exit_after` exit (`epoch_it`, `it`, `loss_val_best`), and `exit(3)`.
- Why no parity run: with `max_it` unset, `max_it = -1` and the branch is never entered. No tensor operation, RNG call or ordering changes. CLAUDE.md allows this without parity; the row is in `phd_work/CHANGES.md`.
- Test: `phd_work/logs/maxit_test.log`, scratch config (`$SCRATCHPAD/maxit_test.yaml`, `out_dir: out/phd_maxit_test`, `max_it: 300`, `validate_every: 300`). Validation at it=300 ran (`Validation metric (iou): 0.5210`), then `Max iterations reached. Exiting.`, exit code 3. `resume_check.py out/phd_maxit_test/model.pt`: `it = 300`, Adam step 300, PASS.

### Test-set evaluation of the 3-class baseline (step 3) and variance run (step 4)

- Configs: `phd_work/configs/voxel_3plane_3cls_test.yaml` (inherits `voxel_3plane_3cls.yaml`, `generation_dir: generation_3cls_best`, `test.model_file: model_best.pt`) and `phd_work/configs/voxel_3plane_3cls_s2.yaml` (`out_dir: out/phd_3cls_s2`, `max_it: 120000`).
- `model_file` is relative to `out_dir` (`src/checkpoints.py` `load_file` joins it to `checkpoint_dir`). The brief's `out/phd_3cls/model_best.pt` would resolve to `out/phd_3cls/out/phd_3cls/model_best.pt`, so the config uses `model_best.pt`, which is the same file.
- Launched in tmux session `phd_3cls_test` (script `chain.sh` in the scratchpad, log of the chain `phd_work/logs/chain_3cls_test.log`): generate → eval_meshes → train s2. Logs: `generate_3cls_best.log`, `eval_3cls_best.log`, `3cls_s2.log`. Exit codes and wall times are in the chain log.
- Chain outcome (`phd_work/logs/chain_3cls_test.log`): generate rc=0, 1189 s (19.8 min). eval_meshes rc=0; the chain's `t=2777` is cumulative from chain start, so eval alone was 2777 − 1189 = 1588 s (26.5 min, matches the progress bar). s2 rc=3 (the `max_it` exit code, `train.py` commit `5836842`), wall 20077 s (5.6 h); the log ends with `Max iterations reached. Exiting.` after the it=120000 validation.

### Results of step 3: test-set eval of the 3-class baseline

- Checkpoint: `model_best.pt` of run1+run2 (best validation IoU 0.6899 at it=106000, see the run2 section). The `it` of the saved file was not read from the checkpoint; it is inferred from the validation curve.
- Mesh CSV: `out/phd_3cls/generation_3cls_best/eval_meshes.csv`, copied to `phd_work/results/voxel_3plane_3cls_best.csv`.
- Reference: chair, table and lamp rows of `phd_work/results/voxel_3plane_pretrained_full.csv`. That file has no mean row, so the reference class mean is the mean of these three rows (computed here, not read from a file). The 3-class eval CSV also has no mean row; its mean is the same calculation.
- Columns: completeness and accuracy (mesh), NC = `normals (mesh)`, Chamfer-L1 = `chamfer-L1 (mesh)` ×10, F-Score = `f-score (mesh)`, IoU = `iou (mesh)`.

| class | source | completeness | accuracy | NC | Chamfer-L1 ×10 | F-Score | IoU |
|---|---|---|---|---|---|---|---|
| chair | 3cls_best | 0.0091 | 0.0086 | 0.9091 | 0.0883 | 0.7064 | 0.7299 |
| chair | pretrained_full | 0.0088 | 0.0086 | 0.9084 | 0.0874 | 0.7094 | 0.7304 |
| lamp | 3cls_best | 0.0139 | 0.0100 | 0.8565 | 0.1196 | 0.6705 | 0.6237 |
| lamp | pretrained_full | 0.0124 | 0.0098 | 0.8601 | 0.1108 | 0.6837 | 0.6311 |
| table | 3cls_best | 0.0089 | 0.0085 | 0.9293 | 0.0868 | 0.7059 | 0.6778 |
| table | pretrained_full | 0.0087 | 0.0085 | 0.9283 | 0.0860 | 0.7100 | 0.6786 |
| class mean (3) | 3cls_best | 0.0106 | 0.0090 | 0.8983 | 0.0982 | 0.6942 | 0.6771 |
| class mean (3) | pretrained_full | 0.0100 | 0.0090 | 0.8990 | 0.0947 | 0.7010 | 0.6800 |

- The 3-class mean IoU (0.6771) matches the `mean` row in `eval_3cls_best.log`, and the mean completeness (0.010621) and accuracy (0.009027) match too. That log's `mean` row reports `f-score-20` (0.9269), a different column from the `f-score (mesh)` used in the table, so it is not comparable with the F-Score column above.
- The eval log contains `Warning: contains1 != contains2 for some points.` lines. This warning is not in "Warnings kept on purpose" and its cause was not investigated in this step; it is recorded here so it can be checked.
- Reading: the 3-class model is slightly below the 13-class pretrained model on the three classes (mean IoU 0.6771 vs 0.6800), mostly in lamp (−0.007 IoU) and in Chamfer-L1 (lamp 0.120 vs 0.111). Chair is equal within 0.001 and table within 0.001. This is one checkpoint on one eval, with no seed spread.

### s2: second 3-class run from scratch (variance check), it=120000

- Config: `phd_work/configs/voxel_3plane_3cls_s2.yaml` (`out_dir: out/phd_3cls_s2`, `max_it: 120000`), otherwise the same as `voxel_3plane_3cls.yaml`. The run starts fresh (`Current best validation metric (iou): -inf`, empty out dir), not a resume. Whether the seed differs from run1 was not checked in the config.
- Exit: rc=3 at it=120000, the expected `max_it` exit. Validations every 2000 iterations: 60 points, it 2000–120000 (`phd_work/logs/3cls_s2.csv`, `3cls_s2.png`).
- `val_curve.py` on `3cls_s2.log`:
  - best val IoU **0.6893 at it=118000**;
  - mean and SD (ddof=1) of the last 10 validations (it 102000–120000): **0.6844, SD 0.0032**;
  - least-squares slope over it ≥ 100000 (11 validations): −0.00175 IoU per 20k, residual SD 0.0034.
- run1+run2 for comparison (`phd_work/logs/3cls_combined.csv`, same script): best 0.6899 at it=106000; last 10 (it 106000–124000): 0.6859, SD 0.0022; slope over it ≥ 104000: −0.00006 per 20k, residual SD 0.0023.
- Reading: the s2 plateau is 0.0015 below run1+run2 (0.6844 vs 0.6859). Its SD is 0.003 and the windows differ by 2–4k iterations, so the gap is about 1.4 SE and is not separable from run-to-run noise at this level. The s2 slope is still slightly negative over its last 20k, which is not seen in run1+run2. Two runs are not enough to say whether the 3-class plateau is seed-stable; a third run or a test-set eval of s2 would be needed.

### Checkpoint `it` and `epoch_it` (loaded with `CheckpointIO`, as in `resume_check.py`)

- `out/phd_3cls/model_best.pt`: `it = 106000`, `epoch_it = 550`, `loss_val_best = 0.6898695`. Matches the expected 106000.
- `out/phd_3cls_s2/model_best.pt`: `it = 118000`, `epoch_it = 612`, `loss_val_best = 0.6893486`. Matches the expected 118000.
- This settles the open point in the step 3 section: the saved `model_best.pt` for run1+run2 is the it=106000 checkpoint.

### `contains1 != contains2` warnings (line counts only, no investigation)

- `phd_work/logs/eval_full.log` (pretrained, 8751 objects): 2 lines, 2/8751 = 0.023 per 1000 objects.
- `phd_work/logs/eval_3cls_best.log` (3-class run1+run2, 3519 objects): 5 lines, 5/3519 = 1.42 per 1000 objects.
- Counts are of lines containing the warning. Each warning appears as its own line in the log. The 3-class rate is about 6× the pretrained rate, but the counts are small, and whether they relate to the model or the object set is not known.
- Also counted for the s2 test eval (`eval_3cls_s2_best.log`, 3519 objects): 5 lines.

### s2 test-set eval (`voxel_3plane_3cls_s2_test.yaml`, tmux `phd_3cls_s2_test`)

- Config: `phd_work/configs/voxel_3plane_3cls_s2_test.yaml` (inherits `voxel_3plane_3cls_s2.yaml`, `generation_dir: generation_s2_best`, `test.model_file: model_best.pt`), so the checkpoint is `out/phd_3cls_s2/model_best.pt` (it=118000).
- Chain log `phd_work/logs/chain_3cls_s2_test.log`: generate rc=0, 1252 s; eval_meshes rc=0, 1609 s.
- Logs: `generate_3cls_s2_best.log`, `eval_3cls_s2_best.log`. Mesh CSV: `out/phd_3cls_s2/generation_s2_best/eval_meshes.csv`, copied to `phd_work/results/voxel_3plane_3cls_s2_best.csv`.

| class | source | completeness | accuracy | NC | Chamfer-L1 ×10 | F-Score | IoU |
|---|---|---|---|---|---|---|---|
| chair | 3cls_s2_best | 0.0090 | 0.0084 | 0.9073 | 0.0869 | 0.7144 | 0.7301 |
| lamp | 3cls_s2_best | 0.0133 | 0.0094 | 0.8560 | 0.1136 | 0.6805 | 0.6257 |
| table | 3cls_s2_best | 0.0088 | 0.0084 | 0.9278 | 0.0859 | 0.7128 | 0.6771 |
| class mean (3) | 3cls_s2_best | 0.0104 | 0.0087 | 0.8970 | 0.0955 | 0.7026 | 0.6776 |
| class mean (3) | 3cls_best (run1+run2) | 0.0106 | 0.0090 | 0.8983 | 0.0982 | 0.6942 | 0.6771 |
| class mean (3) | pretrained_full | 0.0100 | 0.0090 | 0.8990 | 0.0947 | 0.7010 | 0.6800 |

- Per-class IoU, run1+run2 vs s2: chair 0.7299 vs 0.7301, lamp 0.6237 vs 0.6257, table 0.6778 vs 0.6771. The class-mean IoU is 0.6771 vs 0.6776, a difference of 0.0005.
- The class-mean IoU of both 3-class models is below the pretrained 13-class model's 0.6800 by about 0.002–0.003. Chamfer-L1 and F-Score differ more between the 3-class runs than IoU does: F-Score mean 0.6942 (run1+run2) vs 0.7026 (s2).
- These are single checkpoints on the same test split, with no seed spread. The two runs agree to within 0.001 on mean IoU, which is the same level as the earlier val-curve gap of 0.0015 (not separable from noise).

## 2026-11-05 — Dexel hits per ray (`phd_work/dexel_hits_hist.py`, CPU only)

- Ran on `voxel_3plane_3cls.yaml` for train, val and test (12328 / 1758 / 3519 objects). Output: `phd_work/results/dexel_hits_3cls.txt`. Run time 3.9 s total.
- Split list counts: `wc -l` on the `.lst` files gives 1 less per file, because the files have no trailing newline. The script parses them the way `Shapes3dDataset` does, which gives 3519 for test, matching `eval_3cls_best.log`.
- Hits are 2 × runs of consecutive occupied voxels per 32-voxel ray. Max possible is 32. Observed max is 26 (train), 26 (val), 24 (test) over all axes.
- Error: `git add` of `phd_work/results/dexel_hits_3cls.txt` was refused, because `.gitignore:23` ignores `*.txt`. Fix: force-added that one file with `git add -f`; `.gitignore` is unchanged. Other `results/` files are CSVs and are not affected by this rule.

## 2026-11-05 — Dexel slot conversion (`phd_work/dexels.py`, step D1, CPU only)

- `voxels_to_dexel_slots(vox, L)` → float32 `(3, 2L, 32, 32)`; `dexel_slots_to_voxels(slots, plane=0)` → bool `(32, 32, 32)`. The spec gave the reconstruction signature without `plane`; I added `plane` (default 0, as specified) so that the round-trip test can check all three planes.
- Plane mapping: plane 0 'xz' rays along voxel axis 1 (image = axes 0, 2); plane 1 'xy' along axis 2 (image = axes 0, 1); plane 2 'yz' along axis 0 (image = axes 1, 2). Checked on an asymmetric voxel `(3, 7, 11)` in `test_single_voxel_axis_mapping_asymmetric`. Voxel centres and `h = 1/31` follow `src/encoder/voxels.py` (`torch.linspace(-0.5, 0.5, ...)`).
- Truncation: if a ray has more than L hits, keep the first ceil(L/2) and the last floor(L/2) (L=1: first only). Tested for L=2, 4 and 1 on two runs and on the alternating column.
- Reconstruction pairs slots (2j, 2j+1) as (entry, exit). It is exact only with L = 32. It raises for odd L.
- Tests: `phd_work/tests/test_dexels.py`, 23 tests, all pass (`uv run pytest phd_work/tests -q`). They cover the cases in the spec: empty grid, single voxel, full column, two runs (L=4 and L=2), alternating column (L=32, 4, 1), axis mapping, random grids, and round trip on 100 real train models (seed 0, sampled from the three train lists). Round trip is exact on all three planes for all 100 models.
- Timing (100 real train models, binvox loading excluded, `time.perf_counter`, all three planes): L=32: 3.26 ms/object; L=4: 1.09 ms/object.
- Error: pytest was not installed (`ModuleNotFoundError`). Fix: `uv add --dev pytest` (pytest 9.2.0; the lock diff adds only pytest, iniconfig, pluggy, pygments; no existing pin changed). `pyproject.toml` gained a `[dependency-groups] dev` entry.
- Error: first test run failed on my own test, which reconstructed from an L=1 array. The function correctly rejects odd L. Fix: the empty-grid test runs the reconstruction only for even L.
