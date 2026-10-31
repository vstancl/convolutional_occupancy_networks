# ConvONet port: reproduction notes

## What this is

This directory documents a port of the ECCV 2020 code *Convolutional Occupancy Networks*
(upstream: `autonomousvision/convolutional_occupancy_networks`, baseline tag `upstream-838bea5`)
to a current stack: Python 3.12, PyTorch 2.12 (CUDA 13.0), NumPy 1.26, pandas 3, Cython 0.29.

The port changes as little as possible. Upstream model code is unchanged. Every change to an upstream
file is listed in [`CHANGES.md`](CHANGES.md), and every problem hit during the port, with its cause, fix
and effect on model behaviour, is in [`port-log.md`](port-log.md). Reference numbers come only from
[`paper_reference.md`](paper_reference.md).

Scope of the port: the voxel 3-plane pretrained model (`voxel_simple_local` encoder, `simple_local` decoder).
Other encoders and configs are not verified.

## Status

### Results vs the paper (voxel 3-plane, full ShapeNet test set)

Full test-set evaluation, 2026-10-19: 8751/8751 meshes generated and evaluated, 0 missing.
Class-mean values from `eval_meshes.py` (port-log, "Full test-set eval"). Paper values from Table 2, Ours-2D 3×64²
(`paper_reference.md`).

| Metric           | Ours   | Paper  |
| ---------------- | ------ | ------ |
| IoU              | 0.7522 | 0.752  |
| Chamfer-L1 (×10) | 0.086  | 0.092  |
| Normal consistency | 0.9086 | 0.905 |
| F-Score          | 0.7299 | 0.735  |

IoU, normal consistency and F-score match the paper. Chamfer-L1 is lower than the paper: 0.0086 vs 0.0092 raw,
about 6.5% (the port-log and `paper_reference.md` round this to 7%). The cause is not known. See
"Known deviations" below.

### Training-step benchmark (2026-10-26)

Run: `train.py phd_work/configs/voxel_3plane_bench.yaml --exit-after 600`, RTX 4070, batch 64 × 1024 points.

- 5100 iterations in 600 s.
- Median 0.1052 s/iteration (49 intervals from iteration 300, range 0.102–0.113 s). Includes data loading.
- GPU memory (`nvidia-smi`): peak 5835 MiB, idle 2348 MiB, so about 3.49 GB. This is an upper bound: it includes
  the CUDA context and allocator cache.

No paper memory figure is recorded in `paper_reference.md`, so this is not compared with the paper.
The benchmark is a speed and memory measurement, not a training-quality result.

## Requirements

Tested environment (port-log, "Environment"):

- OS: Fedora 44. GPU: NVIDIA RTX 4070 12 GB (sm_89), driver 615.78.08 (CUDA ≤ 13.4).
- GCC 16.2.1. Python 3.12.15 (installed by uv; `.python-version` pins 3.12).
- Packages, exact pins in `uv.lock`: torch 2.12.0+cu130, torchvision 0.27.0+cu130, torch-scatter 2.1.2+pt212cu130,
  numpy 1.26.4, Cython 0.29.37, pandas 3.0.6, trimesh 5.1.1.
- `uv` (tested with 0.13.0). Disk: about 73 GB for the dataset zip, about 80 GB extracted.
- A CUDA GPU for generation, evaluation of the GPU path, training and the parity GPU check. The CPU parity check
  (`--grad`) runs without a GPU.
- Platform: Linux x86_64. macOS (arm64) is read-only; see limitations.

## Reproduce

All commands are run from the repository root.

Verification status (see the port-log for details): steps 1–3, 5 and 6 were run from a fresh clone of the committed
repository. Step 4 was checked by size, a full `unzip -tq` (no errors) and archive listing; the dataset was not downloaded again and not re-extracted. Steps 7 and 8 were
run during the port; their outputs are in `phd_work/logs/` and `phd_work/results/`, and were not re-run for this document.

### 1. Clone

```bash
git clone https://github.com/vstancl/convolutional_occupancy_networks.git convonet
cd convonet
```

The default branch is `main` (the port). `master` mirrors upstream.

### 2. Python environment

```bash
export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2   # large wheels; default 30 s timed out (port-log #1)
uv sync
```

`uv sync` installs from `uv.lock`. Torch and torch-scatter come from the PyTorch and PyG indexes configured in
`pyproject.toml`, Linux only.

### 3. Build the C/C++ extensions

```bash
uv run python setup.py build_ext --inplace
```

Expected: six extensions built (`src/utils/*/…cpython-312-x86_64-linux-gnu.so`: kdtree, mcubes, triangle_hash,
mise, simplify_mesh, voxelize). The log is long (about 100 warnings) and the exit code is 0. Warnings seen in the
fresh build, all harmless: `Attempted to use ninja … could not find ninja` (torch falls back to distutils),
`Not all members given for struct`, `language_level not set, using 2`, comparisons of signed and unsigned integers,
the NumPy deprecated-API `#warning`, and narrowing conversions in the Cython output of `simplify_mesh` and `mise`.
See port-log, "Warnings kept on purpose".

### 4. Dataset

The ShapeNet data is the preprocessed set from the Occupancy Networks project. Download it into a directory
outside the repository, then link it as `data`:

```bash
mkdir -p ~/PhD/data/onet && cd ~/PhD/data/onet
wget http://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip
```

Check the size. It must be exactly this many bytes (`73419791657`):

```bash
test "$(stat -c %s dataset_small_v1.1.zip)" -eq 73419791657 && echo "size OK"
```

Check the archive and extract it. Extraction creates `ShapeNet/` in the same directory:

```bash
unzip -tq dataset_small_v1.1.zip     # expect "No errors detected"; reads the whole file, takes a long time
unzip -q dataset_small_v1.1.zip
```

Link the dataset into the repository (the configs read `data/ShapeNet`):

```bash
cd /path/to/convonet
ln -s ~/PhD/data/onet data
ls data/ShapeNet            # 13 class directories and metadata.yaml
```

The `data` symlink is ignored by git. The dataset has 8751 test objects (port-log, "Dataset").

### 5. Smoke test (no dataset needed)

```bash
uv run python phd_work/smoke.py --device cuda
```

Loads the pretrained voxel 3-plane checkpoint (downloaded from the URL in its config on first use, then cached),
runs one forward pass on random input, and prints `OK: all logits finite`.

### 6. Parity checks

Reference outputs are committed in `phd_work/parity/`. They were captured from commit `d4789b2`
(before the `torch_scatter` replacement). The check passes on the current code, so the replacement does not change
the output.

```bash
uv run python phd_work/parity.py compare --device cuda        # encoder features and logits, GPU
uv run python phd_work/parity.py compare --grad --device cpu  # plus gradients of encoder.conv_in and decoder.fc_p, CPU
CUDA_VISIBLE_DEVICES= uv run python phd_work/parity.py compare --grad --device cpu  # same, with the GPU hidden (no GPU needed)
```

Each prints `max|diff|` per tensor and `PARITY OK` or `PARITY FAIL`. The exit code is 0 or 1.
Do not compare across devices: CUDA and CPU differ by about 0.03 in logits, and the script refuses the mismatch.

To regenerate a reference, run `capture` (or `capture --grad --device cpu`) on a clean tree. `capture` refuses to run
if `src/` or `configs/` have uncommitted changes.

### 7. Full test set: generate and evaluate

```bash
uv run python generate.py configs/voxel/pretrained/shapenet_3plane.yaml
uv run python eval_meshes.py configs/voxel/pretrained/shapenet_3plane.yaml
```

Outputs go to `out/voxels/shapenet_3plane/generation_pretrained/` (meshes, `eval_meshes.csv`,
`eval_meshes_full.pkl`). Generation over 8751 objects takes hours. The port-log does not record the wall time.
The evaluation of the full run is committed as `phd_work/results/voxel_3plane_pretrained_full.csv`.

Quick version on 26 objects: `uv run python generate.py phd_work/configs/voxel_3plane_smoke.yaml`, then
`uv run python eval_meshes.py phd_work/configs/voxel_3plane_smoke.yaml`.

### 8. Training-step benchmark

```bash
uv run python train.py phd_work/configs/voxel_3plane_bench.yaml --exit-after 600
nvidia-smi --query-gpu=memory.used --format=csv -l 5    # in a second terminal, for the GPU-memory numbers
```

Writes to `out/phd_bench/`. Logs from the reference run: `phd_work/logs/bench_train.log`, `phd_work/logs/bench_gpumem.csv`.

## Known deviations and limitations

- **Chamfer-L1 is about 6.5% below the paper** (0.0086 vs 0.0092 raw). IoU, normal consistency and F-score match.
  Cause not known. Candidates: trimesh surface sampling, and paper-run vs released-checkpoint differences. The locked
  trimesh is 5.1.1 (an earlier "trimesh 4.x" note was wrong). The decisive test (same meshes in the legacy environment)
  is not done.
- **`torch_scatter` is still required.** `src/encoder/__init__.py` imports `pointnet` unconditionally, and
  `pointnet.py` imports `torch_scatter` at module level. So `import src` fails without it. The voxel encoder no
  longer uses it. `pointnet.py` is not ported; porting it needs its own parity reference.
- **macOS: read-only.** `pykdtree` is compiled with `-fopenmp`, which Apple clang does not support, and
  `torch_scatter` is not installed on macOS. Mac is for reading code and logs only.
- **GPU results are not reproducible by default.** Two identical GPU runs of unchanged code differed by about 0.02 in
  encoder features (scatter atomics and cuDNN). The parity harness turns on deterministic mode. `generate.py`,
  `eval_meshes.py` and `train.py` do not, so their outputs can differ slightly from run to run.
- **GPU backward of `scatter_reduce_` is not parity-tested.** The gradient check runs on CPU, where it is
  bit-identical to the old code. CUDA backward of `grid_sampler_2d` has no deterministic implementation, so it cannot
  be compared exactly.
- **Parity scope.** Checked: the voxel 3-plane path (`voxel_simple_local` encoder, `simple_local` decoder). Not checked:
  the pointnet and PointNet++ encoders, the 1-plane and grid32 configs, and training beyond the benchmark run.
- **Full training to convergence was not run.** Only the 600 s benchmark.
- **Warnings kept.** Cython `language_level` (upstream was built with Python 2 semantics), and the upstream
  `contains1 != contains2` warning in evaluation (benign, seen twice in the full run).

## Data license

- **Upstream code:** MIT, Copyright (c) 2020 Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, Andreas Geiger
  (`LICENSE`).
- **Port code in `phd_work/`:** licence as stated by the author in the previous version of this file (MIT). Confirm before
  relying on it; no `LICENSE` file covers this directory.
- **Data:** ShapeNet is distributed under its own terms of use (shapenet.org). The preprocessed dataset used here is the
  one published by the Occupancy Networks project. This repository does not include any data. Read the current
  ShapeNet terms before use or redistribution; this README does not summarise them.
- **Pretrained model:** the checkpoint URL in `configs/voxel/pretrained/shapenet_3plane.yaml` is the upstream
  release. Its licence is not stated in the repository.
