# h5 survey: interfaces for ray decoders

Survey only. No model code was written or changed. Read-only access to the repositories and data below.

Paths:

- `csgbox/` = `~/PhD/csgbox` (generator, ceilings, gate; branch `dev`)
- `dexelize/` = `~/PhD/dexelize` (`python/dexio.py`)
- `convonet/` = `~/PhD/convonet` (ConvONet fork)
- `data/` = `~/PhD/data/csgbox/v1`

Where the code says nothing, the answer is marked **ambiguous** or **not defined**, and no interpretation is chosen here.

---

## Q1. `csgbox.ceilings`, `hits_8`: slot layout

**What exists in code.** `hits_L` is a *variable-length CSR truncation* of the ground-truth (GT) hits, not a fixed-slot layout.

- `csgbox/csgbox/ceilings.py:88-100` `decode_hits(gt, L)`:
  - `t32 = gt.t.astype(float32).astype(float64)` (line 95): GT hits rounded to float32.
  - `keep = (rank < L//2) | (rank >= counts - L//2)` (line 99): the first `L/2` and the last `L/2` hits of each ray, by rank.
  - Output is `_csr(...)` (line 100): a new `RayHits` (CSR, sorted, float64 `t`, float32 is kept as values).
- Module docstring, `ceilings.py:6`: "the first and last `L/2` hits per ray kept."
- Decision 44 (`csgbox/docs/decisions.md:292-297`): the same rule; "a ray with at most `L` hits is kept whole".

**Per-n behaviour** (from the code, `ceilings.py:99`, `L = 8`):

| n (hits) | ranks kept (CSR) | intervals kept |
|---|---|---|
| 0 | none | none |
| 2 | 0, 1 | all |
| 4 | 0..3 | all |
| 6 | 0..5 | all |
| 8 | 0..7 | all |
| 10 | 0..3, 6..9 | drops interval 2 (ranks 4, 5) |
| 12 | 0..3, 8..11 | drops intervals 2, 3 (ranks 4..7) |

1. **Are intervals duplicated between the first-4 and last-4 halves when n < 8?** In `decode_hits` **no**. The boolean `keep` is evaluated once per hit, so a hit with rank in both halves is stored once (`ceilings.py:99`). The CSR count is therefore `min(n, 8)`, not 8. A *fixed-slot* layout would have to duplicate or mask, but no such layout is in the code (see below).
2. **Slot layout (which hit goes to which slot).** **Not defined in code.** The only slot language is:
   - `csgbox/SPEC.md:24` (Czech table): "na paprsek L slotů: offset hloubky k buňce + maska" = "per ray L slots: depth offset to the cell + mask".
   - `csgbox/docs/decisions.md:348-352` (decision 50): "the t offset and the mask of each slot", `3 · N² · L · GATE_HIT_NETWORK_FIELDS` values (`csgbox/csgbox/constants.py:133-134`, `gate.py:66-74`).
   - No code assigns a hit to a slot index, and no code writes a mask or an offset.
3. **"Depth offset to cell": reference cell, units, range.** **Ambiguous; not defined anywhere.** Candidate readings, not chosen:
   - (a) `t - BOX_LO` in world units (range `[0, 1.10001]`);
   - (b) `t` relative to the lower face of the cell that contains it, `(t - cell_lo)`, in world units (range `[0, h)`, `h = 1.10001/128 = 0.0085938`);
   - (c) the same as (b) divided by `h` (range `[0, 1)`).
   The cell index itself is not stored anywhere in the CSR. Decision 50 does not say which reading it means.
4. **How masks are defined.** **Not defined.** Presumably "1 if the slot holds a hit", but that is not written down.
5. **How slots decode back to hits.** **Not defined.** There is no inverse function. The CSR decoders (`decode_tsdf`, `decode_binary`, `decode_occfrac`, `decode_points`) output `RayHits`; none accepts (offset, mask) pairs. Decoding slots would also need the cell index, which is not stored (see 3).

**Consequence for H5:** the survey's requested answer ("exact slot layout") cannot be given from the code. The layout (front/back halves, alignment, duplication, the offset reference, the mask) has to be decided by Wik and written into `csgbox/docs/decisions.md` before any `hits_8` decoder is built. Section Q8 gives a *stated reading* that was used to count slot rates.

---

## Q2. `occfrac_128`, `tsdf_128`, `binary_128`: per-cell values, decoders, subset calls

Notation: `M = 128`, `h = cell_size(128) = (BOX_HI - BOX_LO)/128 = 1.10001/128 = 0.008593828` (`csgbox/csgbox/grid.py:28-30`; constants at `csgbox/csgbox/constants.py:19-20`).

Sample grid: `z_k = BOX_LO + (k + 1/2) h`, `k = 0..M-1` (`ceilings.py:108-111`). **Cell centres, point samples.**

### `tsdf_128`: point samples, signed clamped distance

- Encoder: `ceilings.py:146-152` `tsdf_values(gt, m) -> (values float32 (R, M), z float64 (M,), h float)`.
  - Inside flag: **parity** of the number of GT hits with `t < z_k` (`_signed_distance`, `ceilings.py:114-131`, parity at line 125). Odd = inside.
  - Distance: the **1-D distance along the ray** to the nearest GT hit (`ceilings.py:131`; `min` of the next and previous hit). It is not a Euclidean distance.
  - Clamp: `dist = min(dist, GATE_TSDF_TAU_CELLS * h)` (line 150), with `GATE_TSDF_TAU_CELLS = 2.5` (`constants.py:119`). A ray without hits gets `inf` and so the clamped value.
  - Sign: `-dist` if inside, `+dist` otherwise (line 151). Cast to float32 (line 151).
  - Output range: `[-2.5 h, +2.5 h]` = `[-0.0215, +0.0215]` at M = 128. **Not** normalized to `[-1, 1]`. The SPEC asks for `[-1, 1]` with L1 loss (`csgbox/SPEC.md` §2, TSDF column). The normalization is not implemented, and it is **ambiguous** whether it means `/ (2.5 h)`.
- Decoder: `ceilings.py:155-164` `decode_tsdf(values, z, h) -> RayHits`. Hits at sign changes between consecutive samples; `t = z_k + h v_k/(v_k - v_{k+1})` (line 162); `enter = outside before the crossing` (line 163). Decision 45 (`decisions.md:299-306`).
- Subset of K rays: **yes** (see the "Subsets" note below).

### `binary_128`: point samples, occupancy

- Encoder: `ceilings.py:182-186` `binary_occupancy(gt, m) -> bool (R, M)`. Same parity rule as tsdf (`inside`), at the same centres. No sign, no truncation.
- Decoder: `ceilings.py:172-179` `decode_binary(gt, m) -> RayHits`. Each transition is decoded at the **midpoint** `t = (z_k + z_{k+1})/2` (line 177), which is the cell boundary `BOX_LO + (k+1) h`. `enter` = outside before (line 178). Decision 47 (`decisions.md:317-322`).
- Subset: **yes**.

### `occfrac_128`: cell integral (fraction of the cell that is material)

- Encoder: `ceilings.py:227-256` `occ_fraction(gt, m) -> float64 (R, M)`. Cell `k` is `[BOX_LO + k h, BOX_LO + (k+1) h)`. **Cell integral**, not a centre sample: each GT material interval `[enter, next hit]` is spread over the cells it overlaps (line 253 overlap, line 254 `frac = overlap / cell length`), summed per cell (line 255), clipped to `1.0` (line 256). A cell covered by one interval over its full length gets exactly `1.0` (docstring lines 230-232). The values are cast to float32 by the caller (`gate.py:86`).
- Decoder: `ceilings.py:259-299` `decode_occfrac(f: (R, M)) -> RayHits`. Best effort (decision 49, `decisions.md:333-346`):
  - neighbour states `s = f > GATE_OCC_THRESHOLD` (`constants.py:131`, `0.5`; exactly 0.5 is outside, line 271);
  - partial cells `0 < f < 1` (line 272) give one or two hits as stated in the docstring (lines 262-265; formulas at 276-288);
  - cells with `f` in `{0, 1}` give a transition at the shared boundary of two neighbours (lines 290-294).
  - The box ends count as outside (padding at line 269).

### Subsets of K rays of one shape

- **Yes** for `tsdf_values`, `binary_occupancy`, `occ_fraction`. They use only `gt.n_rays`, `gt.offsets`, `gt.t`, `gt.enter` and `gt.ray_of()`. Nothing refers to the 128 × 128 grid, so a `RayHits` with K rays works, and the encoders are vectorized over rays and samples.
- **No helper exists** to take K rays out of a `RayHits`. A subset helper would be needed, e.g.:
  ```python
  def subset_rays(gt: RayHits, rays: np.ndarray) -> RayHits:
      """Rays `rays` (int array) of gt, offsets re-based; enter, t, normal sliced in order."""
  ```
  It is a few lines of numpy (`np.diff` for counts, `np.repeat` to build the hit index, `np.concatenate` of the per-ray slices, or a boolean `np.isin(ray_of, rays)` mask with `np.cumsum`). It is not in the code and nothing was written here.
- **Input from the data**: `dexelize`'s `AxisArrays` gives `offsets` (int64), `t` (**float32**), `enter` (uint8). `RayHits` needs float64 `t` and bool `enter`. The gate computes the GT in float64 from the tree (`ceilings.py:76-80`); training from `dexels_128.npz` sees the float32 `t` of the file. The difference is at the float32 rounding level (about `3e-8` at `|t| = 0.5`), which is well below `h`, but it is not bit-identical to the gate.
- **Function signatures** (all in `csgbox/csgbox/ceilings.py`):
  ```python
  tsdf_values(gt: RayHits, m: int) -> tuple[np.ndarray, np.ndarray, float]   # (R,M) float32, z (M,), h
  binary_occupancy(gt: RayHits, m: int) -> np.ndarray                       # (R,M) bool
  occ_fraction(gt: RayHits, m: int) -> np.ndarray                           # (R,M) float64
  decode_tsdf(values: np.ndarray, z: np.ndarray, h: float) -> RayHits
  decode_binary(gt: RayHits, m: int) -> RayHits
  decode_occfrac(f: np.ndarray) -> RayHits                                  # f (R,M)
  ```

---

## Q3. Pairing and metrics

All of these work on `RayHits` (`ceilings.py:47-65`): `offsets` int64 `(R+1,)`, `t` float64 `(H,)` sorted within each ray, `enter` bool `(H,)`, and `normal` float64 `(H, 3)` (needed only by `thin_pairs`). CSR, one ray index per hit. No per-ray Python loop.

- **Matching**: `ceilings.py:314-351` `match(gt, dec, delta) -> Match(gt_matched, dec_matched, dt)`. Two-pointer greedy per ray, vectorized over rays. A pair is matched if the enter types agree and `|dt| <= delta`. Decision 46 (`decisions.md:308-315`). `delta` values: `GATE_HIT_DELTA = 0.002` (`constants.py:127`) and `h(128)/2`.
- **Precision, recall, F1**: `scripts/gate_ceilings.py:93-95`: `prec = matched / dec_hits`, `rec = matched / gt_hits`, `F1 = 2 prec rec / (prec + rec)`. The counts come from `evaluate_axis` (`ceilings.py:382-403`, `matched` at line 397).
- **|dt|**: mean from `dt_sum / dt_n` (`evaluate_axis`, lines 398-399); p95 from a histogram of `GATE_DT_BINS = 900` log10 bins over `[1e-9, 1]` (`dt_histogram`, `ceilings.py:376-379`; `constants.py:142`). Decision 46.
- **Hit-count error**: `count_err_sum = Σ_rays |n_dec - n_gt|` (`ceilings.py:387`).
- **Thin features (recall by class and L/h bin)**: `thin_pairs(gt, axis)` (`ceilings.py:359-368`) returns the first hit index `k` of each GT thin feature (a material interval or gap between hits `k` and `k+1` on one ray, shorter than `4 h(128)`), its length, and a class index (wall, gap, grazing). The class uses `cos = min(|n_k[axis]|, |n_{k+1}[axis]|)` (line 367) and `THIN_WALL_MIN_COS = 0.5` (`thin.py`, decision 36). `thin_bins(length, h)` (lines 371-373) uses `GATE_BIN_EDGES` (`constants.py:137`). Recall of a feature = both its bounding GT hits matched (`evaluate_axis`, line 393).
- **Thin recall by bin, with a per-ray pairing**: the `k` index is into the *flat* hit array of the axis. From a `dexio` file, the ray of feature `k` is `np.searchsorted(offsets, k, side="right") - 1`.

**Is `thin.npz` needed?** **No.** `thin_pairs` recomputes the thin population from the GT normals. The table in `thin.npz` has members `length`, `cos` (float32), `axis`, `kind` (uint8) (confirmed on shape `00000`, 4845 features). It has **no ray index and no hit index**, so it cannot map a thin feature to its ray (see Q4). The dexel file already carries normals (`x_normal`, `y_normal`, `z_normal`, float32), which are enough for `thin_pairs`. The only difference is float32 normals against float64 normals, which could move a feature across `cos = 0.5` in rare cases.

**Runtime.** Measured on shape `csg/00000` (52 766 GT hits over 3 × 128² rays), `csgbox.gate.process_shape` (all 10 representations, both deltas, all 3 axes, plus the manifold surface area): **3.2 s wall time** (44 s CPU, multithreaded), in `csgbox`'s venv. The matching alone was not timed separately. Measured, not extrapolated.

---

## Q4. Data: loading `dexels_128.npz`, ray index, axes, box, cell centres, thin mapping

**Loading.** `dexio.read(path) -> DexelFile` (`dexelize/python/dexio.py:350-385`; dataclasses `AxisArrays` at line 54 and `DexelFile` at line 66). `DexelFile.header` is a dict (`N`, `lo`, `hi`, `axes`, `has_normals`, `eps`, `sampler`, ...). `DexelFile.axes[a]` for `a` in `x, y, z` is an `AxisArrays`:

| member | field | dtype | shape |
|---|---|---|---|
| `{a}_offsets` | `offsets` | int64 | `(N² + 1,)` = `(16385,)` |
| `{a}_t` | `t` | float32 | `(K,)` (hits of the axis) |
| `{a}_enter` | `enter` | uint8 (1 = enter) | `(K,)` |
| `{a}_normal` | `normal` | float32 | `(K, 3)` |
| `{a}_start_inside` | `start_inside` | uint8 | `(N²,)`, all zero |
| `{a}_flags` | `flags` | uint8 | `(N²,)` (bit 4 = no hit, bit 0 = invalid) |

Observed on shape `00000`: x: 17 850 hits, y: 16 784, z: 18 132. `dexio.validity_flags` (line 220) and the box test are run on read.

**Ray index.** `r = i_col + N · i_row` (`csgbox/csgbox/grid.py:47-53`, `ray_grid`; the flat index is `r % N = i_col`, `r // N = i_row`). The hits of ray `r` are `t[offsets[r]:offsets[r+1]]`.

**Which world axes are `col` and `row`** (`csgbox/csgbox/grid.py:39-44`, `remaining_axes`: `(col, row)` = the two remaining axes, lower first):

| ray axis | `col` (`i_col`, `r % N`) | `row` (`i_row`, `r // N`) | ConvONet plane |
|---|---|---|---|
| `x` (axis 0) | `y` | `z` | `yz` |
| `y` (axis 1) | `x` | `z` | `xz` |
| `z` (axis 2) | `x` | `y` | `xy` |

Each ray has origin coordinate 0 on its own axis, direction `+e_a`, and `t` is the absolute world coordinate along `a` (`grid.py` docstring; decision 22).

**Box and cell centres.** `lo = -0.550005`, `hi = 0.550005` (`constants.py:19-20`; the header's `lo`/`hi` are the same). `N = 128`, `h = 1.10001 / 128 = 0.00859383`. Centres `u_k = lo + (k + 1/2) h` (`grid.py:33-36`, `cell_centres`). Ray `r` on axis `a` has its `col` and `row` coordinates at `u[r % N]` and `u[r // N]`.

**Does `thin.npz` map a thin feature to its ray?** **No.** Members (`length` float32, `cos` float32, `axis` uint8, `kind` uint8) have no ray index. It can be mapped **from `dexels_128.npz`**: the GT consecutive hits `k, k+1` on a ray with `t[k+1] - t[k] < 4h` give the thin feature, and `k` maps to the ray by `offsets` (see Q3).

---

## Q5. ConvONet decoder: plane sampling and the normalization

**Upstream code** (`convonet/`, this fork):

- `src/common.py:235-257` `normalize_coordinate(p, padding=0.1, plane='xz')`:
  - pair: `xz -> (x, z)` (lines 243-244), `xy -> (x, y)` (lines 245-246), `yz -> (y, z)` (lines 247-248, the `else`);
  - `xy_new = xy / (1 + padding + 10e-6)` (line 250). Note `10e-6 = 1e-5`;
  - `xy_new = xy_new + 0.5` (line 251);
  - clamps: `>= 1` becomes `1 - 10e-6` (line 255), `< 0` becomes `0` (line 257). Silent.
- `src/conv_onet/models/decoder.py:51-56` `LocalDecoder.sample_plane_feature(p, c, plane)`:
  ```python
  xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)   # (B, T, 2) in (0, 1)
  xy = xy[:, :, None].float()
  vgrid = 2.0 * xy - 1.0                                                    # (B, T, 1, 2) in (-1, 1)
  c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
  ```
- `decoder.py:67-79` `LocalDecoder.forward(p, c_plane)`: sums the plane samples in the order `grid`, `xz`, `xy`, `yz` (lines 71-78; `c` starts as 0), then `c = c.transpose(1, 2)` (line 79). The grid term is not used by the 3-plane config.
- `padding` comes from `cfg['data']['padding']` (`src/conv_onet/config.py:29`), passed to the decoder at `config.py:64`. The 3-plane config does not set it, so it is `0.1` from `configs/default.yaml` (`data.padding`), merged by `load_config(args.config, 'configs/default.yaml')` (`train.py:26`).

**Exact formula for a world point `p = (x, y, z)` and plane `P`:**

With `s = 1 + padding + 1e-5` (= `1.10001` at `padding = 0.1`), and `(a, b)` = `(x, z)` for `xz`, `(x, y)` for `xy`, `(y, z)` for `yz`:

```
g_a = a / s + 0.5,   g_b = b / s + 0.5          # then: g >= 1 -> 1 - 1e-5 ;  g < 0 -> 0
vgrid = (2 g_a - 1, 2 g_b - 1)                  # vgrid[..., 0] is the first coordinate
F.grid_sample(c, vgrid, align_corners=True, padding_mode='border', mode=sample_mode)
```

The plane tensor is `c[b, ch, row, col]`. The encoder writes a point to flat index `i_a + R · i_b` with `i_a = floor(g_a R)` and `i_b = floor(g_b R)` (`src/common.py:300-313`, `coordinate2index`, line 311; `src/encoder/pointnet.py:69-80`, reshape at line 78). So the first plane coordinate is the **column** (`dim 3`) and the second is the **row** (`dim 2`). `grid_sample`'s `vgrid[..., 0]` is the column, which matches.

**Recommendation.** A ray decoder should call `LocalDecoder.sample_plane_feature(p, c, plane)` (or the same `normalize_coordinate` followed by `2g - 1`) rather than re-deriving the formula, so both decoders read the identical positions.

**Two facts a ray decoder must know** (not bugs in this survey, but they change the result):

1. **Encoder cells and sampler nodes are not aligned.** The encoder puts a point into cell `floor(g R)`, whose centre is `g = (i + 1/2)/R`. With `align_corners=True`, `grid_sample` reads pixel `i` at `g = i / (R - 1)`. So the centre of cell `i` is read at pixel position `(i + 1/2)(R-1)/R`, which is a stretch by `(R-1)/R` and a shift of up to about half a pixel (`0.49` pixel at `i = 0`, `-0.49` at `i = R-1` for `R = 64`). This is upstream behaviour, shared by the point decoder. It is not changed here.
2. **On a ray along axis `a`, the plane of `a` is constant.** For a `z`-ray, the `xy` plane is sampled at the fixed `(x, y)` of the ray for every `t`. Only `xz` and `yz` vary along `t`. The ray decoder's dependence on `t` comes from the other two planes.

---

## Q6. Upstream pointcloud config: `pointnet_local_pool`, 3 planes, 64², `c_dim` 32, U-Net

**Config:** `convonet/configs/pointcloud/shapenet_3plane.yaml` (the 128² variant is `room_3plane.yaml`, `plane_resolution: 128`, so it does not match). The relevant keys:

| key | line | value |
|---|---|---|
| `data.input_type` | 3 | `pointcloud` |
| `data.pointcloud_n` | 6 | 3000 |
| `data.pointcloud_noise` | 7 | 0.005 |
| `data.points_subsample` | 8 | 2048 |
| `data.padding` | (inherited from `configs/default.yaml`, line 30) | 0.1 |
| `model.encoder` | 13 | `pointnet_local_pool` (`src/encoder/__init__.py:7`) |
| `model.encoder_kwargs.hidden_dim` | 15 | 32 |
| `model.encoder_kwargs.plane_type` | 16 | `['xz', 'xy', 'yz']` |
| `model.encoder_kwargs.plane_resolution` | 17 | 64 |
| `model.encoder_kwargs.unet` | 18 | `True` |
| `model.encoder_kwargs.unet_kwargs` | 19-22 | `depth: 4`, `merge_mode: concat`, `start_filts: 32` |
| `model.decoder` | 23 | `simple_local` (`LocalDecoder`, `src/conv_onet/models/__init__.py:7-8`) |
| `model.decoder_kwargs` | 24-26 | `sample_mode: bilinear`, `hidden_size: 32` |
| `model.c_dim` | 27 | 32 |
| `training.batch_size` | 30 | 32 |

Decoder `n_blocks` and `leaky` are left at their defaults (5, False). The pretrained variant is `configs/pointcloud/pretrained/shapenet_3plane.yaml`.

---

## Q7. Environment: are `csgbox` and `dexio` importable from the ConvONet venv?

**No.** Checked in `convonet/.venv` (Python 3.12.15, numpy 1.26.4, torch 2.12, torch_scatter 2.1.2):

- `import dexio`: `ModuleNotFoundError`.
- `import csgbox`: `ModuleNotFoundError`.

The **csgbox** venv (`csgbox/.venv`, numpy 2.5.5, pytest, manifold3d 3.5.5) has `dexio` 0.1.0 installed editable from `dexelize/python`. The csgbox pyproject pins `requires-python = ">=3.12,<3.13"` and `numpy>=2.5.5` (`csgbox/pyproject.toml:4-6`), so the ConvONet venv (numpy 1.26.4) is outside that pin.

**Proposal (not executed):**

1. **dexio (read-only data access):** `PYTHONPATH=$HOME/PhD/dexelize/python` when running ConvONet code. dexio is one module that needs only numpy, and it states support for numpy 1.26 and 2.x (`dexio.py` docstring). No change to ConvONet's `pyproject.toml` or `uv.lock`. The alternative is `uv pip install -e ~/PhD/dexelize/python` into the venv, which changes the venv but not the lock.
2. **csgbox ceilings (interfaces only):** `PYTHONPATH=$HOME/PhD/csgbox`. The modules used by the decoders (`ceilings`, `grid`, `thin`, `tree`, `intervals`, `primitives`, `constants`) do not import `mesh` or `manifold3d`, so `manifold3d` is not needed for them. **Untested** with numpy 1.26, and the pyproject pin says numpy 2.5.5. Do not add csgbox to the ConvONet dependencies without approval (the dependency rule in `csgbox/CLAUDE.md`; a new dependency needs the user's approval).
3. Do not add `h5` to the `phd_work/` rule without a decision: `convonet/CLAUDE.md` says "All new code goes into `phd_work/`", while this survey was placed in `h5/` at the user's request. See "Open items".

---

## Q8. Ray statistics (`h5/scripts/ray_stats.py`)

Script: `h5/scripts/ray_stats.py`. Selects 200 training shapes (`status ok`, `split train`, from `data/generation.jsonl`, sorted by id) with `numpy.random.default_rng(0).choice(..., replace=False)`. Reads each `dexels_128.npz` with `dexio`. numpy and dexio only; nothing from csgbox, ConvONet or `phd_work`. Output is byte-identical on two runs. Full output: `/tmp/claude-1000/-home-wish-PhD-csgbox/c184cd15-2d39-4fdf-a630-aba76061ffad/scratchpad/ray_stats_output.txt`.

Rays per axis: 200 × 128² = **3 276 800**.

**Hit-count fractions**, per axis:

| axis | 0 | 2 | 4 | 6 | 8 | >8 | odd |
|---|---|---|---|---|---|---|---|
| x | 0.610189 | 0.280612 | 0.066896 | 0.025058 | 0.006508 | 0.010738 | 0 |
| y | 0.625109 | 0.266547 | 0.065879 | 0.025056 | 0.006734 | 0.010675 | 0 |
| z | 0.624072 | 0.265862 | 0.067309 | 0.025608 | 0.006245 | 0.010904 | 0 |

No odd hit counts occur, so every ray is valid and ends outside.

**Slot positive rate, CURRENT `hits_8` layout** (inferred reading, see Q1; slots 0-3 = ranks 0-3 from the front, valid if rank < n; slots 4-7 = ranks n-4..n-1 right-aligned, valid if rank ≥ 0):

| axis | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |
|---|---|---|---|---|---|---|---|---|
| x | 0.389811 | 0.389811 | 0.109199 | 0.109199 | 0.109199 | 0.109199 | 0.389811 | 0.389811 |
| y | 0.374891 | 0.374891 | 0.108344 | 0.108344 | 0.108344 | 0.108344 | 0.374891 | 0.374891 |
| z | 0.375928 | 0.375928 | 0.110067 | 0.110067 | 0.110067 | 0.110067 | 0.375928 | 0.375928 |

**Slot positive rate, TWO-SIDED layout** (slots 0-1 = intervals 0 and 1 from the front; slots 2-3 = intervals K-1 and K-2 from the back, with K = n/2; valid if the interval exists):

| axis | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| x | 0.389811 | 0.109199 | 0.389811 | 0.109199 |
| y | 0.374891 | 0.108344 | 0.374891 | 0.108344 |
| z | 0.375928 | 0.110067 | 0.375928 | 0.110067 |

**Check: both layouts decode the same hit set on every ray.** **Passed**: 0 rays differ on any axis, and 0 rays decode a split interval in the current layout. For even n, the current layout keeps ranks `0..3` and `n-4..n-1`, which are the first two and last two intervals. So the two layouts decode the same intervals. The check is therefore true by construction for even n, and it confirms that the two-sided layout does not lose information relative to the current one. It does not distinguish them when n > 8: both drop the same middle intervals.

**Duplicates.** Under both layouts, the rays with duplicates are exactly those with 2 ≤ n ≤ 6 (K < 4), since the two halves overlap. Per axis (x): 1 220 822 rays have duplicates under the current layout (2 880 054 duplicate slots) and under the two-sided layout (1 440 027 duplicate slots). The two counts of duplicate rays match on every axis.

**Truncation**: rays with more than 8 hits, which lose middle hits: x 35 185, y 34 981, z 35 730.

---

## Open items (need Wik's decision; nothing chosen here)

1. **`hits_8` slot layout** (Q1): front/back split, right- or left-alignment of the back half, duplication for n < 8, the meaning of "depth offset to cell" (reference cell, units, range), and the mask. None is defined in code. Write the decision in `csgbox/docs/decisions.md` before implementing a decoder.
2. **TSDF normalization** to `[-1, 1]` (SPEC §2): not implemented in `csgbox`; the scale (`2.5 h`?) is undecided.
3. **Sampler alignment** (Q5, fact 1): upstream `grid_sample` with `align_corners=True` vs the encoder's cell scatter. Shared by all decoders; not a choice for H5 alone.
4. **Repository rule conflict**: `convonet/CLAUDE.md` says new code goes in `phd_work/`, push after each commit, and work on `main`. The request placed the work in `h5/` and said not to push. The request was followed.
5. **Environment** (Q7): the decision to put dexio/csgbox on the ConvONet path (PYTHONPATH or install), and the numpy version pin.
