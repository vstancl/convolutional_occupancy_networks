"""[phd] Switch the voxel-encoder scatter between the upstream and the native implementation.

After the revert of ca728fb, src/encoder/voxels.py imports torch_scatter.scatter_mean (upstream
at tag upstream-838bea5). generate_plane_features looks up the module-level name `scatter_mean`
at call time, so replacing the module attribute changes every plane scatter (voxel and dexel
encoders) without editing upstream files.

- upstream (default): scatter_mean_upstream(src, index, out) = torch_scatter.scatter_mean(src, index, out=out)
  (dim=-1, the default), the call at tag upstream-838bea5.
- native (SCATTER=native): scatter_mean_native(src, index, out) = out.scatter_reduce_(...,
  reduce='mean', include_self=False), the implementation of ca728fb. Slower in training:
  0.159 vs 0.108 s/it (P7, P8); kept for comparison only.

apply_from_env() reads SCATTER=torch_scatter|native (default torch_scatter) and is called by
phd_work/run.py, phd_work/profile_step.py and phd_work/scatter_setup_check.py.
"""
import os

import src.encoder.voxels as _voxels


def scatter_mean_upstream(src, index, out):
    import torch_scatter

    return torch_scatter.scatter_mean(src, index, out=out)


def scatter_mean_native(src, index, out):
    return out.scatter_reduce_(-1, index.expand_as(src), src, reduce='mean', include_self=False)


def use_torch_scatter():
    _voxels.scatter_mean = scatter_mean_upstream


def use_native():
    _voxels.scatter_mean = scatter_mean_native


def current():
    return 'native' if _voxels.scatter_mean is scatter_mean_native else 'torch_scatter'


def apply_from_env():
    choice = os.environ.get('SCATTER', 'torch_scatter').strip()
    if choice == 'torch_scatter':
        use_torch_scatter()
    elif choice == 'native':
        use_native()
    else:
        raise ValueError('SCATTER must be torch_scatter or native, got %r' % choice)
    return choice
