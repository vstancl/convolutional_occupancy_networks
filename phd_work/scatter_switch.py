"""[phd] Switch the voxel-encoder scatter between the native and the torch_scatter implementation.

src/encoder/voxels.py (after ca728fb) defines a module-level scatter_mean(src, index, out) that
uses scatter_reduce_. generate_plane_features looks that name up at call time, so replacing the
module attribute changes every plane scatter (voxel and dexel encoders), without an upstream edit.

use_torch_scatter() installs the upstream call: torch_scatter.scatter_mean(src, index, out=out)
(dim=-1, the default), as at tag upstream-838bea5. use_native() restores the current one.

apply_from_env() reads SCATTER=torch_scatter|native (default native) and is called by
phd_work/run.py and phd_work/profile_step.py.
"""
import os

from src.encoder import voxels as _voxels

_NATIVE = _voxels.scatter_mean


def use_native():
    _voxels.scatter_mean = _NATIVE


def use_torch_scatter():
    import torch_scatter

    def scatter_mean_upstream(src, index, out):
        return torch_scatter.scatter_mean(src, index, out=out)

    _voxels.scatter_mean = scatter_mean_upstream


def current():
    return 'torch_scatter' if _voxels.scatter_mean is not _NATIVE else 'native'


def apply_from_env():
    choice = os.environ.get('SCATTER', 'native').strip()
    if choice == 'torch_scatter':
        use_torch_scatter()
    elif choice == 'native':
        use_native()
    else:
        raise ValueError('SCATTER must be torch_scatter or native, got %r' % choice)
    return choice
