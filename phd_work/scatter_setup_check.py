"""[phd] Setup check for the weekend chain: the scatter_mean in use (run via phd_work/run.py).

Prints the function object and the SCATTER env value; raises if they disagree.
"""
import os

import src.encoder.voxels as voxels_mod

choice = os.environ.get('SCATTER', 'native')
name = voxels_mod.scatter_mean.__qualname__
print('SCATTER=%s; scatter_mean in use: %s from %s' % (choice, name, voxels_mod.scatter_mean.__module__))
want = 'scatter_mean_upstream' if choice == 'torch_scatter' else 'scatter_mean'
if not name.endswith(want):
    raise RuntimeError('SCATTER=%s but scatter_mean in use is %s' % (choice, name))
