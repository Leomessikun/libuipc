"""Recorded sensor interventions; never modify ground-truth cloth or success geometry."""
from __future__ import annotations

import numpy as np


def perturb_observation(observation, spec, *, noise_m=0., dropout=0., seed=0,
                        body=0, replica=0, frame=0, voxel_shift=None, voxel_size=.0625):
    """Perturb camera points, protecting goal/tool points and proprioception.

    The RNG is indexed by case and frame, independent of policy, wall time and
    completion of other slots. Voxel shift changes selection only, not coordinates.
    If set, it replaces the bridge's downsampling (the caller must use voxel=0).
    The original bridge grids in the tool-relative simulation frame before its
    axis rotation, uses the first point in each voxel, and always keeps the tool.
    """
    result = observation.copy()
    pos, feat, valid, _ = spec.unpack_numpy(result)
    physical = valid & ((feat[:, 0] > .5) | (feat[:, 1] > .5))
    physical &= (feat[:, 2] < .5) & (feat[:, 3] < .5)
    rng = np.random.default_rng(np.random.SeedSequence([seed, body, replica, frame]))
    # Draw full fixed-budget arrays so enabling noise does not change dropout draws.
    noise = rng.normal(0., 1., pos.shape)
    keep = rng.random(len(pos)) >= dropout
    if noise_m:
        pos[physical] += (noise[physical] * noise_m).astype(pos.dtype)
    if dropout:
        feat[physical & ~keep] = 0.
    if voxel_shift is not None:
        if voxel_size <= 0:
            raise ValueError("Voxel shift requires a positive voxel size")
        eligible = (feat[:, 0] > .5) | (feat[:, 1] > .5) | (feat[:, 3] > .5)
        ids = np.flatnonzero(eligible)
        keys = np.floor((pos[ids].astype(np.float64) - np.asarray(voxel_shift)) / voxel_size).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        selected = np.zeros(len(pos), bool)
        selected[ids[first]] = True
        selected |= feat[:, 3] > .5
        feat[eligible & ~selected] = 0.
    return result
