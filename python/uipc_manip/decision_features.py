"""Fixed-size features of a dressing decision, in three information sets.

The three sets exist to separate questions that are otherwise confounded when a
policy fails: whether the better action is *identifiable* from what the robot can
see at deployment, from that plus a short history, or only from what the simulator
knows.

* :func:`observation_features` reads one packed point-cloud observation, the exact
  input the deployed actor receives.
* :func:`history_features` adds the previous decisions' observations and the
  commands that were executed, including how far the garment actually moved per
  commanded metre, which is what distinguishes a sleeve that is still sliding from
  one that has stopped.
* :func:`privileged_features` reads the simulator's own low-dimensional state, which
  is not available on a robot.

Every function returns a flat float array of a fixed length, so the same model class
can be fitted to each set and only the information differs.
"""
from __future__ import annotations

import numpy as np

from .obs import EXTRA_DIM, POINT_DIM

OBSERVATION_FEATURE_NAMES = (
    "cloth_fraction", "arm_fraction",
    "cloth_centroid_x", "cloth_centroid_y", "cloth_centroid_z",
    "cloth_spread_x", "cloth_spread_y", "cloth_spread_z",
    "arm_centroid_x", "arm_centroid_y", "arm_centroid_z",
    "goal_x", "goal_y", "goal_z", "goal_distance",
    "cloth_along_goal", "cloth_extent_along_goal", "cloth_beyond_tool_fraction",
    "cloth_min_distance", "cloth_mean_distance",
    "cloth_arm_min_gap", "cloth_arm_close_fraction", "attached",
)


def unpack_observation(flat: np.ndarray, point_budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a packed observation into tool-relative points, flags, validity and extras."""
    flat = np.asarray(flat, dtype=np.float64).reshape(-1)
    expected = point_budget * POINT_DIM + EXTRA_DIM
    if flat.size != expected:
        raise ValueError(f"Observation of {flat.size} values is not {expected} for a {point_budget}-point budget")
    block = flat[: point_budget * POINT_DIM].reshape(point_budget, POINT_DIM)
    feat = block[:, 3:]
    return block[:, :3], feat, feat.sum(axis=-1) > 0.5, flat[point_budget * POINT_DIM:]


def observation_features(flat: np.ndarray, point_budget: int, close_m: float = 0.02) -> np.ndarray:
    """Deployment-side features of one observation: segmented cloud geometry only."""
    pos, feat, valid, extra = unpack_observation(flat, point_budget)
    cloth = pos[valid & (feat[:, 0] > 0.5) & (feat[:, 1] < 0.5)]
    arm = pos[valid & (feat[:, 1] > 0.5)]
    goal = extra[3:6]
    goal_distance = float(np.linalg.norm(goal))
    direction = goal / goal_distance if goal_distance > 1e-9 else np.zeros(3)
    out = [len(cloth) / point_budget, len(arm) / point_budget]
    for block in (cloth, arm):
        out.extend(block.mean(axis=0) if len(block) else np.zeros(3))
        if block is cloth:
            out.extend(block.std(axis=0) if len(block) else np.zeros(3))
    out.extend(goal)
    out.append(goal_distance)
    if len(cloth):
        along = cloth @ direction
        distance = np.linalg.norm(cloth, axis=1)
        out.extend([float(along.mean()), float(np.ptp(along)), float((along > 0).mean()),
                    float(distance.min()), float(distance.mean())])
    else:
        out.extend([0.0] * 5)
    if len(cloth) and len(arm):
        gaps = np.linalg.norm(cloth[:, None, :] - arm[None, :, :], axis=-1)
        nearest = gaps.min(axis=1)
        out.extend([float(nearest.min()), float((nearest < close_m).mean())])
    else:
        out.extend([0.0, 0.0])
    out.append(float(extra[6]))
    result = np.asarray(out, dtype=np.float64)
    if result.size != len(OBSERVATION_FEATURE_NAMES):
        raise AssertionError(f"{result.size} features against {len(OBSERVATION_FEATURE_NAMES)} names")
    return result


def history_features(observations: np.ndarray, actions: np.ndarray, point_budget: int,
                     max_translation: float, close_m: float = 0.02) -> np.ndarray:
    """Features of a history window: per-frame observation features, the commands, and the response.

    ``observations[k]`` is the observation before ``actions[k]``; the last row is the
    decision's own observation. The response terms are the garment centroid's motion
    between consecutive frames divided by the metres of translation commanded in
    between, the quantity that separates a sliding sleeve from a stalled one.
    """
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions) + 1:
        raise ValueError("Need one more observation than action in a history window")
    frames = np.stack([observation_features(o, point_budget, close_m) for o in observations])
    out = [frames.reshape(-1), actions.reshape(-1)]
    centroids = frames[:, 2:5]
    commanded = np.linalg.norm(actions[:, :3], axis=1) * float(max_translation)
    moved = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    out.append(moved)
    out.append(moved / np.maximum(commanded, 1e-9))
    out.append(commanded)
    return np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in out])


def privileged_features(privileged: np.ndarray) -> np.ndarray:
    """The simulator's own state of the slot, unavailable at deployment."""
    return np.asarray(privileged, dtype=np.float64).reshape(-1)


def standardize(train: np.ndarray, *others: np.ndarray):
    """Centre and scale by the training rows; constant columns are left at zero."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return tuple((x - mean) / scale for x in (train, *others))


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge weights for ``y ~ [x, 1]``, penalising the slopes only."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    penalty = alpha * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def ridge_predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.concatenate([x, np.ones((len(x), 1))], axis=1) @ weights
