"""Goal-reaching manipulation tasks on a cloth sheet or an elastic cable.

Each task fixes which vertices the robot holds at the start of an episode
(the Newton dressing teacher likewise starts with the garment cuff already
attached), which vertices form the *marker* whose centroid must reach the
goal, and how goals are sampled. Rewards are dense progress rewards, the
episode always runs to the time limit, and success is reported as a metric
rather than used as a terminal condition, matching the FMVP pretraining MDP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

TABLE_REST_HEIGHT = 0.01
"""Rest height of a flat deformable above the plane (collision radius + d_hat + margin)."""


@dataclass(frozen=True)
class TaskSpec:
    name: str
    deformable: str
    grasp_vertices: Callable[[np.ndarray], np.ndarray]
    marker_vertices: Callable[[np.ndarray], np.ndarray]
    sample_goal: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray]
    success_tolerance: float
    grasp_height: float
    description: str


def _corner_patch(rest: np.ndarray, sign_x: float, sign_y: float, radius: float = 0.03) -> np.ndarray:
    corner = np.array(
        [rest[:, 0].max() if sign_x > 0 else rest[:, 0].min(), rest[:, 1].max() if sign_y > 0 else rest[:, 1].min()]
    )
    d = np.linalg.norm(rest[:, :2] - corner[None, :], axis=1)
    idx = np.flatnonzero(d <= radius)
    if idx.size == 0:
        idx = np.array([int(np.argmin(d))])
    return idx


def _planar_offset_goal(
    rng: np.random.Generator, marker_centroid: np.ndarray, min_r: float, max_r: float, height: float
) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    r = rng.uniform(min_r, max_r)
    goal = marker_centroid.copy()
    goal[0] += r * np.cos(angle)
    goal[1] += r * np.sin(angle)
    goal[2] = height
    return goal


def _cloth_drag_goal(rng: np.random.Generator, rest: np.ndarray, marker_centroid: np.ndarray) -> np.ndarray:
    return _planar_offset_goal(rng, marker_centroid, 0.06, 0.14, TABLE_REST_HEIGHT)


def _cloth_fold_goal(rng: np.random.Generator, rest: np.ndarray, marker_centroid: np.ndarray) -> np.ndarray:
    # The held corner must land on the diagonally opposite corner. A small
    # random shift keeps the target from being a constant.
    opposite = rest[_corner_patch(rest, -1.0, -1.0, radius=0.01)].mean(axis=0)
    goal = opposite.copy()
    goal[:2] += rng.uniform(-0.02, 0.02, size=2)
    goal[2] = TABLE_REST_HEIGHT + 0.02
    return goal


def _cable_drag_goal(rng: np.random.Generator, rest: np.ndarray, marker_centroid: np.ndarray) -> np.ndarray:
    return _planar_offset_goal(rng, marker_centroid, 0.05, 0.12, TABLE_REST_HEIGHT)


def _cable_end(rest: np.ndarray) -> np.ndarray:
    return np.array([0], dtype=np.int64)


def _cable_middle(rest: np.ndarray) -> np.ndarray:
    n = rest.shape[0]
    mid = n // 2
    return np.arange(max(0, mid - 1), min(n, mid + 2), dtype=np.int64)


def _all_vertices(rest: np.ndarray) -> np.ndarray:
    return np.arange(rest.shape[0], dtype=np.int64)


TASKS: dict[str, TaskSpec] = {
    "cloth_drag": TaskSpec(
        name="cloth_drag",
        deformable="cloth",
        grasp_vertices=lambda rest: _corner_patch(rest, 1.0, 1.0),
        marker_vertices=_all_vertices,
        sample_goal=_cloth_drag_goal,
        success_tolerance=0.02,
        grasp_height=TABLE_REST_HEIGHT + 0.02,
        description="Hold one corner of a flat sheet and move the sheet centroid to a planar goal.",
    ),
    "cloth_fold": TaskSpec(
        name="cloth_fold",
        deformable="cloth",
        grasp_vertices=lambda rest: _corner_patch(rest, 1.0, 1.0),
        marker_vertices=lambda rest: _corner_patch(rest, 1.0, 1.0),
        sample_goal=_cloth_fold_goal,
        success_tolerance=0.03,
        grasp_height=TABLE_REST_HEIGHT + 0.02,
        description="Bring the held corner onto the diagonally opposite corner (a diagonal fold).",
    ),
    "cable_drag": TaskSpec(
        name="cable_drag",
        deformable="cable",
        grasp_vertices=_cable_end,
        marker_vertices=_cable_middle,
        sample_goal=_cable_drag_goal,
        success_tolerance=0.02,
        grasp_height=TABLE_REST_HEIGHT + 0.02,
        description="Hold one end of a cable lying on the table and move its midpoint to a planar goal.",
    ),
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown task {name!r}; available: {sorted(TASKS)}") from exc


def heuristic_action(marker_rel: np.ndarray, goal_rel: np.ndarray, max_translation: float) -> np.ndarray:
    """Scripted baseline: move the tool so the marker centroid heads for the goal.

    Both inputs are tool-relative, so their difference is the displacement the
    marker still needs. Moving the tool by that displacement is exact for a
    rigidly attached marker and a reasonable first guess otherwise.
    """
    delta = np.asarray(goal_rel, dtype=np.float32) - np.asarray(marker_rel, dtype=np.float32)
    return np.clip(delta / float(max_translation), -1.0, 1.0).astype(np.float32)
