"""Flat observation layout shared by the environment, replay buffer, and agent.

The observation follows the Wang RSS 2023 / FMVP convention used by the Newton
dressing teacher: a segmented point cloud expressed relative to the tool point,
with the tool inserted as an explicit point, plus a short vector of per-graph
scalars. Everything is packed into one float32 vector so replay stays a dense
array.

Point features (one-hot style, several flags may be set on one point):

``[is_deformable, is_marker, is_goal, is_tool]``

Padded rows have every flag equal to zero, which is how :func:`unpack` derives
the validity mask. The trailing scalars are

``[tool_x, tool_y, tool_z, goal_x - tool_x, goal_y - tool_y, goal_z - tool_z, attached]``

in world units, so the actor and critic can condition on the absolute tool
position and the goal offset even when the goal point is beyond the encoder's
ball-query radii.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POINT_DIM = 7
"""Floats per point: xyz relative to the tool plus four segmentation flags."""

FEATURE_DIM = 4
"""Segmentation flags per point."""

EXTRA_DIM = 7
"""Per-graph scalars appended after the point block."""

CONSTRAINT_FLAG_DIM = 1
"""Optional extra scalar: whether the episode's absorbing constraint is already violated.

A trajectory-level criterion is absorbing -- once the grasp has left its tolerance the
episode can no longer be a valid success, and the decision problem after that point is a
different one. A policy that cannot see the flag cannot represent that difference, and a
critic that cannot see it must average two incompatible continuations. The slot is opt-in
so every observation, checkpoint and replay written before it keeps its exact width."""

FLAG_DEFORMABLE = 0
FLAG_MARKER = 1
FLAG_GOAL = 2
FLAG_TOOL = 3


@dataclass(frozen=True)
class ObsSpec:
    """Dimensions of the flat observation for a fixed point budget."""

    point_budget: int
    constraint_flag: bool = False
    """Append the absorbing-constraint flag to the observation tail."""
    episode_clock: bool = False
    """Append elapsed episode fraction for finite-horizon control (zero at reset)."""

    def __post_init__(self) -> None:
        if int(self.point_budget) < 3:
            raise ValueError("point_budget must hold at least the tool, the goal, and one deformable point")

    @property
    def extra_dim(self) -> int:
        return EXTRA_DIM + int(self.constraint_flag) + int(self.episode_clock)

    @property
    def dim(self) -> int:
        return int(self.point_budget) * POINT_DIM + self.extra_dim

    @property
    def deformable_budget(self) -> int:
        """Deformable points that fit beside the tool and goal points."""
        return int(self.point_budget) - 2

    def pack(
        self,
        deformable_rel: np.ndarray,
        marker_mask: np.ndarray,
        goal_rel: np.ndarray,
        tool_world: np.ndarray,
        attached: bool,
        violated: bool = False,
        elapsed_fraction: float = 0.0,
    ) -> np.ndarray:
        """Build one flat observation.

        Args:
            deformable_rel: ``[M, 3]`` deformable points relative to the tool,
                ``M <= deformable_budget``.
            marker_mask: ``[M]`` boolean, which deformable points are markers.
            goal_rel: ``[3]`` goal position relative to the tool.
            tool_world: ``[3]`` absolute tool position.
            attached: Whether the deformable is currently attached to the tool.
        """
        deformable_rel = np.asarray(deformable_rel, dtype=np.float32).reshape(-1, 3)
        marker_mask = np.asarray(marker_mask, dtype=bool).reshape(-1)
        count = deformable_rel.shape[0]
        if count != marker_mask.shape[0]:
            raise ValueError("deformable_rel and marker_mask must have the same length")
        if count > self.deformable_budget:
            raise ValueError(f"{count} deformable points exceed the budget of {self.deformable_budget}")
        block = np.zeros((int(self.point_budget), POINT_DIM), dtype=np.float32)
        block[:count, :3] = deformable_rel
        block[:count, 3 + FLAG_DEFORMABLE] = 1.0
        block[:count, 3 + FLAG_MARKER] = marker_mask.astype(np.float32)
        block[count, :3] = np.asarray(goal_rel, dtype=np.float32).reshape(3)
        block[count, 3 + FLAG_GOAL] = 1.0
        # The tool sits at the origin of the tool-relative frame.
        block[count + 1, 3 + FLAG_TOOL] = 1.0
        extra = np.zeros(self.extra_dim, dtype=np.float32)
        extra[0:3] = np.asarray(tool_world, dtype=np.float32).reshape(3)
        extra[3:6] = np.asarray(goal_rel, dtype=np.float32).reshape(3)
        extra[6] = 1.0 if attached else 0.0
        if self.constraint_flag:
            extra[EXTRA_DIM] = 1.0 if violated else 0.0
        elif violated:
            raise ValueError("This observation spec has no constraint slot to record a violation in")
        if self.episode_clock:
            extra[-1] = elapsed_fraction
        return np.concatenate([block.reshape(-1), extra])

    def pack_labeled(
        self,
        points_rel: np.ndarray,
        flags: np.ndarray,
        goal_rel: np.ndarray,
        tool_world: np.ndarray,
        attached: bool,
        violated: bool = False,
        elapsed_fraction: float = 0.0,
    ) -> np.ndarray:
        """Build one flat observation from points that carry their own segmentation flags.

        Args:
            points_rel: ``[M, 3]`` points relative to the tool, ``M <= deformable_budget``.
            flags: ``[M, FEATURE_DIM]`` segmentation flags per point (for example arm
                points flagged as markers and cloth points flagged as deformable).
            goal_rel: ``[3]`` goal position relative to the tool.
            tool_world: ``[3]`` absolute tool position.
            attached: Whether the deformable is currently attached to the tool.
        """
        points_rel = np.asarray(points_rel, dtype=np.float32).reshape(-1, 3)
        flags = np.asarray(flags, dtype=np.float32).reshape(-1, FEATURE_DIM)
        count = points_rel.shape[0]
        if count != flags.shape[0]:
            raise ValueError("points_rel and flags must have the same length")
        if count > self.deformable_budget:
            raise ValueError(f"{count} points exceed the budget of {self.deformable_budget}")
        block = np.zeros((int(self.point_budget), POINT_DIM), dtype=np.float32)
        block[:count, :3] = points_rel
        block[:count, 3:] = flags
        block[count, :3] = np.asarray(goal_rel, dtype=np.float32).reshape(3)
        block[count, 3 + FLAG_GOAL] = 1.0
        block[count + 1, 3 + FLAG_TOOL] = 1.0
        extra = np.zeros(self.extra_dim, dtype=np.float32)
        extra[0:3] = np.asarray(tool_world, dtype=np.float32).reshape(3)
        extra[3:6] = np.asarray(goal_rel, dtype=np.float32).reshape(3)
        extra[6] = 1.0 if attached else 0.0
        if self.constraint_flag:
            extra[EXTRA_DIM] = 1.0 if violated else 0.0
        elif violated:
            raise ValueError("This observation spec has no constraint slot to record a violation in")
        if self.episode_clock:
            extra[-1] = elapsed_fraction
        return np.concatenate([block.reshape(-1), extra])

    def unpack_numpy(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split a flat observation (or a batch of them) into arrays.

        Returns ``(pos [.., N, 3], feat [.., N, 4], valid [.., N], extra [.., extra_dim])``.
        """
        flat = np.asarray(flat, dtype=np.float32)
        squeeze = flat.ndim == 1
        flat = flat.reshape(-1, self.dim)
        n = int(self.point_budget)
        block = flat[:, : n * POINT_DIM].reshape(-1, n, POINT_DIM)
        pos = block[:, :, :3]
        feat = block[:, :, 3:]
        valid = feat.sum(axis=-1) > 0.5
        extra = flat[:, n * POINT_DIM :]
        if squeeze:
            return pos[0], feat[0], valid[0], extra[0]
        return pos, feat, valid, extra

    def unpack_torch(self, flat):
        """Torch version of :meth:`unpack_numpy` for ``[B, dim]`` tensors."""
        n = int(self.point_budget)
        block = flat[:, : n * POINT_DIM].reshape(flat.shape[0], n, POINT_DIM)
        pos = block[:, :, :3]
        feat = block[:, :, 3:]
        valid = feat.sum(dim=-1) > 0.5
        extra = flat[:, n * POINT_DIM :]
        return pos, feat, valid, extra


def marker_centroid_rel(flat: np.ndarray, spec: ObsSpec) -> np.ndarray:
    """Tool-relative centroid of the marker points inside one observation."""
    pos, feat, valid, _ = spec.unpack_numpy(flat)
    mask = valid & (feat[:, FLAG_MARKER] > 0.5)
    if not mask.any():
        return np.zeros(3, dtype=np.float32)
    return pos[mask].mean(axis=0)


def goal_rel(flat: np.ndarray, spec: ObsSpec) -> np.ndarray:
    """Tool-relative goal position stored in the observation tail."""
    _, _, _, extra = spec.unpack_numpy(flat)
    return extra[3:6]


def constraint_flag(flat, spec: ObsSpec):
    """The absorbing-constraint flag of one flat observation, or of a batch of them.

    Accepts numpy arrays and torch tensors, and returns the same kind. Raises when the
    spec has no constraint slot, so a silent zero can never stand in for "not violated".
    """
    if not spec.constraint_flag:
        raise ValueError("This observation spec does not carry a constraint flag")
    index = int(spec.point_budget) * POINT_DIM + EXTRA_DIM
    if hasattr(flat, "ndim") and not isinstance(flat, np.ndarray):  # torch tensor
        return flat[..., index]
    flat = np.asarray(flat, dtype=np.float32)
    return flat[..., index]


def episode_fraction(flat, spec: ObsSpec):
    """Elapsed fraction of the finite episode, for numpy or torch observations."""
    if not spec.episode_clock:
        raise ValueError("This observation spec does not carry an episode clock")
    return flat[..., int(spec.point_budget) * POINT_DIM + EXTRA_DIM + int(spec.constraint_flag)]
