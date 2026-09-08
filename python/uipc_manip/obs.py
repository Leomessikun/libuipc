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

FLAG_DEFORMABLE = 0
FLAG_MARKER = 1
FLAG_GOAL = 2
FLAG_TOOL = 3


@dataclass(frozen=True)
class ObsSpec:
    """Dimensions of the flat observation for a fixed point budget."""

    point_budget: int

    def __post_init__(self) -> None:
        if int(self.point_budget) < 3:
            raise ValueError("point_budget must hold at least the tool, the goal, and one deformable point")

    @property
    def dim(self) -> int:
        return int(self.point_budget) * POINT_DIM + EXTRA_DIM

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
        extra = np.zeros(EXTRA_DIM, dtype=np.float32)
        extra[0:3] = np.asarray(tool_world, dtype=np.float32).reshape(3)
        extra[3:6] = np.asarray(goal_rel, dtype=np.float32).reshape(3)
        extra[6] = 1.0 if attached else 0.0
        return np.concatenate([block.reshape(-1), extra])

    def unpack_numpy(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split a flat observation (or a batch of them) into arrays.

        Returns ``(pos [.., N, 3], feat [.., N, 4], valid [.., N], extra [.., EXTRA_DIM])``.
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
