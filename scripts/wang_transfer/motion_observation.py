"""Causal arm registration and episode-local history; no simulator motion input."""
from __future__ import annotations

import numpy as np


class ObservedArmMotion:
    """Estimate a one-decision rigid displacement from visible segmented points.

    Clouds are tool-relative; measured tool position removes robot ego-motion.
    Registration is approximate for articulated arms. Invalid estimates become
    zero velocity, never a fallback to commanded body motion or future samples.
    """

    def __init__(self, register, roi, *, dt, max_speed=.5, max_rms=.02):
        self.register, self.roi = register, roi
        self.dt, self.max_speed, self.max_rms = dt, max_speed, max_rms
        self.previous = None
        self.transform = np.eye(4)
        self.diagnostics = dict(valid=False, reason="first_observation")

    def update(self, positions, flags, valid, tool):
        arm = np.asarray(positions)[np.asarray(valid) & (np.asarray(flags)[:, 1] > .5)]
        cloud = self.roi(arm + np.asarray(tool), tool)
        self.transform = np.eye(4)
        if self.previous is not None:
            transform, diag = self.register(self.previous, cloud)
            if diag.get("valid") and diag.get("rms_m", np.inf) <= self.max_rms:
                displacement = cloud @ transform[:3, :3].T + transform[:3, 3] - cloud
                speed = float(np.max(np.linalg.norm(displacement, axis=1))) / self.dt
                diag = dict(diag, max_observed_speed_m_s=speed)
                if speed <= self.max_speed:
                    self.transform = transform
                else:
                    diag.update(valid=False, reason="excess_observed_speed")
            else:
                diag = dict(diag, valid=False, reason=diag.get("reason", "registration_residual"))
            self.diagnostics = diag
        self.previous = cloud.copy()
        return dict(self.diagnostics)

    def displacement(self, points):
        points = np.asarray(points, float)
        delta = points @ self.transform[:3, :3].T + self.transform[:3, 3] - points
        # Far-away body points can amplify a noisy angular estimate. Bound the
        # extrapolated velocity as well as the observed ROI velocity.
        norm = np.linalg.norm(delta, axis=-1, keepdims=True)
        return delta * np.minimum(1., self.max_speed * self.dt / np.maximum(norm, 1e-12))


def history_indices(length, frames):
    """Causal, left-padded indices for ONE episode, including the current frame."""
    if length < 1 or frames < 1:
        raise ValueError("History needs positive episode length and frame count")
    return np.maximum(np.arange(length)[:, None] - np.arange(frames - 1, -1, -1), 0)


def history_inputs(features, tools, frames, *, current_only=False):
    """Identical-size inputs for history and repeated-current ablation.

    Features are FMVP's frozen point encoder output. Tool positions, expressed
    in the model frame, are relative to the current measured tool position.
    No motion ID, episode clock, human joint states or future frames enter.
    """
    features, tools = np.asarray(features), np.asarray(tools)
    if features.ndim != 2 or tools.shape != (len(features), 3):
        raise ValueError("Expected episode features [T,F] and tool positions [T,3]")
    index = history_indices(len(features), frames)
    if current_only:
        index = np.repeat(np.arange(len(features))[:, None], frames, axis=1)
    relative_tool = (tools[index] - tools[:, None]) / .02
    return np.concatenate((features[index], relative_tool), axis=-1).reshape(len(features), -1)
