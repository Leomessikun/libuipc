"""Outcome value for dressing: P(final success | privileged state), shared by training and IPC lookahead.

The features use only the current state and changes over the last WINDOW decisions, so the same function
applies to an archived state (from the stored arrays) and to the end of an IPC candidate rollout (from the
live env). The value is trained on the policy's own rollouts, so at a candidate's end it estimates the
success probability of continuing with that policy: choosing the candidate that maximizes it is a one-step
policy improvement with an exact short physics rollout and a learned continuation.

The model is a small MLP stored as numpy arrays; inference needs only numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WINDOW = 4
GARMENTS = ("tshirt_4", "tshirt_26", "tshirt_68", "tshirt_392", "hospital_gown")
NAMES = ("time", "tcp_along", "tcp_radial", "grip_N", "grip_max_N", "track_m", "track_max_m", "upper", "fore",
         "cuff_s", "proximal", "d_proximal", "d_upper", *(f"action_{i}" for i in range(6)),
         *(f"garment_{g}" for g in GARMENTS))


def state_features(*, step, tcp, finger, shoulder, grip, grip_window, track, track_window, upper, fore, cuff, prox,
                   prox_before, upper_before, action, garment, horizon=450):
    arm = np.asarray(shoulder, float) - np.asarray(finger, float)
    length = float(np.linalg.norm(arm))
    axis = arm / length
    rel = np.asarray(tcp, float) - np.asarray(finger, float)
    along = float(rel @ axis) / length
    radial = float(np.linalg.norm(rel - (rel @ axis) * axis))
    return np.asarray([step / horizon, along, radial, grip, max(grip_window), track, max(track_window), upper, fore,
                       cuff, prox, prox - prox_before, upper - upper_before, *np.asarray(action, float),
                       *(float(garment == g) for g in GARMENTS)], np.float32)


def archive_features(z, garment, steps):
    """Features at the given state indices of an archived collector episode (np.load result)."""
    force = np.linalg.norm(z["gripper_force"], axis=1)
    track = np.concatenate([[0.], z["tracking_error"]])
    act = np.concatenate([np.zeros((1, 6)), z["actions"]])
    out = []
    for k in steps:
        lo = max(0, k - WINDOW)
        out.append(state_features(step=k, tcp=z["tcp"][k], finger=z["finger"], shoulder=z["shoulder"],
                                  grip=force[k], grip_window=force[lo:k + 1], track=track[k],
                                  track_window=track[lo:k + 1], upper=z["upperarm_ratio"][k],
                                  fore=z["forearm_ratio"][k], cuff=z["sleeve_cuff_s"][k],
                                  prox=z["sleeve_proximal_upper_fraction"][k],
                                  prox_before=z["sleeve_proximal_upper_fraction"][lo],
                                  upper_before=z["upperarm_ratio"][lo], action=act[k], garment=garment))
    return np.stack(out)


class OutcomeValue:
    """Numpy MLP: standardized features -> logit of final success."""

    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.mean, self.std = data["mean"], data["std"]
        self.layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(int(data["depth"]))]
        self.meta = json.loads(str(data["meta"]))
        self.keep = np.asarray(self.meta.get("keep", list(range(len(NAMES)))), int)

    def __call__(self, features):
        x = np.atleast_2d(np.asarray(features, np.float32))
        if x.shape[1] == len(NAMES):
            x = x[:, self.keep]
        x = (x - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.)
        return 1. / (1. + np.exp(-x[:, 0]))


def save(path, mean, std, layers, meta):
    arrays = dict(mean=mean, std=std, depth=np.array(len(layers)), meta=np.array(json.dumps(meta)))
    for i, (w, b) in enumerate(layers):
        arrays[f"w{i}"], arrays[f"b{i}"] = w, b
    np.savez(path, **arrays)
