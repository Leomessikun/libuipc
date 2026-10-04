"""State-feedback labels of the scripted expert's translation (dressing_heuristic, no rotation), from stored states.

The scripted expert is a stage machine: it advances from "move to the hover point above the finger" to "drive
the sleeve opening past the elbow" only when its own gripper reaches the hover point. Run in shadow while a
student executes (DAgger), it stays in the first stage whenever the student passes the hand without touching
that point, and its labels then pull the gripper back along the arm (93 % of the first DAgger round's labels).
Here the stage is a function of the state instead: the second stage applies once the gripper has come within
``reach`` of the hover point or has passed it along the arm. Translation magnitudes follow the expert (8 mm
per decision, clipped to the speed cap).
"""
from __future__ import annotations

import numpy as np

Z_OFFSET, ELBOW_OVERSHOOT, STEP, TOLERANCE = 0.12, 0.08, 0.008, 0.012


def expert_labels(z, max_translation: float, *, reach: float = TOLERANCE, past: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-state 3-D action labels and stage (0 hover approach, 2 opening to elbow) for an episode npz."""
    finger, elbow, shoulder = (np.asarray(z[k], float) for k in ("finger", "elbow", "shoulder"))
    tcp = np.asarray(z["tcp"], float)
    opening = np.asarray(z["positions"], float)[:, np.asarray(z["opening_idx"])].mean(axis=1)
    hover = finger + np.array([0., 0., Z_OFFSET])
    forearm = (elbow - finger) / np.linalg.norm(elbow - finger)
    middle = elbow + forearm * ELBOW_OVERSHOOT
    axis = (shoulder - finger) / np.linalg.norm(shoulder - finger)
    n = len(tcp) - 1
    labels = np.zeros((n, 3))
    stages = np.zeros(n, np.int8)
    reached = False
    for t in range(n):
        reached |= (np.linalg.norm(hover - tcp[t]) < reach) or ((tcp[t] - hover) @ axis > past)
        point, target = (opening[t], middle) if reached else (tcp[t], hover)
        d = target - point
        dist = float(np.linalg.norm(d))
        if dist >= TOLERANCE:
            labels[t] = d / dist * STEP / max_translation
        stages[t] = 2 if reached else 0
    return np.clip(labels, -1., 1.), stages
