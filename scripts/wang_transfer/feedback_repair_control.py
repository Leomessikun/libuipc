"""Bounded feedback repairs using segmented camera clouds and tool history.

No simulator object is passed to a deployable controller. The separately named
privileged-state control is an information ablation, not a true oracle bound.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def pilot_settings():
    return dict(start=60, prefix_steps=12, suffix_steps=80,
                prefixes=["half", "no_rotation", "lift"],
                suffixes=["half", "no_rotation"], lift_residual=.25,
                sensor_noise_m=0., sensor_dropout=0., seed=2026100507,
                feature_contract="18 cloud statistics at repair start, their response delta, and 3 measured tool displacements; no force/goal/attached input")


def cloud_features(observation, tool, origin):
    """Unpack only camera xyz/segmentation; ignore all privileged scalar extras.

    Position quantiles are relative to the measured tool position at repair
    start. Tool compensation prevents arm motion caused by a moving camera
    frame from masquerading as material response.
    """
    observation = np.asarray(observation)
    if observation.ndim != 1 or (len(observation) - 7) % 7:
        raise ValueError("Unsupported packed camera observation")
    points = observation[:-7].reshape(-1, 7)
    cloth = points[points[:, 3] > .5, :3]
    arm = points[points[:, 4] > .5, :3]
    if len(cloth) < 3 or len(arm) < 3:
        return None
    translation = np.asarray(tool) - np.asarray(origin)
    # Quantiles give a small frozen pilot representation, not a proposed encoder.
    shape = np.quantile(cloth + translation, [.1, .5, .9], axis=0).ravel()
    arm_center = np.median(arm + translation, axis=0)
    gap = np.quantile(cKDTree(arm).query(cloth)[0], [.1, .5, .9])
    spread = np.std(cloth, axis=0)
    result = np.r_[shape, arm_center, gap, spread]
    return result.tolist() if np.isfinite(result).all() else None


def route(router, feature):
    if feature is None:
        return int(router.get("missing_action", 0))
    if router["feature"] is None:
        return int(router["left"])
    return int(router["left"] if feature[router["feature"]] <= router["threshold"] else router["right"])


def transform(action, kind, lift=.25):
    """Use the half/no-rotation/vertical-residual proposals of IPCActionFilter."""
    result = np.asarray(action, dtype=np.float32).copy()
    if kind == "half":
        result *= .5
    elif kind == "no_rotation":
        result[3:] = 0.
    elif kind == "lift":
        result[2] += lift
    else:
        raise ValueError(f"Unknown repair proposal {kind}")
    return np.clip(result, -1., 1.)


class FeedbackRepair:
    def __init__(self, settings, entry, seed, fitted=None):
        self.settings, self.entry = settings, entry
        self.fitted = fitted
        self.rng = np.random.default_rng(seed)
        self.origin = self.initial = self.response = self.privileged = None
        self.mode = self.branch = None
        self.root_reached = self.branch_reached = False
        self.mode_switches = 0
        self.last_mode = None
        if entry["kind"] not in ("base", "fixed") and fitted is None:
            raise ValueError("Target controller requires frozen source fit")

    def _sample(self, table):
        return int(self.rng.choice(len(table["repairs"]), p=table["weights"]))

    def _table(self):
        kind = self.entry["kind"]
        return self.fitted["tables"][kind if kind in ("flat", "initial", "privileged") else "observable"]

    def act(self, *, step, observation, tool, nominal, privileged=None):
        s, kind = self.settings, self.entry["kind"]
        end_prefix = s["start"] + s["prefix_steps"]
        end_repair = end_prefix + s["suffix_steps"]
        if step == s["start"]:
            self.origin = np.asarray(tool).copy()
            self.initial = cloud_features(observation, tool, self.origin)
            self.root_reached = True
            if kind == "fixed":
                self.mode = dict(prefix=self.entry["prefix"], assignment=[self.entry["suffix"]] * 2)
            elif kind == "base":
                self.mode = None
            else:
                table = self._table()
                self.mode = table["repairs"][self._sample(table)]
        if step == end_prefix and self.root_reached:
            current = cloud_features(observation, tool, self.origin)
            if current is not None and self.initial is not None:
                self.response = np.r_[self.initial, np.asarray(current) - self.initial,
                                      np.asarray(tool) - self.origin].tolist()
            self.branch_reached = True
            # Privileged values are recorded for the explicitly privileged control.
            self.privileged = None if privileged is None else list(map(float, privileged))
        if not s["start"] <= step < end_repair or self.mode is None:
            return np.asarray(nominal).copy(), False
        prefix = self.mode["prefix"]
        if step < end_prefix:
            return transform(nominal, s["prefixes"][prefix], s["lift_residual"]), True
        mode = self.mode
        if kind == "resample":
            table = self._table()
            idx = self._sample(table)
            if self.last_mode is not None and idx != self.last_mode:
                self.mode_switches += 1
            self.last_mode = idx
            mode = table["repairs"][idx]
            if mode is None:
                return np.asarray(nominal).copy(), False
            # Keep the executed prefix fixed; resampling tests suffix consistency.
        if kind in ("fixed", "flat"):
            branch = 0
        else:
            feature = self.privileged if kind == "privileged" else self.initial if kind == "initial" else self.response
            if feature is None:
                return np.asarray(nominal).copy(), False
            branch = route(self._table()["routers"][prefix], feature)
        assignment = mode["assignment"]
        if kind == "shuffled":
            assignment = self.fitted["shuffled_assignments"][str(prefix) + ":" + str(assignment)]
        self.branch = branch
        return transform(nominal, s["suffixes"][assignment[branch]], s["lift_residual"]), True

    def record(self):
        return dict(controller=self.entry, root_reached=self.root_reached,
                    branch_reached=self.branch_reached, initial_feature=self.initial,
                    response_feature=self.response, privileged_feature=self.privileged,
                    selected_mode=self.mode, selected_branch=self.branch,
                    mode_switches=self.mode_switches)


def load_repairs(path, count, seed):
    config = json.loads(Path(path).read_text())
    if len(config["controllers"]) != count:
        raise ValueError("Need one repair controller per variant before replication")
    names = [e["name"] for e in config["controllers"]]
    if len(set(names)) != count:
        raise ValueError("Repair controller names must be unique")
    # Common random numbers couple mode draws across target ablations. Changing
    # a slot's position in a batched world must not change its sampled repair.
    return [FeedbackRepair(config["settings"], e, seed, config.get("fitted"))
            for e in config["controllers"]]
