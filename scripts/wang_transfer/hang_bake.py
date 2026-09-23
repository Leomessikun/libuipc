"""Let the garment hang from Wang's two picker vertices under gravity, away from the arm.

Builds the socket-placed cell, moves the garment 1.2 m out along the arm's outward horizontal so
nothing touches it, holds it by ``--anchors`` vertices, and advances with zero actions. Saves the
cloth relative to the picker every ``--every`` steps, so the settle can be read as it goes.
"""
import argparse, sys, time
from dataclasses import replace
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--garment", default="tshirt_26")
ap.add_argument("--body", type=int, default=14045)
ap.add_argument("--anchors", type=int, default=2)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--every", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, "/home/ge47gax/kun/libuipc/.claude/worktrees/residual-rl/python")
from uipc_manip import dressing_live, pretrain_wang, train_sac
from uipc_manip.dressing_env import GenesisIPCDressingEnv

_build = dressing_live.LiveCellFactory.build


def far_build(self, garment, human):
    cell = _build(self, garment, human)
    out = np.asarray(cell.finger, float) - np.asarray(cell.landmarks["right_elbow"], float)
    out[2] = 0.0
    shift = out / np.linalg.norm(out) * 1.2 + np.array([0, 0, 0.3])
    return replace(cell, cloth=cell.cloth + shift, picker_pos=np.asarray(cell.picker_pos, float) + shift,
                   pull_waypoints=cell.pull_waypoints + shift)


dressing_live.LiveCellFactory.build = far_build
_, targs, _ = pretrain_wang.prepare(["teacher", "--region", "13", "--seed", "1", "--obs-mode", "wang_static_arm", "--no-obs-augment"])
cfg = replace(train_sac.dressing_config(targs), cells=((a.garment, a.body),), cell_source="live", horizon=a.steps + 10,
              seed=1, show_viewer=False, decision_watchdog=False, anchor_count=a.anchors, settle_steps=30)
env = GenesisIPCDressingEnv(cfg, num_envs=1)
env.reset([1000])
c = env.cells[0]
print(f"[hang] anchors {env._pickers[0]['anchor_idx'].tolist()} picker {np.round(env._anchor[0], 3)}", flush=True)
zero = np.zeros((1, env.action_dim), np.float32)
prev, t0, snaps = env.positions()[0].copy(), time.time(), {}
for k in range(1, a.steps + 1):
    env.step(zero)
    if k % a.every == 0:
        P = env.positions()[0]
        g = np.asarray(env._anchor[0], float)
        rel = P - g
        v = rel[c.opening_idx].mean(0)
        moved = float(np.linalg.norm(P - prev, axis=1).max()); prev = P.copy()
        print(f"[hang] k={k} opening rel picker {v.round(3)} (|{np.linalg.norm(v):.3f}|, {np.degrees(np.arcsin(-v[2]/np.linalg.norm(v))):.0f} deg down) "
              f"centroid {rel.mean(0).round(3)} max move over {a.every} steps {moved*1000:.1f} mm ({time.time()-t0:.0f}s)", flush=True)
        snaps[k] = rel.copy()
np.savez(a.out, **{f"k{k}": v for k, v in snaps.items()}, opening_idx=c.opening_idx, faces=c.faces)
env.close()
