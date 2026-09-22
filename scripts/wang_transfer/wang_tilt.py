"""Released policy on a cell whose garment is swung about the picker so the opening hangs below it.

FMVP's PyBullet start (where this checkpoint dresses, upper 0.991) has the opening 24 cm straight
below the gripper and 13.5 cm below the fingertip; our socket placement holds it level with the
fingertip, 40 degrees below the horizontal from the picker. This rotates the placed garment about
the picker by --tilt degrees toward straight down, checks the start is legal, and runs the policy
with no action clamp (rotation off), yaw 267.
"""
import argparse, json, sys, time
from dataclasses import replace
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--garment", default="tshirt_26")
ap.add_argument("--body", type=int, default=14045)
ap.add_argument("--yaw", type=float, default=267.0)
ap.add_argument("--tilt", type=float, default=38.0)
ap.add_argument("--drop", type=float, default=0.0, help="extra metres the whole garment+picker moves down")
ap.add_argument("--out-shift", type=float, default=0.0)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--expert", action="store_true")
ap.add_argument("--checkpoint", default="/home/ge47gax/Desktop/vision_based_policy.pt")
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, "/home/ge47gax/kun/libuipc/.claude/worktrees/residual-rl/python")
from uipc_manip import dressing_live, pretrain_wang, train_sac
from uipc_manip.dressing_env import GenesisIPCDressingEnv
from uipc_manip.obs import ObsSpec
from uipc_manip.wang_bridge import up_axis_rotation
from uipc_manip.wang_client import WangPolicyClient


def rodrigues(axis, angle):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


_build = dressing_live.LiveCellFactory.build


def tilted_build(self, garment, human):
    cell = _build(self, garment, human)
    picker = np.asarray(cell.picker_pos, float)
    v = cell.cloth[cell.opening_idx].mean(0) - picker
    down = np.array([0, 0, -1.0])
    Rt = rodrigues(np.cross(v, down), np.radians(a.tilt)) if a.tilt else np.eye(3)
    cloth = (cell.cloth - picker) @ Rt.T + picker
    lm = cell.landmarks
    fa = np.asarray(lm["right_elbow"], float) - np.asarray(cell.finger, float)
    shift = np.array([0, 0, -a.drop]) - fa / np.linalg.norm(fa) * a.out_shift
    cloth = cloth + shift
    gap = dressing_live.garment_arm_gap(cloth, cell.faces, cell.arm_points, cell.arm_faces)
    v2 = cloth[cell.opening_idx].mean(0) - (picker + shift)
    print(f"[tilt] {garment}/{human}: tilt {a.tilt} drop {a.drop}; picker->opening {v.round(3)} -> {v2.round(3)} "
          f"({np.degrees(np.arcsin(-v2[2] / np.linalg.norm(v2))):.0f} deg below horizontal); arm gap {gap*1000:.1f} mm", flush=True)
    if gap < 0.003:
        raise SystemExit(f"illegal start: gap {gap*1000:.1f} mm")
    return replace(cell, cloth=cloth, picker_pos=picker + shift,
                   pull_waypoints=np.stack([picker + shift, cell.pull_waypoints[-1]]))


dressing_live.LiveCellFactory.build = tilted_build

n = 2 if a.expert else 1
_, targs, _ = pretrain_wang.prepare(["teacher", "--region", "13", "--seed", "1", "--obs-mode", "wang_static_arm", "--no-obs-augment"])
cfg = replace(train_sac.dressing_config(targs), cells=tuple((a.garment, a.body) for _ in range(n)), cell_source="live",
              horizon=a.steps + 10, seed=1, show_viewer=False, decision_watchdog=False, clip_rotation_to_yz=False)
env = GenesisIPCDressingEnv(cfg, num_envs=n)
spec = ObsSpec(targs.point_budget)
obs = env.reset([a.seed] * n)
R = up_axis_rotation(a.yaw)
c = env.cells[0]
finger, shoulder = np.asarray(c.finger, float), np.asarray(c.shoulder, float)
axis = shoulder - finger
mf = lambda v: (np.asarray(v, float) @ R.T).round(3).tolist()
P = env.positions()
g0 = np.asarray(env._anchor[0], float)
print(f"[tilt] start (model frame, rel finger): grip {mf(g0 - finger)} opening {mf(P[0][c.opening_idx].mean(0) - finger)}", flush=True)

client = WangPolicyClient(checkpoint=a.checkpoint, yaw_deg=a.yaw)
acts = np.zeros((n, env.action_dim), np.float32)
rows, best, t0 = [], dict(upper=0.0, forearm=0.0, threaded=0.0), time.time()
for k in range(a.steps):
    pos, feat, valid, _ = (x[0] for x in spec.unpack_numpy(obs))
    vm = valid.astype(bool)
    act = client.act(pos[vm], feat[vm]); act[3:] = 0.0
    acts[0] = act
    if a.expert:
        acts[1] = env.scripted_actions()[1]
    obs, _, done, infos = env.step(acts)
    P = env.positions()
    g = np.asarray(env._anchor[0], float)
    along = float((g - finger) @ axis / (axis @ axis))
    inf = infos[0]
    row = dict(k=k, act=mf(act[:3]), grip=mf(g - finger), open=mf(P[0][c.opening_idx].mean(0) - finger), along=round(along, 3),
               forearm=round(float(inf.get("forearm_ratio", 0)), 3), upper=round(float(inf.get("upperarm_ratio", 0)), 3),
               threaded=float(inf.get("threaded", 0)), sim_error=bool(inf.get("sim_error", False)))
    if a.expert:
        row["exp"] = dict(forearm=round(float(infos[1].get("forearm_ratio", 0)), 3), upper=round(float(infos[1].get("upperarm_ratio", 0)), 3))
    rows.append(row)
    for key in best:
        best[key] = max(best[key], row[key])
    if k % 20 == 0 or k == a.steps - 1:
        print(f"[tilt] k={k:3d} act{row['act']} grip{row['grip']} open{row['open']} along {along:+.2f} fore {row['forearm']} "
              f"upper {row['upper']} thr {row['threaded']:.0f}" + (f" | exp {row['exp']}" if a.expert else "") + f" ({time.time()-t0:.0f}s)", flush=True)
    if row["sim_error"]:
        print("[tilt] sim error", inf.get("error")); break
client.close()
json.dump(dict(args=vars(a), rows=rows, best=best), open(a.out, "w"), indent=1)
print("[tilt] best", best, flush=True)
env.close()
