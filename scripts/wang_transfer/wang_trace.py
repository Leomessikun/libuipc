"""Step-by-step trace of the released policy (slot 0) beside the expert (slot 1), in the model frame.

Logs, per decision: the commanded model-frame action, the anchor's actual displacement (to see
steps the 12 mm no-move rule drops), the gripper and the opening centre relative to the fingertip
in model coordinates, and forearm ratio / threaded / tracking error.
"""
import argparse, json, sys, time
from dataclasses import replace
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--garment", default="tshirt_26")
ap.add_argument("--body", type=int, default=14045)
ap.add_argument("--yaw", type=float, default=267.0)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--checkpoint", default="/home/ge47gax/Desktop/vision_based_policy.pt")
ap.add_argument("--no-collision-rule", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, "/home/ge47gax/kun/libuipc/.claude/worktrees/residual-rl/python")
from uipc_manip import pretrain_wang, train_sac
from uipc_manip.dressing_env import GenesisIPCDressingEnv
from uipc_manip.obs import ObsSpec
from uipc_manip.wang_bridge import up_axis_rotation, to_reference_cloud
from uipc_manip.wang_client import WangPolicyClient

_, targs, _ = pretrain_wang.prepare(["teacher", "--region", "13", "--seed", "1", "--obs-mode", "wang_static_arm", "--no-obs-augment"])
cells = ((a.garment, a.body), (a.garment, a.body))
kw = dict(no_move_collision_threshold=0.0) if a.no_collision_rule else {}
cfg = replace(train_sac.dressing_config(targs), cells=cells, cell_source="live", horizon=a.steps + 10, seed=1,
              show_viewer=False, decision_watchdog=False, clip_rotation_to_yz=False, **kw)
env = GenesisIPCDressingEnv(cfg, num_envs=2)
spec = ObsSpec(targs.point_budget)
obs = env.reset([1000, 1000])
R = up_axis_rotation(a.yaw)
c = env.cells[0]
finger, shoulder = np.asarray(c.finger, float), np.asarray(c.shoulder, float)
elbow = np.asarray(getattr(c, "elbow", finger), float)
client = WangPolicyClient(checkpoint=a.checkpoint, yaw_deg=a.yaw)
m = lambda v: (np.asarray(v, float) @ R.T).round(3).tolist()

pos, feat, valid, _ = (x[0] for x in spec.unpack_numpy(obs))
rp, rx = to_reference_cloud(pos[valid.astype(bool)], feat[valid.astype(bool)], yaw_deg=a.yaw)
print(f"[trace] voxelised obs: arm {int(rx[:,0].sum())} cloth {int(rx[:,1].sum())}; arm extent "
      f"{rp[rx[:,0]==1].min(0).round(2)}..{rp[rx[:,0]==1].max(0).round(2)}; cloth {rp[rx[:,1]==1].min(0).round(2)}..{rp[rx[:,1]==1].max(0).round(2)}", flush=True)
np.savez(a.out.replace(".json", "_obs0.npz"), pos=rp, x=rx, finger=m(finger - env._anchor[0]), elbow=m(elbow - env._anchor[0]), shoulder=m(shoulder - env._anchor[0]))

rows, acts, t0 = [], np.zeros((2, env.action_dim), np.float32), time.time()
for k in range(a.steps):
    pos, feat, valid, _ = (x[0] for x in spec.unpack_numpy(obs))
    vm = valid.astype(bool)
    act = client.act(pos[vm], feat[vm]); act[3:] = 0.0
    acts[0] = act
    acts[1] = env.scripted_actions()[1]
    before = [np.asarray(env._anchor[s], float).copy() for s in range(2)]
    obs, _, done, infos = env.step(acts)
    P = env.positions()
    row = {"k": k, "act_model": m(act[:3])}
    for s, tag in ((0, "pol"), (1, "exp")):
        g = np.asarray(env._anchor[s], float)
        op = P[s][c.opening_idx].mean(axis=0)
        row[tag] = dict(moved_mm=round(float(np.linalg.norm(g - before[s])) * 1000, 2),
                        grip_rel_finger=m(g - finger), open_rel_finger=m(op - finger),
                        forearm=round(float(infos[s].get("forearm_ratio", 0)), 3),
                        upper=round(float(infos[s].get("upperarm_ratio", 0)), 3),
                        threaded=float(infos[s].get("threaded", 0)),
                        track_mm=round(float(infos[s].get("tracking_error", 0)) * 1000, 1))
    rows.append(row)
    if k % 10 == 0:
        p, e = row["pol"], row["exp"]
        print(f"[trace] k={k:3d} act{row['act_model']} moved {p['moved_mm']}mm grip{p['grip_rel_finger']} open{p['open_rel_finger']} "
              f"fore {p['forearm']} trk {p['track_mm']} | exp grip{e['grip_rel_finger']} open{e['open_rel_finger']} fore {e['forearm']} ({time.time()-t0:.0f}s)", flush=True)
client.close()
json.dump(dict(args=vars(a), finger_model=m(finger), rows=rows), open(a.out, "w"), indent=1)
env.close()
