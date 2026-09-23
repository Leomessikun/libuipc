"""Released policies on the gravity-hung start, with the IPC contact force fed to FMVP's FiLM.

The garment shape is the one it settles into hanging from Wang's two picker vertices
(hang_bake.py). It is placed with the picker at FMVP-PyBullet's offset from the fingertip and
turned about the vertical to the legal turn whose opening is nearest PyBullet's. Slots are given
as "<ckpt>:<force_scale>" with ckpt vision | sim | real; the force is minus the summed contact
force on the arm (the force on the garment, as FMVP sums it), times the scale, in our frame.
The last slot is the scripted expert.
"""
import argparse, json, sys, time
from dataclasses import replace
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--garment", default="tshirt_26")
ap.add_argument("--body", type=int, default=14045)
ap.add_argument("--yaw", type=float, default=267.0)
ap.add_argument("--hang", required=True, help="hang_bake.py npz")
ap.add_argument("--hang-key", default="k300")
ap.add_argument("--slots", default="vision:0,sim:0,sim:0.01,sim:0.03")
ap.add_argument("--anchors", type=int, default=48)
ap.add_argument("--drop", type=float, default=0.0)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--success", type=float, default=0.7)
ap.add_argument("--hold-after", type=int, default=30, help="decisions to hold still after upper >= 0.99")
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, "/home/ge47gax/kun/libuipc/.claude/worktrees/residual-rl/python")
from uipc_manip import dressing_live, pretrain_wang, train_sac
from uipc_manip.contact_force import vertex_forces_multi
from uipc_manip.dressing_env import GenesisIPCDressingEnv
from uipc_manip.obs import ObsSpec
from uipc_manip.wang_bridge import up_axis_rotation
from uipc_manip.wang_client import WangPolicyClient

CKPT = {"vision": "/home/ge47gax/Desktop/vision_based_policy.pt", "sim": "/home/ge47gax/Desktop/fmvp_sim.pt",
        "real": "/home/ge47gax/Desktop/fmvp_real.pt"}
PB_GRIP = np.array([-0.033, 0.106, -0.003])
PB_OPEN = np.array([-0.085, -0.135, -0.050])
R = up_axis_rotation(a.yaw)
hang = np.load(a.hang)[a.hang_key]                     # cloth relative to the picker


def rod_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


_build = dressing_live.LiveCellFactory.build
chosen = {}


def hung_build(self, garment, human):
    cell = _build(self, garment, human)
    finger = np.asarray(cell.finger, float)
    target = finger + (PB_GRIP - np.array([0, a.drop, 0])) @ R
    best = None
    for psi in range(0, 360, 5):
        cloth = hang @ rod_z(np.radians(psi)).T + target
        gap = dressing_live.garment_arm_gap(cloth, cell.faces, cell.arm_points, cell.arm_faces)
        if gap < 0.003:
            continue
        op = (cloth[cell.opening_idx].mean(0) - finger) @ R.T
        err = float(np.linalg.norm(op - PB_OPEN))
        if best is None or err < best[0]:
            best = (err, psi, gap, op, cloth)
    if best is None:
        raise SystemExit("no legal turn")
    err, psi, gap, op, cloth = best
    chosen.update(psi=psi, gap_mm=round(gap * 1000, 1), opening=op.round(3).tolist(), err_cm=round(err * 100, 1))
    print(f"[hang] body {human}: turn {psi} deg, gap {gap*1000:.1f} mm, opening rel fingertip (model) {op.round(3)} "
          f"vs PyBullet {PB_OPEN}, err {err*100:.1f} cm", flush=True)
    return replace(cell, cloth=cloth, picker_pos=target, pull_waypoints=np.stack([target, cell.pull_waypoints[-1]]))


dressing_live.LiveCellFactory.build = hung_build

slots = [(s.split(":")[0], float(s.split(":")[1])) for s in a.slots.split(",")]
n = len(slots) + 1
_, targs, _ = pretrain_wang.prepare(["teacher", "--region", "13", "--seed", "1", "--obs-mode", "wang_static_arm", "--no-obs-augment"])
cfg = replace(train_sac.dressing_config(targs), cells=tuple((a.garment, a.body) for _ in range(n)), cell_source="live",
              horizon=a.steps + 10, seed=1, show_viewer=False, decision_watchdog=False, clip_rotation_to_yz=False,
              anchor_count=a.anchors, contact_force_readout=True)
env = GenesisIPCDressingEnv(cfg, num_envs=n)
spec = ObsSpec(targs.point_budget)
obs = env.reset([a.seed] * n)
clients = {k: WangPolicyClient(checkpoint=CKPT[k], yaw_deg=a.yaw) for k in {s for s, _ in slots}}
c = env.cells[0]
finger, shoulder = np.asarray(c.finger, float), np.asarray(c.shoulder, float)
axis = shoulder - finger
P = env.positions()
print(f"[hang] settled start: grip {((env._anchor[0]-finger)@R.T).round(3)} opening {((P[0][c.opening_idx].mean(0)-finger)@R.T).round(3)}", flush=True)

force = np.zeros((n, 3))
acts = np.zeros((n, env.action_dim), np.float32)
rows, t0 = [], time.time()
succ = [None] * n
track_max = np.zeros(n)
best = [dict(upper=0.0, forearm=0.0, threaded=0.0) for _ in range(n)]
for k in range(a.steps):
    for s, (ck, scale) in enumerate(slots):
        if succ[s] is not None:                             # success: hold still, as PyBullet's episode would end
            acts[s] = 0.0
            continue
        pos, feat, valid, _ = (x[s] for x in spec.unpack_numpy(obs))
        vm = valid.astype(bool)
        act = clients[ck].act(pos[vm], feat[vm], force[s] * scale if ck != "vision" else None)
        act[3:] = 0.0
        acts[s] = act
    acts[-1] = env.scripted_actions()[-1] if succ[-1] is None else 0.0
    obs, _, done, infos = env.step(acts)
    if env._force_feature is not None:
        pairs = vertex_forces_multi(env._force_feature, cfg.dt, env._force_blocks)
        force = np.stack([-(nrm + fr).sum(0) for nrm, fr in pairs])       # force on the garment
    row = {"k": k, "slots": []}
    for s in range(n):
        inf = infos[s]
        g = np.asarray(env._anchor[s], float)
        r = dict(along=round(float((g - finger) @ axis / (axis @ axis)), 3), forearm=round(float(inf.get("forearm_ratio", 0)), 3),
                 upper=round(float(inf.get("upperarm_ratio", 0)), 3), threaded=float(inf.get("threaded", 0)),
                 force=np.round(force[s] @ R.T, 3).tolist(), sim_error=bool(inf.get("sim_error", False)),
                 track_mm=round(float(inf.get("tracking_error", 0)) * 1000, 1), grasp_valid=bool(inf.get("grasp_valid", True)))
        if succ[s] is None:
            track_max[s] = max(track_max[s], r["track_mm"])
            if r["upper"] >= a.success:
                succ[s] = dict(k=k, track_max_mm=float(track_max[s]), force_N=float(np.linalg.norm(force[s])))
        row["slots"].append(r)
        for key in best[s]:
            best[s][key] = max(best[s][key], r[key])
    rows.append(row)
    if k % 25 == 0 or k == a.steps - 1:
        names = [f"{ck}:{sc:g}" for ck, sc in slots] + ["expert"]
        txt = " | ".join(f"{nm} 沿{r['along']:+.2f} 前{r['forearm']:.2f} 上{r['upper']:.2f} 穿{r['threaded']:.0f} F{np.linalg.norm(r['force']):.0f} trk{r['track_mm']:.0f}"
                         for nm, r in zip(names, row["slots"]))
        print(f"[hang] k={k:3d} ({time.time()-t0:.0f}s) {txt}", flush=True)
    if all(x is not None and k - x["k"] >= a.hold_after for x in succ):
        break
    if any(r["sim_error"] for r in row["slots"]):
        print("[hang] sim error"); break
for cl in clients.values():
    cl.close()
final = [dict(upper_final=rows[-1]["slots"][s]["upper"], success=succ[s]) for s in range(n)]
json.dump(dict(args=vars(a), chosen=chosen, slots=slots, rows=rows, best=best, final=final), open(a.out, "w"), indent=1)
print("[hang] best", json.dumps(best), flush=True)
print("[hang] final", json.dumps(final), flush=True)
env.close()
