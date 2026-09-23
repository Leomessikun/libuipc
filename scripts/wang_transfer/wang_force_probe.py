"""Is the gripper's load a reliable signal? Same commands in two slots, force read two ways.

Slot 0 runs fmvp_sim (force input zero) from the gravity-hung start; slot 1 receives slot 0's
exact commands every decision, so any difference between them is the simulator's, not the
policy's. Slots 2 and 3 do the same for the scripted expert. Per decision it records:

* the gripper load, the sensor reading a real wrist would give: the hold is a spring of stiffness
  constraint_strength * m_i / dt^2 per held vertex, so F = -sum k_i (target_i - x_i) is the force
  the garment exerts on the gripper (vertex mass from the areal density, 2 * density * thickness);
* the arm contact force from the IPC contact export, for comparison.
"""
import argparse, json, sys, time
from dataclasses import replace
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--body", type=int, default=14046)
ap.add_argument("--hang", required=True)
ap.add_argument("--steps", type=int, default=260)
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--yaw", type=float, default=267.0)
ap.add_argument("--drop", type=float, default=0.0)
ap.add_argument("--hang-key", default="k300")
ap.add_argument("--out", required=True)
a = ap.parse_args()
mine = a

sys.path.insert(0, "/tmp/claude-4102472/-home-ge47gax-kun-libuipc/9eae3121-9812-4580-bceb-cfa79670c932/scratchpad")
sys.argv = [sys.argv[0], "--hang", a.hang, "--body", str(a.body), "--slots", "sim:0", "--steps", "0", "--out", "/dev/null"]
# Reuse the hung placement from wang_hang_run3 without running its loop.
src = open("/tmp/claude-4102472/-home-ge47gax-kun-libuipc/9eae3121-9812-4580-bceb-cfa79670c932/scratchpad/wang_hang_run3.py").read()
head = src.split("slots = [(s.split")[0]
exec(compile(head, "wang_hang_run3_head", "exec"))
a = mine

from uipc_manip.contact_force import vertex_forces_multi

n = 4
_, targs, _ = pretrain_wang.prepare(["teacher", "--region", "13", "--seed", "1", "--obs-mode", "wang_static_arm", "--no-obs-augment"])
cfg = replace(train_sac.dressing_config(targs), cells=tuple(("tshirt_26", a.body) for _ in range(n)), cell_source="live",
              horizon=a.steps + 10, seed=1, show_viewer=False, decision_watchdog=False, clip_rotation_to_yz=False,
              contact_force_readout=True)
env = GenesisIPCDressingEnv(cfg, num_envs=n)
spec = ObsSpec(targs.point_budget)
obs = env.reset([a.seed] * n)
client = WangPolicyClient(checkpoint=CKPT["sim"], yaw_deg=a.yaw)
c = env.cells[0]
finger, shoulder = np.asarray(c.finger, float), np.asarray(c.shoulder, float)
axis = shoulder - finger

# Lumped vertex mass on the rest mesh: areal density (2 r rho, libuipc's thickness is a half-thickness) x area / 3.
areal = 2.0 * cfg.cloth_thickness * cfg.cloth_density
tri = c.cloth[c.faces]
area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
mass = np.zeros(len(c.cloth)); np.add.at(mass, c.faces.ravel(), np.repeat(area / 3.0, 3))
mass *= areal
print(f"[probe] garment mass {mass.sum():.3f} kg, held {len(env._pickers[0]['anchor_idx'])} vertices", flush=True)


def grip_force(i, P):
    idx = env._pickers[i]["anchor_idx"]
    target = env._anchor[i][None, :] + env._offsets[i]
    k = cfg.constraint_strength * mass[idx] / cfg.dt ** 2
    return -(k[:, None] * (target - P[i][idx])).sum(0)       # garment on gripper


acts = np.zeros((n, env.action_dim), np.float32)
rows, t0 = [], time.time()
for k in range(a.steps):
    pos, feat, valid, _ = (x[0] for x in spec.unpack_numpy(obs))
    vm = valid.astype(bool)
    act = client.act(pos[vm], feat[vm]); act[3:] = 0.0
    acts[0] = acts[1] = act
    e = env.scripted_actions()
    acts[2] = acts[3] = e[2]
    obs, _, done, infos = env.step(acts)
    P = env.positions()
    pairs = vertex_forces_multi(env._force_feature, cfg.dt, env._force_blocks)
    row = {"k": k, "slots": []}
    for i in range(n):
        g = np.asarray(env._anchor[i], float)
        row["slots"].append(dict(grip=grip_force(i, P).round(3).tolist(), arm=(pairs[i][0] + pairs[i][1]).sum(0).round(3).tolist(),
                                 along=round(float((g - finger) @ axis / (axis @ axis)), 4),
                                 upper=round(float(infos[i].get("upperarm_ratio", 0)), 3),
                                 forearm=round(float(infos[i].get("forearm_ratio", 0)), 3),
                                 opening=(P[i][c.opening_idx].mean(0)).round(4).tolist()))
    rows.append(row)
    if k % 20 == 0:
        s = row["slots"]
        print(f"[probe] k={k:3d} pol |Fg| {np.linalg.norm(s[0]['grip']):6.2f} vs twin {np.linalg.norm(s[1]['grip']):6.2f}  |Farm| {np.linalg.norm(s[0]['arm']):6.1f} vs {np.linalg.norm(s[1]['arm']):6.1f}  "
              f"upper {s[0]['upper']:.2f}/{s[1]['upper']:.2f} | exp |Fg| {np.linalg.norm(s[2]['grip']):6.2f} vs {np.linalg.norm(s[3]['grip']):6.2f} "
              f"|Farm| {np.linalg.norm(s[2]['arm']):6.1f} vs {np.linalg.norm(s[3]['arm']):6.1f} upper {s[2]['upper']:.2f}/{s[3]['upper']:.2f} ({time.time()-t0:.0f}s)", flush=True)
client.close()
json.dump(dict(args=vars(a), mass=float(mass.sum()), rows=rows), open(a.out, "w"))
env.close()
