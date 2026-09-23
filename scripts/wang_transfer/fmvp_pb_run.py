"""Run a released checkpoint inside FMVP's own PyBullet env, with FMVP's own action transform.

Ground truth for the bridge: does the policy dress in *a* simulator other than softgym, and what
does its observation look like there (point counts, extents, arm direction in the model frame).
Run under the `dressing` conda env from fmvp_pb/dressing_pb (see prepare.sh).
"""
import argparse, hashlib, json, os, sys, types
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="/home/ge47gax/Desktop/vision_based_policy.pt")
ap.add_argument("--garment", type=int, default=1)
ap.add_argument("--motion", type=int, default=0)
ap.add_argument("--elbow-deg", type=float, default=-90.)
ap.add_argument("--shoulder-deg", type=float, default=80.)
ap.add_argument("--horizon", type=int, default=250)
ap.add_argument("--seed", type=int, default=1004)
ap.add_argument("--action-scale", type=float, default=0.025)
ap.add_argument("--zero-force", action="store_true")
ap.add_argument("--hold-from-state", type=int, default=None,
                help="Replace actor commands with zero Cartesian motion from this recorded state onward.")
ap.add_argument("--hold-steps", type=int, default=10,
                help="Number of zero-motion decisions to record after --hold-from-state.")
ap.add_argument("--out", required=True)
ap.add_argument("--save-traj", action="store_true",
                help="Save cloth, FMVP observation, Sawyer joints and TCP at every step beside the JSON trace.")
a = ap.parse_args()
if a.hold_from_state is not None and not (0 <= a.hold_from_state < a.horizon):
    ap.error("--hold-from-state must be in [0, horizon)")
if a.hold_steps < 1:
    ap.error("--hold-steps must be positive")
np.random.seed(a.seed)
torch.manual_seed(a.seed)

from algo.SAC_AWAC import Actor
from assistive_gym.dressing_envs import DressingSawyerHumanEnv
import assistive_gym
import pybullet as p
from assistive_gym.garment_idx_utils import shoulder_polygon_particle_indices, grasping_particle_indices

# FMVP's arm-motion files are opened relative to the dressing_pb checkout.
a.out = os.path.abspath(a.out)
os.chdir(os.path.dirname(os.path.dirname(assistive_gym.__file__)))

required_bullet_api = ("RESET_USE_DEFORMABLE_WORLD", "MESH_DATA_SIMULATION_MESH",
                       "createSoftBodyAnchor", "getSoftBodyData")
missing_bullet_api = [name for name in required_bullet_api if not hasattr(p, name)]
if missing_bullet_api:
    raise RuntimeError("FMVP requires the pybullet_3_0_9 branch of Zackory/bullet3; "
                       "missing APIs: " + ", ".join(missing_bullet_api))

payload = torch.load(a.checkpoint, map_location="cpu")
state = payload["model_state_dict"]
film = any("film_layers" in k for k in state)
args = types.SimpleNamespace(pc_feature_dim=3, pc_num_layers=3, sa_radius=[0.05, 0.1], sa_ratio=[1, 1],
                             sa_mlp_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
                             linear_mlp_list=[128, 128], fp_mlp_list=[[256, 256], [256, 128], [128, 128, 128]],
                             fp_k=[1, 3, 3], film_force=film, use_force_hist=False, freeze_weights=False,
                             freeze_encoder_only=False)
actor = Actor((30000,), (6,), 1024, "pointcloud_flow", 50, -10, 2, 3, 32, args)
actor.load_state_dict(state, strict=True)
actor.eval()
print(f"[pb] {os.path.basename(a.checkpoint)} film={film} step={payload.get('step')}", flush=True)

gif = os.path.join(os.path.dirname(a.out), "gifs"); os.makedirs(gif, exist_ok=True)
env = DressingSawyerHumanEnv(gender="female", body_size="default", policy=0, horizon=a.horizon,
                             camera_pos="front", render=False, gif_path=gif, mass=0.16, friction=0.5,
                             repulsion=0, elbow_rand=a.elbow_deg, shoulder_rand=a.shoulder_deg)
obs, force = env.reset(garment_id=a.garment, motion_id=a.motion, pose_id=-1, step_idx=0)
sawyer_base_pos, sawyer_base_quat = p.getBasePositionAndOrientation(env.robot.body,
                                                                    physicsClientId=env.id)
rows, done, t = [], False, 0
states = {k: [] for k in ("cloth", "line", "tcp", "tcp_quat", "robot_q", "force", "obs_pos", "obs_x")}
actions_raw, actions_world, rewards, dones = [], [], [], []

def record_state(observation, force_vector):
    _, verts = p.getMeshData(env.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH,
                             physicsClientId=env.id)
    states["cloth"].append(np.asarray(verts, dtype=np.float32))
    states["line"].append(np.asarray(env.line_points, dtype=np.float32))
    tcp_pos, tcp_quat = env.robot.get_pos_orient(env.robot.left_end_effector)
    states["tcp"].append(np.asarray(tcp_pos, dtype=np.float32))
    states["tcp_quat"].append(np.asarray(tcp_quat, dtype=np.float32))
    states["robot_q"].append(np.asarray(env.robot.get_joint_angles(env.robot.controllable_joint_indices), dtype=np.float32))
    states["force"].append(np.asarray(force_vector, dtype=np.float32).reshape(-1))
    states["obs_pos"].append(observation.pos.numpy().astype(np.float32))
    states["obs_x"].append(observation.x.numpy().astype(np.float32))

if a.save_traj:
    record_state(obs, force)
while not done:
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([obs])
    f = torch.zeros(1, 3) if (a.zero_force or not film) else torch.as_tensor(np.asarray(force), dtype=torch.float32).reshape(1, 3)
    with torch.no_grad():
        mu, _, _, _ = actor(batch, f, compute_pi=False, compute_log_pi=False)
        act = mu.numpy().reshape(-1, 6)[-1].copy()  # last point is the gripper
    if not np.isfinite(act).all() or np.max(np.abs(act)) > 1.000001:
        raise RuntimeError("FMVP action must be finite and squashed to [-1, 1]")
    x, y, z = act[:3] * a.action_scale
    step_action = np.zeros(6)
    step_action[:3] = [z, x, y]
    yr = act[4]
    dtheta = abs(yr)
    if dtheta > np.deg2rad(5):
        dtheta *= np.deg2rad(5) / np.sqrt(3)
    step_action[5] = np.sign(yr) * dtheta
    external_hold = a.hold_from_state is not None and t >= a.hold_from_state
    if external_hold:
        step_action[:] = 0.0
    # observation statistics in the model frame, gripper-centred
    pos, xf = obs.pos.numpy(), obs.x.numpy()
    arm, cloth = pos[xf[:, 0] == 1], pos[xf[:, 1] == 1]
    lp = np.asarray(env.line_points)[:, [1, 2, 0]] - env.ee_pose.reshape(3)   # finger, elbow, shoulder
    _, verts = p.getMeshData(env.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=env.id)
    verts = np.asarray(verts)[:, [1, 2, 0]]
    ee = env.ee_pose.reshape(3)
    open_c = verts[shoulder_polygon_particle_indices[env.garment]].mean(0) - ee
    grasp_c = verts[grasping_particle_indices[env.garment]].mean(0) - ee
    obs_next, reward, env_done, info, force = env.step(step_action)
    done = bool(env_done or (external_hold and t + 1 >= a.hold_from_state + a.hold_steps))
    if a.save_traj:
        actions_raw.append(act.astype(np.float32))
        actions_world.append(step_action.astype(np.float32))
        rewards.append(float(reward))
        dones.append(bool(done))
        record_state(obs_next, force)
    rows.append(dict(t=t, act=act.round(3).tolist(), external_hold=external_hold,
                     n_arm=len(arm), n_cloth=len(cloth),
                     arm_min=arm.min(0).round(3).tolist() if len(arm) else None,
                     arm_max=arm.max(0).round(3).tolist() if len(arm) else None,
                     cloth_min=cloth.min(0).round(3).tolist() if len(cloth) else None,
                     cloth_max=cloth.max(0).round(3).tolist() if len(cloth) else None,
                     finger=lp[0].round(3).tolist(), elbow=lp[1].round(3).tolist(), shoulder=lp[2].round(3).tolist(),
                     force=np.asarray(force).reshape(-1).round(3).tolist(),
                     upper=float(info["upperarm_ratio"]), open_rel_grip=open_c.round(3).tolist(), grasp_rel_grip=grasp_c.round(3).tolist(), whole=float(info["whole_arm_ratio"])))
    if t == 0:
        np.savez(a.out.replace(".json", "_obs0.npz"), pos=pos, x=xf, line=lp)
    if t % 5 == 0 and t <= 40 or t % 25 == 0:
        print(f"[pb] t={t} act={act.round(2)} arm {len(arm)} cloth {len(cloth)} upper {info['upperarm_ratio']:.3f} "
              f"whole {info['whole_arm_ratio']:.3f} open-grip {open_c.round(3)} finger-grip {lp[0].round(3)} grasp-grip {grasp_c.round(3)}", flush=True)
    obs, t = obs_next, t + 1
json.dump(rows, open(a.out, "w"))
if a.save_traj:
    obs_ptr = np.r_[0, np.cumsum([len(x) for x in states["obs_pos"]])]
    trajectory_path = os.path.splitext(a.out)[0] + "_traj.npz"
    np.savez_compressed(
        trajectory_path, cloth=np.asarray(states["cloth"]), line=np.asarray(states["line"]),
        tcp=np.asarray(states["tcp"]), tcp_quat=np.asarray(states["tcp_quat"]),
        robot_q=np.asarray(states["robot_q"]),
        robot_joint_indices=np.asarray(env.robot.controllable_joint_indices, dtype=np.int32),
        force=np.asarray(states["force"]), obs_pos=np.concatenate(states["obs_pos"]),
        obs_x=np.concatenate(states["obs_x"]), obs_ptr=obs_ptr,
        action_raw=np.asarray(actions_raw), action_world=np.asarray(actions_world),
        reward=np.asarray(rewards), done=np.asarray(dones),
        upperarm_ratio=np.r_[np.nan, [r["upper"] for r in rows]],
        opening_idx=np.asarray(shoulder_polygon_particle_indices[env.garment], dtype=np.int32),
        metadata_json=json.dumps({"checkpoint": a.checkpoint,
                                  "checkpoint_sha256": hashlib.sha256(open(a.checkpoint, "rb").read()).hexdigest(),
                                  "garment": a.garment, "motion": a.motion, "horizon": a.horizon,
                                  "seed": a.seed,
                                  "elbow_deg": a.elbow_deg, "shoulder_deg": a.shoulder_deg,
                                  "mass_kg": env.mass, "robot": "Sawyer",
                                  "hold_from_state": a.hold_from_state,
                                  "hold_steps": a.hold_steps if a.hold_from_state is not None else None,
                                  "robot_base_pos": sawyer_base_pos,
                                  "robot_base_quat": sawyer_base_quat,
                                  "actor_action_postprocess": "official FMVP Actor.forward (single squash)",
                                  "grasp": "soft-body anchors; no physical gripper closure",
                                  "robot_cloth_collision": "disabled in FMVP PyBullet environment",
                                  "states": len(states["cloth"]), "actions": len(actions_raw)}))
    print("[pb] saved aligned robot and cloth trajectory:", trajectory_path, flush=True)
print(f"[pb] final upper {rows[-1]['upper']:.3f} whole {rows[-1]['whole']:.3f} after {t} steps", flush=True)
