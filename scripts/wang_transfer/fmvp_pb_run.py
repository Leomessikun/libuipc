"""Run a released checkpoint inside FMVP's own PyBullet env, with FMVP's own action transform.

Ground truth for the bridge: does the policy dress in *a* simulator other than softgym, and what
does its observation look like there (point counts, extents, arm direction in the model frame).
Run under the `dressing` conda env from fmvp_pb/dressing_pb (see prepare.sh).
"""
import argparse, json, os, sys, types
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="/home/ge47gax/Desktop/vision_based_policy.pt")
ap.add_argument("--garment", type=int, default=1)
ap.add_argument("--motion", type=int, default=0)
ap.add_argument("--horizon", type=int, default=250)
ap.add_argument("--action-scale", type=float, default=0.025)
ap.add_argument("--zero-force", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

from models.encoder import make_encoder
from assistive_gym.dressing_envs import DressingSawyerHumanEnv
import pybullet as p
from assistive_gym.garment_idx_utils import shoulder_polygon_particle_indices, grasping_particle_indices

payload = torch.load(a.checkpoint, map_location="cpu")
state = payload["model_state_dict"]
film = any("film_layers" in k for k in state)
args = types.SimpleNamespace(pc_feature_dim=3, pc_num_layers=3, sa_radius=[0.05, 0.1], sa_ratio=[1, 1],
                             sa_mlp_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
                             linear_mlp_list=[128, 128], fp_mlp_list=[[256, 256], [256, 128], [128, 128, 128]],
                             fp_k=[1, 3, 3], film_force=film, use_force_hist=False, freeze_weights=False,
                             freeze_encoder_only=False)
encoder = make_encoder("pointcloud_flow", None, 50, 3, 32, args, output_logits=True, residual=False)
trunk = nn.Sequential(nn.Linear(50, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 12))
encoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("encoder.")}, strict=True)
trunk.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("trunk.")}, strict=True)
encoder.eval(); trunk.eval()
print(f"[pb] {os.path.basename(a.checkpoint)} film={film} step={payload.get('step')}", flush=True)

gif = os.path.join(os.path.dirname(a.out), "gifs"); os.makedirs(gif, exist_ok=True)
env = DressingSawyerHumanEnv(gender="female", body_size="default", policy=0, horizon=a.horizon,
                             camera_pos="front", render=False, gif_path=gif, mass=0.16, friction=0.5,
                             repulsion=0, elbow_rand=-90, shoulder_rand=80)
obs, force = env.reset(garment_id=a.garment, motion_id=a.motion, pose_id=-1, step_idx=0)
rows, done, t = [], False, 0
while not done:
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([obs])
    f = torch.zeros(1, 3) if (a.zero_force or not film) else torch.as_tensor(np.asarray(force), dtype=torch.float32).reshape(1, 3)
    with torch.no_grad():
        feat, _ = encoder(batch, f)
        mu, _ = trunk(feat).chunk(2, dim=-1)
        act = torch.tanh(mu).numpy().reshape(-1, 6)[-1].copy()      # last point is the gripper
    x, y, z = act[:3] * a.action_scale
    step_action = np.zeros(6)
    step_action[:3] = [z, x, y]
    yr = act[4]
    dtheta = abs(yr)
    if dtheta > np.deg2rad(5):
        dtheta *= np.deg2rad(5) / np.sqrt(3)
    step_action[5] = np.sign(yr) * dtheta
    # observation statistics in the model frame, gripper-centred
    pos, xf = obs.pos.numpy(), obs.x.numpy()
    arm, cloth = pos[xf[:, 0] == 1], pos[xf[:, 1] == 1]
    lp = np.asarray(env.line_points)[:, [1, 2, 0]] - env.ee_pose.reshape(3)   # finger, elbow, shoulder
    _, verts = p.getMeshData(env.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=env.id)
    verts = np.asarray(verts)[:, [1, 2, 0]]
    ee = env.ee_pose.reshape(3)
    open_c = verts[shoulder_polygon_particle_indices[env.garment]].mean(0) - ee
    grasp_c = verts[grasping_particle_indices[env.garment]].mean(0) - ee
    obs_next, reward, done, info, force = env.step(step_action)
    rows.append(dict(t=t, act=act.round(3).tolist(), n_arm=len(arm), n_cloth=len(cloth),
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
print(f"[pb] final upper {rows[-1]['upper']:.3f} whole {rows[-1]['whole']:.3f} after {t} steps", flush=True)
