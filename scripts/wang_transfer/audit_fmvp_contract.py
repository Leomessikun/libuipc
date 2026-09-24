"""Check the bridge against the official actor and quantify input alterations.

Run with the curl Python environment. Uses saved observations; no simulation,
plotting, training or changes to checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.claude/worktrees/residual-rl/python'))
sys.path.insert(0, '/home/ge47gax/kun/fmvp_pb/dressing_pb')
sys.path.insert(0, '/home/ge47gax/kun')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('/home/ge47gax/Desktop/fmvp_sim.pt'))
    a = p.parse_args()
    import torch
    from torch_geometric.data import Batch, Data
    from uipc_manip.wang_bridge import ENCODER_ARGS, ReferencePolicy, _install_compat, to_reference_cloud
    from uipc_manip.obs import ObsSpec
    _install_compat()
    from algo.SAC_AWAC import Actor

    torch.set_num_threads(2)
    args = types.SimpleNamespace(**ENCODER_ARGS, film_force=True, use_force_hist=False,
                                 freeze_weights=False, freeze_encoder_only=False)
    actor = Actor((30000,), (6,), 1024, 'pointcloud_flow', 50, -10, 2, 3, 32, args)
    state = torch.load(a.checkpoint, map_location='cpu')['model_state_dict']
    actor.load_state_dict(state, strict=True)
    actor.eval()
    bridge = ReferencePolicy(a.checkpoint, yaw_deg=267, voxel=0)
    rotation = bridge.rotation
    rows = []
    for path in sorted((ROOT/'output/uipc_manip/fmvp_pb_robot_traj_verified_20260923').glob('*_traj.npz')):
        with np.load(path) as d:
            for t in np.unique(np.linspace(0, len(d['obs_ptr'])-2, 8, dtype=int)):
                start, end = d['obs_ptr'][t:t+2]
                pos, features = d['obs_pos'][start:end], d['obs_x'][start:end]
                f = d['force'][t].reshape(3)
                batch = Batch.from_data_list([Data(pos=torch.as_tensor(pos), x=torch.as_tensor(features))])
                flags = np.zeros((len(pos), 4), np.float32)
                flags[:, 1], flags[:, 0], flags[:, 3] = features.T
                world_pos, world_force = pos @ rotation, f @ rotation
                with torch.no_grad():
                    expected = actor(batch, torch.as_tensor(f[None]), compute_pi=False, compute_log_pi=False)[0][-1].numpy()
                world_action = bridge.act(world_pos, flags, world_force)
                actual = np.r_[world_action[:3] @ rotation.T, world_action[3:] @ rotation.T]
                no_force = bridge.act(world_pos, flags, np.zeros(3))
                rows.append(dict(path=str(path), state=int(t), max_actor_error=float(abs(actual-expected).max()),
                                 force_norm=float(np.linalg.norm(f)),
                                 zero_force_action_change=float(np.linalg.norm(no_force-world_action))))
    if max(r['max_actor_error'] for r in rows) > 1e-4:
        raise AssertionError('Official actor and bridge disagree; inspect actor parity before rollout')

    cloud_rows=[]
    dataset=ROOT/'output/uipc_manip/fmvp_dataset_500_20260924'
    for body in (14045, 14046, 14049, 14053, 14056, 14060):
        path=sorted(dataset.glob(f'batch_*_body_{body}/body_*/*baseline.npz'))[0]
        with np.load(path) as d:
            spec=ObsSpec((d['obs'].shape[1]-7)//7)
            positions, features, valid, _=spec.unpack_numpy(d['obs'])
            for t in np.unique(np.linspace(0, len(positions)-1, 9, dtype=int)):
                pos, feat=positions[t,valid[t].astype(bool)], features[t,valid[t].astype(bool)]
                cloud0,x0=to_reference_cloud(pos,feat,yaw_deg=267,voxel=0)
                cloud1,x1=to_reference_cloud(pos,feat,yaw_deg=267,voxel=.0625)
                bridge.voxel=0
                action0=bridge.act(pos,feat)
                bridge.voxel=.0625
                action1=bridge.act(pos,feat)
                cloud_rows.append(dict(body=body,path=str(path),state=int(t),
                    arm_before=int(x0[:,0].sum()),cloth_before=int(x0[:,1].sum()),
                    arm_after=int(x1[:,0].sum()),cloth_after=int(x1[:,1].sum()),
                    action_change=float(np.linalg.norm(action0-action1))))
    summary=dict(actor_cases=len(rows),max_actor_error=max(r['max_actor_error'] for r in rows),
        max_zero_force_action_change=max(r['zero_force_action_change'] for r in rows),
        median_zero_force_action_change=float(np.median([r['zero_force_action_change'] for r in rows])),
        input_cases=len(cloud_rows),median_cloth_fraction_removed=float(np.median([
            1-r['cloth_after']/max(1,r['cloth_before']) for r in cloud_rows])),
        median_double_voxel_action_change=float(np.median([r['action_change'] for r in cloud_rows])),
        max_double_voxel_action_change=max(r['action_change'] for r in cloud_rows))
    report=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),summary=summary,
                actor_cases=rows,input_cases=cloud_rows)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
