"""Flow-matching action-chunk policy on top of the FMVP point-cloud encoder.

Observation: the checkpoint's own input (arm and garment cloud, gripper-centred, model frame) through the
frozen FMVP PointNet++ encoder, summarised as the gripper point's 50-d feature plus max and mean pools over
arm and garment points (250-d), and the gripper load (model frame, newtons / 100, norm-clipped at 3).
Action: a chunk of ``horizon`` future actions, each (dx, dy, dz) in the model frame in the environment's
unit action and the rotation about the vertical in its unit action. A conditional flow (rectified flow,
linear path from noise) generates the chunk; inference integrates ``steps`` Euler steps.

``obs_features`` is the single definition used for training data and for serving, so the two cannot drift.
Runs in the ``curl`` environment (torch_geometric) like ``wang_bridge``.
"""
from __future__ import annotations

import math

import numpy as np

FEATURE_DIM = 250
FORCE_DIM = 3
ACTION_DIM = 4


def obs_features(policy, pos_rel, flags, torch):
    """Encoder summary of one observation, given a ``wang_bridge.ReferencePolicy`` (for its encoder and frame)."""
    from torch_geometric.data import Batch, Data

    from .wang_bridge import ARM, CLOTH, GRIPPER, to_reference_cloud

    pos, x = to_reference_cloud(pos_rel, flags, yaw_deg=policy.yaw_deg, voxel=policy.voxel)
    batch = Batch.from_data_list([Data(x=torch.as_tensor(x), pos=torch.as_tensor(pos))]).to(policy.device)
    with torch.no_grad():
        out = policy.encoder(batch, torch.zeros(1, 3, device=policy.device)) if policy.film else policy.encoder(batch)
        feat = out[0]
        parts = [feat[batch.x[:, GRIPPER] == 1][0]]
        for kind in (ARM, CLOTH):
            sel = feat[batch.x[:, kind] == 1]
            if len(sel) == 0:
                sel = torch.zeros(1, feat.shape[1], device=feat.device)
            parts += [sel.max(0).values, sel.mean(0)]
    return torch.cat(parts).cpu().numpy().astype(np.float32)


def force_feature(gripper_force_ours, rotation):
    f = np.asarray(gripper_force_ours, np.float64) / 100.0
    n = np.linalg.norm(f)
    if n > 3.0:
        f *= 3.0 / n
    return (f @ rotation.T).astype(np.float32)


def build_net(torch, horizon, width=512, depth=4):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

        def forward(self, h):
            return h + self.ff(self.norm(h))

    class VelocityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.horizon = horizon
            self.cond = nn.Sequential(nn.Linear(FEATURE_DIM + FORCE_DIM, width), nn.GELU(), nn.Linear(width, width))
            self.inp = nn.Linear(horizon * ACTION_DIM + 64, width)
            self.blocks = nn.ModuleList(Block() for _ in range(depth))
            self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, horizon * ACTION_DIM))
            self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000.0), 32)))

        def forward(self, x, t, cond):
            te = t[:, None] * self.freqs[None]
            te = torch.cat([te.sin(), te.cos()], 1)
            h = self.inp(torch.cat([x.flatten(1), te], 1)) + self.cond(cond)
            for block in self.blocks:
                h = block(h)
            return self.out(h).view(-1, self.horizon, ACTION_DIM)

    return VelocityNet()


class FlowPolicy:
    """Serves a trained flow policy through the ``wang_bridge`` interface: ``act(pos_rel, flags, force)``."""

    def __init__(self, checkpoint, device="cpu", yaw_deg=267.0, voxel=None, steps=10, seed=0):
        import torch

        from .wang_bridge import VOXEL_SIZE, ReferencePolicy

        self.torch = torch
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        meta = payload["flow_policy"]
        self.encoder_policy = ReferencePolicy(meta["encoder_checkpoint"], device=device, yaw_deg=yaw_deg,
                                              voxel=VOXEL_SIZE if voxel is None else voxel)
        self.rotation = self.encoder_policy.rotation
        self.net = build_net(torch, meta["horizon"], meta["width"], meta["depth"]).to(device).eval()
        self.net.load_state_dict(payload["state_dict"])
        self.stats = {k: torch.as_tensor(np.asarray(v, np.float32), device=device) for k, v in meta["stats"].items()}
        self.device, self.steps = device, int(steps)
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.step = int(payload.get("step", -1))
        self.best_return = None

    def chunk(self, pos_rel, flags, force=None):
        torch = self.torch
        feat = obs_features(self.encoder_policy, pos_rel, flags, torch)
        f = force_feature(np.zeros(3) if force is None else force, self.rotation)
        cond = torch.as_tensor(np.concatenate([feat, f])[None], device=self.device)
        cond = (cond - self.stats["cond_mean"]) / self.stats["cond_std"]
        x = torch.randn(1, self.net.horizon, ACTION_DIM, generator=self.generator, device=self.device)
        with torch.no_grad():
            for k in range(self.steps):
                t = torch.full((1,), k / self.steps, device=self.device)
                x = x + self.net(x, t, cond) / self.steps
        a = x[0] * self.stats["act_std"] + self.stats["act_mean"]
        return a.clamp(-1, 1).cpu().numpy()

    def act(self, pos_rel, flags, force=None):
        """First action of the chunk as the environment's 6-D action in our frame (receding horizon)."""
        a = self.chunk(pos_rel, flags, force)[0]
        out = np.zeros(6, np.float32)
        out[:3] = a[:3] @ self.rotation            # model -> ours (rows of ``rotation`` are model axes)
        out[5] = a[3]
        return out
