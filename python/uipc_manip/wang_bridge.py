"""Run Wang's released dressing policy against observations from this project's environment.

Their checkpoint cannot be loaded into our agent: the networks differ in module names and shapes.
It can be loaded into *their* network, which is pure PyTorch and torch_geometric and needs no
simulator, and then fed an observation we build. This module is that bridge.

It runs under the `curl` conda environment (python 3.9, torch 2.1.2, torch_geometric 2.6.1), not
the Genesis venv, because their model imports `dressing.models` and `torch_scatter`:

    PYTHONPATH=/home/ge47gax/kun:<this repo>/python \
    /home/ge47gax/miniconda3/envs/curl/bin/python -m uipc_manip.wang_bridge --self-test

What the reference expects, read off `dressing/curl` and confirmed against the checkpoint's shapes:

* A torch_geometric `Data` with `pos` [N, 3] and `x` [N, 3], one graph per decision.
* `x` is a one-hot over point type, in the order `dress_env.py:808-812` writes it:
  `0` the human's right arm, `1` the garment, `2` the gripper, and the gripper is one point.
  Ours is `[is_deformable, is_marker, is_goal, is_tool]` (`obs.py`), so the map is
  marker -> 0, deformable -> 1, tool -> 2, and the goal point is dropped: they have none.
* Positions centred on the gripper (`dress_env.py:790`, `center_pointcloud_at_gripper=True`),
  which is what `obs.py::pack_labeled` already does.
* **Their world is y-up** (`gravity = np.array([0, 1, 0])`, `dress_env.py:643`) and ours is z-up.
  The yaw about that axis is 267 degrees, measured against FMVP's PyBullet deployment of this
  checkpoint, whose first observation puts the forearm on model +x and the upper arm on +z
  (agent_docs/performance/2026-09-23-wang-checkpoint-frame-and-placement.md). Alignment with the
  gripper-to-shoulder chord picked 330, which is 63 degrees off.
* Voxel-downsampled at `voxel_size = 0.00625 * 10` m (`launch_train_curl.py:239`). The encoder's
  ball-query radii (0.05, 0.1 m) are absolute, so feeding our denser cloud changes what each query
  sees; `--voxel` reproduces their density.

The actor is `SAC_AWAC.Actor`: encoder then a three-layer trunk to `2 * action_dim`, squashed.
Under the flow encoder the trunk runs per point and the gripper's row is the action, which is how
`Actor.get_logprob` reads it (`log_pi[obs.x[:, gripper_idx] == 1]`, `gripper_idx=2`). We rebuild
those two modules rather than importing `SAC_AWAC`, whose module-level `from dressing.curl.se3
import flow2pose` pulls in pytorch3d whereas the forward pass does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Their encoder hyperparameters, from `dressing/imitation/launch_train.py` and
# `dressing/curl/launch_train_curl.py` at `pc_num_layers = 3`. Every one is confirmed by a strict
# `load_state_dict` of the released checkpoint.
ENCODER_ARGS = dict(
    pc_feature_dim=3,
    pc_num_layers=3,
    sa_radius=[0.05, 0.1],
    sa_ratio=[1, 1],
    sa_mlp_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
    fp_mlp_list=[[256, 256], [256, 128], [128, 128, 128]],
    fp_k=[1, 3, 3],
    linear_mlp_list=[128, 128],
)
FEATURE_DIM = 50
HIDDEN_DIM = 1024
ACTION_DIM = 6
VOXEL_SIZE = 0.00625 * 10
GRIPPER_FEATURE = 2
FMVP_ROOT = "/home/ge47gax/kun/fmvp_pb/dressing_pb"

ARM, CLOTH, GRIPPER = 0, 1, 2


def _install_compat() -> None:
    """torch_geometric renamed PointConv to PointNetConv; their models import the old name."""
    import torch_geometric.nn as tgnn

    if not hasattr(tgnn, "PointConv") and hasattr(tgnn, "PointNetConv"):
        tgnn.PointConv = tgnn.PointNetConv


def up_axis_rotation(yaw_deg: float = 0.0) -> np.ndarray:
    """Our z-up frame to their y-up frame, with a free rotation about the shared vertical.

    The yaw turns about our own vertical first, then -90 degrees about x sends our up (0, 0, 1) to
    their up (0, 1, 0), keeping the frame right-handed: (x, y, z) -> (x, z, -y). Composing in that
    order leaves the vertical fixed at every yaw, which turning about their y before the map would
    not. The reference code does not pin the yaw down, so it stays a parameter.

    Returns a matrix to right-multiply row vectors by, as ``points @ R.T``.
    """
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    about_our_up = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    to_y_up = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return to_y_up @ about_our_up


def voxel_downsample(points: np.ndarray, voxel: float, keep: np.ndarray | None = None) -> np.ndarray:
    """Indices of one point per occupied voxel, the first in each, as open3d's downsample keeps one.

    ``keep`` marks points that survive whatever their voxel holds; the gripper is a single point and
    must not be merged away.
    """
    points = np.asarray(points, dtype=np.float64)
    if voxel <= 0.0:
        return np.arange(len(points))
    keys = np.floor(points / float(voxel)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    chosen = np.zeros(len(points), dtype=bool)
    chosen[first] = True
    if keep is not None:
        chosen |= np.asarray(keep, dtype=bool)
    return np.flatnonzero(chosen)


def to_reference_cloud(pos_rel: np.ndarray, flags: np.ndarray, *, yaw_deg: float = 0.0,
                       voxel: float = VOXEL_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Our gripper-centred cloud and 4-flag features as their cloud and 3-way one-hot.

    ``pos_rel`` is [N, 3] already relative to the tool, ``flags`` is [N, 4] in ``obs.py`` order.
    Returns their ``pos`` and ``x``. The tool point is re-inserted at the origin as their single
    gripper point, and the goal point is dropped.
    """
    pos_rel = np.asarray(pos_rel, dtype=np.float64).reshape(-1, 3)
    flags = np.asarray(flags, dtype=np.float64).reshape(-1, 4)
    if len(pos_rel) != len(flags):
        raise ValueError(f"{len(pos_rel)} positions against {len(flags)} feature rows")

    is_cloth = flags[:, 0] > 0.5      # FLAG_DEFORMABLE
    is_arm = flags[:, 1] > 0.5        # FLAG_MARKER
    is_tool = flags[:, 3] > 0.5       # FLAG_TOOL
    keep = is_cloth | is_arm | is_tool
    if not keep.any():
        raise ValueError("The observation carries no arm, cloth or tool point")

    index = voxel_downsample(pos_rel[keep], voxel, keep=is_tool[keep])
    sub_pos, sub_cloth, sub_arm, sub_tool = (a[keep][index] for a in (pos_rel, is_cloth, is_arm, is_tool))

    # Their order is arm, cloth, gripper, and the encoder is permutation-invariant, but the
    # gripper row is selected by its feature rather than its position, so the order is cosmetic.
    x = np.zeros((len(sub_pos), 3), dtype=np.float32)
    x[sub_arm, ARM] = 1.0
    x[sub_cloth, CLOTH] = 1.0
    x[sub_tool, GRIPPER] = 1.0
    if x[:, GRIPPER].sum() != 1.0:
        raise ValueError(f"Expected exactly one gripper point, found {int(x[:, GRIPPER].sum())}")

    return (sub_pos @ up_axis_rotation(yaw_deg).T).astype(np.float32), x


class ReferencePolicy:
    """Wang's released actor, loaded into their network and fed our observations."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", yaw_deg: float = 0.0,
                 voxel: float = VOXEL_SIZE) -> None:
        import types

        import torch
        import torch.nn as nn

        _install_compat()

        self.torch = torch
        self.device = device
        self.yaw_deg = float(yaw_deg)
        self.voxel = float(voxel)
        self.rotation = up_axis_rotation(self.yaw_deg)

        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        # FMVP's fine-tunes (fmvp_sim.pt, fmvp_real.pt) add FiLM layers on a 3-vector force; only
        # FMVP's encoder has them, and with FiLM off it is Wang's encoder under the same key names.
        self.film = any("film_layers" in k for k in state)
        self.step = int(payload.get("step", -1))
        self.best_return = payload.get("best_avg_return_test")

        args = types.SimpleNamespace(**ENCODER_ARGS)
        if self.film:
            import sys

            sys.path.insert(0, FMVP_ROOT)
            from models.encoder import make_encoder as make_fmvp_encoder

            fmvp_args = types.SimpleNamespace(**ENCODER_ARGS, film_force=True, use_force_hist=False,
                                              freeze_weights=False, freeze_encoder_only=False)
            self.encoder = make_fmvp_encoder("pointcloud_flow", None, FEATURE_DIM, ENCODER_ARGS["pc_num_layers"],
                                             32, fmvp_args, output_logits=True, residual=False)
        else:
            from dressing.curl.encoder import make_encoder

            self.encoder = make_encoder("pointcloud_flow", None, FEATURE_DIM, ENCODER_ARGS["pc_num_layers"],
                                        32, args, output_logits=True, residual=False)
        self.trunk = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2 * ACTION_DIM),
        )
        # strict: a silently partial load is the failure mode that looks like "it does not transfer".
        self.encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                                      if k.startswith("encoder.")}, strict=True)
        self.trunk.load_state_dict({k[len("trunk."):]: v for k, v in state.items()
                                    if k.startswith("trunk.")}, strict=True)
        self.encoder.to(device).eval()
        self.trunk.to(device).eval()

    def act(self, pos_rel: np.ndarray, flags: np.ndarray, force: np.ndarray | None = None) -> np.ndarray:
        """The deterministic action for one observation, in *our* frame.

        ``force`` is the FiLM input of FMVP's fine-tunes, a 3-vector in our frame (the force on the
        garment from the arm, as FMVP sums it); it is rotated into the model frame and ignored by a
        checkpoint without FiLM. None means zero.
        """
        from torch_geometric.data import Batch, Data

        torch = self.torch
        pos, x = to_reference_cloud(pos_rel, flags, yaw_deg=self.yaw_deg, voxel=self.voxel)
        batch = Batch.from_data_list([Data(x=torch.as_tensor(x), pos=torch.as_tensor(pos))]).to(self.device)
        with torch.no_grad():
            if self.film:
                f = np.zeros(3) if force is None else np.asarray(force, dtype=np.float64).reshape(3) @ self.rotation.T
                feature, _ = self.encoder(batch, torch.as_tensor(f, dtype=torch.float32).reshape(1, 3))
            else:
                feature, _ = self.encoder(batch)
            mu, _ = self.trunk(feature).chunk(2, dim=-1)
            action = torch.tanh(mu)[batch.x[:, GRIPPER] == 1]
        action = action.cpu().numpy().reshape(ACTION_DIM)
        # Back to our frame: translation and rotation are both vectors in their frame.
        inverse = self.rotation.T
        return np.concatenate([action[:3] @ inverse.T, action[3:] @ inverse.T]).astype(np.float32)


def serve(policy: "ReferencePolicy") -> None:
    """Answer action requests on stdin until it closes.

    Our environment runs under the Genesis venv on python 3.13 and their model under the `curl`
    environment on 3.9, so the two cannot share a process. Each request is an 8-byte little-endian
    length followed by that many bytes of a `.npz` holding `pos` and `flags`; each reply is the
    six float32 of the action. Framing is explicit because a pipe gives no message boundaries.
    """
    import io
    import struct
    import sys

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    print("[bridge] ready", file=sys.stderr, flush=True)
    while True:
        header = stdin.read(8)
        if len(header) < 8:
            break
        size = struct.unpack("<Q", header)[0]
        payload = b""
        while len(payload) < size:                       # a pipe may hand back a short read
            chunk = stdin.read(size - len(payload))
            if not chunk:
                return
            payload += chunk
        request = np.load(io.BytesIO(payload))
        action = policy.act(request["pos"], request["flags"], request["force"] if "force" in request.files else None)
        stdout.write(np.asarray(action, dtype=np.float32).tobytes())
        stdout.flush()


def _self_test(checkpoint: str) -> None:
    """Load the checkpoint and drive it with a synthetic arm, garment and gripper."""
    rng = np.random.default_rng(0)
    n_arm, n_cloth = 300, 400
    arm = np.stack([np.linspace(0.0, 0.45, n_arm), rng.normal(0, 0.02, n_arm), rng.normal(0, 0.02, n_arm)], axis=1)
    cloth = arm[rng.integers(0, n_arm, n_cloth)] + rng.normal(0, 0.04, (n_cloth, 3))
    pos = np.concatenate([arm, cloth, np.zeros((1, 3))], axis=0)
    flags = np.zeros((len(pos), 4), dtype=np.float32)
    flags[:n_arm, 1] = 1.0                    # FLAG_MARKER, the arm
    flags[n_arm:n_arm + n_cloth, 0] = 1.0     # FLAG_DEFORMABLE, the garment
    flags[-1, 3] = 1.0                        # FLAG_TOOL

    ref_pos, ref_x = to_reference_cloud(pos, flags)
    print(f"[bridge] {len(pos)} points -> {len(ref_pos)} after a {VOXEL_SIZE * 100:.2f} cm voxel; "
          f"arm {int(ref_x[:, ARM].sum())} cloth {int(ref_x[:, CLOTH].sum())} gripper {int(ref_x[:, GRIPPER].sum())}")

    policy = ReferencePolicy(checkpoint)
    print(f"[bridge] checkpoint at step {policy.step}, best test return {policy.best_return}")
    action = policy.act(pos, flags)
    print(f"[bridge] action in our frame: {np.round(action, 3)}")
    assert action.shape == (ACTION_DIM,) and np.all(np.abs(action) <= 1.0)
    print("[bridge] self-test passed")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="/home/ge47gax/Desktop/vision_based_policy.pt")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--serve", action="store_true", help="Answer action requests on stdin; see serve().")
    p.add_argument("--device", default="cpu")
    p.add_argument("--yaw", type=float, default=267.0, help="Rotation about the vertical, in degrees.")
    p.add_argument("--voxel", type=float, default=VOXEL_SIZE)
    a = p.parse_args(argv)
    if a.self_test:
        _self_test(a.checkpoint)
    elif a.serve:
        serve(ReferencePolicy(a.checkpoint, device=a.device, yaw_deg=a.yaw, voxel=a.voxel))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
