"""CPU checks of the actor experiment's pieces: the torch observation reproduces the voxel centroids
and spreads a centroid's gradient over its visible members; the last-frame command mapping and the
unit action follow the chain's conventions."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uipc_manip import physics_gradient_actor as actor  # noqa: E402
from uipc_manip.obs import FLAG_DEFORMABLE, FLAG_MARKER, FLAG_TOOL, POINT_DIM, ObsSpec  # noqa: E402


def test_tool_motion_changes_arm_goal_and_extras_and_cancels_held_cloth_motion():
    spec = ObsSpec(8)
    tool0 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    points = np.array([[0.5, 0.4, 0.6], [0.2, 0.3, 0.5]], dtype=np.float32)
    flags = np.zeros((2, 4), dtype=np.float32)
    flags[0, FLAG_MARKER], flags[1, FLAG_DEFORMABLE] = 1, 1
    flat = spec.pack_labeled(points - tool0, flags, np.array([0.8, 0.7, 0.6]), tool0, True)
    cap = dict(flat=flat, visible=np.array([0]), inverse=np.array([0]), counts=np.ones(1),
               deformable_rows=np.array([1]), tool=tool0)
    agent = SimpleNamespace(spec=spec, device=torch.device("cpu"), _cut_padding=False)
    capture = actor.ObservationCapture.__new__(actor.ObservationCapture)
    delta = torch.tensor([0.03, -0.02, 0.01], requires_grad=True)
    pos, feat, valid, extra = capture.torch_observation(cap, torch.tensor(points[1:]) + delta, agent,
                                                      tool=torch.tensor(tool0) + delta)
    # A held point follows the tool, so its relative coordinate must not change.
    assert torch.allclose(pos[0, 1], torch.tensor(points[1] - tool0))
    assert torch.allclose(torch.autograd.grad(pos[0, 1].sum(), delta, retain_graph=True)[0], torch.zeros(3))
    assert torch.allclose(pos[0, 0], torch.tensor(points[0] - tool0) - delta)
    assert torch.allclose(pos[0, 2], torch.tensor([0.8, 0.7, 0.6]) - delta)
    assert torch.allclose(pos[0, 3:], torch.zeros((5, 3)))  # tool and padding stay zero
    assert torch.allclose(extra[0, :3], torch.tensor(tool0) + delta)
    assert torch.allclose(extra[0, 3:6], torch.tensor([0.8, 0.7, 0.6]) - delta)
    assert torch.allclose(torch.autograd.grad(pos[0, 0].sum(), delta)[0], -torch.ones(3))


def test_torch_observation_matches_the_captured_centroids_and_spreads_the_gradient():
    rng = np.random.default_rng(0)
    n, budget = 40, 40
    positions = rng.uniform(0.0, 0.25, (n, 3))
    visible = np.sort(rng.choice(n, 24, replace=False))
    size = 0.0625
    vox = np.floor(positions[visible].astype(np.float32) / size).astype(np.int64) + (1 << 15)
    key = (vox[:, 0] << 32) | (vox[:, 1] << 16) | vox[:, 2]
    uniq, inverse = np.unique(key, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    centroids = np.zeros((uniq.shape[0], 3))
    np.add.at(centroids, inverse, positions[visible])
    centroids /= counts[:, None]
    tool = np.array([0.1, 0.0, 0.2], dtype=np.float32)
    spec = ObsSpec(budget)
    flags = np.zeros((uniq.shape[0], 4), dtype=np.float32)
    flags[:, FLAG_DEFORMABLE] = 1.0
    flat = spec.pack_labeled(centroids - tool, flags, np.array([0.3, 0.0, 0.0]), tool, True)
    cap = {"flat": flat, "visible": visible, "inverse": inverse, "counts": counts, "deformable_rows": np.arange(uniq.shape[0]), "tool": tool}
    agent = SimpleNamespace(spec=spec, device=torch.device("cpu"), _cut_padding=False)
    capture = actor.ObservationCapture.__new__(actor.ObservationCapture)
    x = torch.as_tensor(positions, dtype=torch.float32).requires_grad_(True)
    pos, feat, valid, extra = capture.torch_observation(cap, x, agent)
    assert pos.shape == (1, budget, 3)
    assert np.allclose(pos[0, : uniq.shape[0]].detach().numpy(), centroids - tool, atol=1e-6)
    # A unit pull on every deformable row spreads 1/count over each voxel's visible members, nothing elsewhere.
    pos[0, : uniq.shape[0], 0].sum().backward()
    g = x.grad.numpy()
    expected = np.zeros((n, 3))
    expected[visible, 0] = 1.0 / counts[inverse]
    assert np.allclose(g, expected, atol=1e-6)


def test_last_frame_gradient_follows_the_chain_conventions():
    n, extra = 6, 2
    layout = {"dof_offset": 3 * extra, "dof_count": 3 * n, "n": n, "mass": np.full(n, 0.5), "strength": 10.0,
              "anchor_idx": np.array([1, 4])}
    env = SimpleNamespace(cfg=SimpleNamespace(max_translation=0.01, max_rotation=0.1, clip_rotation_to_yz=True))
    g = np.zeros(3 * n)
    g[3 * 1: 3 * 1 + 3] = [1.0, 0.0, 0.0]

    class Feature:
        def dof_count(self):
            return 3 * (n + extra)

        def solve(self, rhs, tol, rounds):
            return rhs.copy(), 1e-9  # H = I: λ = g

    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])[:2]
    offsets[0] = [0.0, 0.02, 0.0]  # held vertex 1 sits 2 cm along +y from the anchor
    grad, residual, _ = actor.last_frame_gradient(Feature(), layout, g, {"offsets": offsets}, env)
    # Translation: K λ on vertex 1 = 10 × 0.5 × (1, 0, 0), times the action scale.
    assert np.allclose(grad[:3], [5.0 * 0.01, 0.0, 0.0])
    # Rotation: offset × Kλ = (0, 0.02, 0) × (5, 0, 0) = (0, 0, -0.1), times the action scale, x-rotation clipped.
    assert np.allclose(grad[3:], [0.0, 0.0, -0.1 * 0.1])
    assert residual == 1e-9


def test_unit_action_scales_translation_and_rotation_separately():
    env = SimpleNamespace(cfg=SimpleNamespace(max_translation=0.01, max_rotation=0.1, clip_rotation_to_yz=True))
    a = actor.unit_action(np.array([3.0, 0.0, 4.0, 9.0, 0.0, 1.0]), env, 0.005, 0.05)
    assert np.allclose(a[:3], [0.3, 0.0, 0.4])
    assert np.allclose(a[3:], [0.0, 0.0, 0.5])
    assert np.all(np.abs(actor.unit_action(np.ones(6), env, 1.0, 1.0)) <= 1.0)


def test_proposal_queries_preserve_parameter_gradients():
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, obs, **kwargs):
            return self.scale * obs, None, None, None

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(3.0))

        def forward(self, obs, action):
            q = self.scale * (obs + action).sum().reshape(1, 1)
            return q, q + 1.0

    policy, critic = Policy(), Critic()
    agent = SimpleNamespace(actor=policy, critic=critic, device=torch.device("cpu"), _unpack=lambda x: x)
    capture = SimpleNamespace(capture=lambda x: {"visible": np.arange(1), "counts": np.ones(1)},
                              torch_observation=lambda cap, x, agent: x)
    # An existing learning gradient must survive a diagnostic query unchanged.
    policy.scale.grad, critic.scale.grad = torch.tensor(7.0), torch.tensor(8.0)
    result = actor.value_and_gradient(agent, capture, np.ones((1, 3)))
    assert np.allclose(result["gradient"], 9.0)  # d[3(x + 2x)] / dx.
    result = actor.sac_action_gradient(agent, np.ones(3))
    assert np.allclose(result["dQ_da"], 3.0)
    assert policy.scale.grad.item() == 7.0 and critic.scale.grad.item() == 8.0
