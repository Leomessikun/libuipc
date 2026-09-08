"""CPU tests for the observation layout, the dense PointNet++ encoder, and the policy heads."""

import numpy as np
import pytest

from uipc_manip.obs import EXTRA_DIM, FLAG_GOAL, FLAG_MARKER, FLAG_TOOL, ObsSpec, goal_rel, marker_centroid_rel
from uipc_manip.tasks import TASKS, heuristic_action


def _pack_random(spec: ObsSpec, count: int, rng: np.random.Generator) -> np.ndarray:
    rel = rng.normal(scale=0.05, size=(count, 3))
    marker = np.zeros(count, dtype=bool)
    marker[: max(1, count // 3)] = True
    return spec.pack(rel, marker, rng.normal(size=3), rng.normal(size=3), attached=True)


def test_pack_unpack_roundtrip():
    spec = ObsSpec(16)
    rng = np.random.default_rng(0)
    rel = rng.normal(size=(10, 3))
    marker = np.zeros(10, dtype=bool)
    marker[[1, 4]] = True
    goal = np.array([0.1, -0.2, 0.3])
    tool = np.array([0.5, 0.0, 0.2])
    flat = spec.pack(rel, marker, goal, tool, attached=True)
    assert flat.shape == (spec.dim,)
    pos, feat, valid, extra = spec.unpack_numpy(flat)
    assert valid.sum() == 12
    np.testing.assert_allclose(pos[:10], rel, atol=1e-6)
    assert feat[:10, 0].all()
    np.testing.assert_array_equal(feat[:10, FLAG_MARKER] > 0.5, marker)
    assert feat[10, FLAG_GOAL] == 1.0 and np.allclose(pos[10], goal)
    assert feat[11, FLAG_TOOL] == 1.0 and np.allclose(pos[11], 0.0)
    assert extra.shape == (EXTRA_DIM,)
    np.testing.assert_allclose(extra[:3], tool)
    np.testing.assert_allclose(goal_rel(flat, spec), goal, atol=1e-6)
    np.testing.assert_allclose(marker_centroid_rel(flat, spec), rel[[1, 4]].mean(axis=0), atol=1e-6)


def test_pack_rejects_overflow():
    spec = ObsSpec(4)
    with pytest.raises(ValueError):
        spec.pack(np.zeros((3, 3)), np.zeros(3, dtype=bool), np.zeros(3), np.zeros(3), attached=False)


def test_task_registry_and_heuristic():
    rest = np.stack(np.meshgrid(np.linspace(0, 0.25, 5), np.linspace(0, 0.25, 5), indexing="ij"), axis=-1).reshape(-1, 2)
    rest = np.column_stack([rest, np.full(rest.shape[0], 0.01)])
    rng = np.random.default_rng(0)
    for name, task in TASKS.items():
        if task.deformable != "cloth":
            continue
        grasp = task.grasp_vertices(rest)
        marker = task.marker_vertices(rest)
        assert grasp.size > 0 and marker.size > 0
        goal = task.sample_goal(rng, rest, rest[marker].mean(axis=0))
        assert goal.shape == (3,)
    a = heuristic_action(np.array([0.0, 0.0, 0.0]), np.array([0.01, -0.5, 0.0]), 0.006)
    assert a.shape == (3,) and np.all(np.abs(a) <= 1.0) and a[1] == -1.0


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_encoder_ignores_padding(torch):
    from uipc_manip.models import EncoderConfig, PointNet2Encoder

    torch.manual_seed(0)
    spec_small, spec_big = ObsSpec(12), ObsSpec(40)
    rng = np.random.default_rng(1)
    rel = rng.normal(scale=0.03, size=(8, 3))
    marker = np.zeros(8, dtype=bool)
    marker[:2] = True
    goal, tool = np.array([0.05, 0.0, 0.0]), np.array([0.5, 0.0, 0.1])
    cfg = EncoderConfig(sa_mlp=[[16, 32], [32, 32], [32, 64]], linear_mlp=[32], output_dim=8, sa_neighbors=[8, 8])
    encoder = PointNet2Encoder(4, cfg).eval()
    outputs = []
    for spec in (spec_small, spec_big):
        flat = torch.as_tensor(spec.pack(rel, marker, goal, tool, True))[None]
        pos, feat, valid, _ = spec.unpack_torch(flat)
        with torch.no_grad():
            outputs.append(encoder(pos, feat, valid))
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(outputs[0]).all()


def test_encoder_permutation_invariant(torch):
    from uipc_manip.models import EncoderConfig, PointNet2Encoder

    torch.manual_seed(0)
    cfg = EncoderConfig(sa_mlp=[[16, 32], [32, 32], [32, 64]], linear_mlp=[32], output_dim=8, sa_neighbors=[8, 8])
    encoder = PointNet2Encoder(4, cfg).eval()
    pos = torch.randn(2, 20, 3) * 0.03
    feat = torch.zeros(2, 20, 4)
    feat[:, :, 0] = 1.0
    valid = torch.ones(2, 20, dtype=torch.bool)
    perm = torch.randperm(20)
    with torch.no_grad():
        a = encoder(pos, feat, valid)
        b = encoder(pos[:, perm], feat[:, perm], valid[:, perm])
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_fps_and_actor_critic_shapes(torch):
    from uipc_manip.models import Actor, Critic, EncoderConfig, farthest_point_sample

    torch.manual_seed(0)
    pos = torch.rand(3, 30, 3)
    valid = torch.ones(3, 30, dtype=torch.bool)
    valid[0, 10:] = False
    idx = farthest_point_sample(pos, valid, 6)
    assert idx.shape == (3, 6)
    assert (idx[0] < 10).all()
    spec = ObsSpec(24)
    rng = np.random.default_rng(2)
    flat = torch.as_tensor(np.stack([_pack_random(spec, 10, rng) for _ in range(4)]))
    cfg = EncoderConfig(sa_mlp=[[16, 32], [32, 32], [32, 64]], linear_mlp=[32], output_dim=8, sa_neighbors=[8, 8], sa_ratio=[1.0, 0.5])
    actor = Actor(spec, 3, 32, cfg)
    critic = Critic(spec, 3, 32, cfg)
    batch = spec.unpack_torch(flat)
    mu, pi, log_pi, log_std = actor(batch)
    assert mu.shape == (4, 3) and pi.shape == (4, 3) and log_pi.shape == (4, 1)
    assert torch.all(mu.abs() <= 1.0)
    q1, q2 = critic(batch, pi)
    assert q1.shape == (4, 1) and q2.shape == (4, 1)
    (q1.sum() + q2.sum() + log_pi.sum()).backward()
    assert all(p.grad is not None for p in actor.encoder.parameters())
