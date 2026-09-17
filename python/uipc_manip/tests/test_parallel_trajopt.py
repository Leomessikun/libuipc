from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uipc_manip.parallel_trajopt import (
    IPCBatchEvaluator, TrajectoryCEM, candidate_schedule, expand_knots,
    project_actions, robust_scores,
)


def test_schedule_balances_every_candidate_across_slots_without_padding():
    seen = np.zeros((2, 5, 3), dtype=int)
    for rep, indices in candidate_schedule(5, 3, 2):
        for slot, idx in enumerate(indices):
            assert 0 <= idx < 5
            seen[rep, idx, slot] += 1
    assert np.all(seen == 1)


def test_projection_preserves_reference_and_caps_actual_command_groups():
    ref = torch.tensor([[.8, -.7, .2, .9, .3, -.2], [0., 0., 0., 0., 0., 0.]])
    candidates = torch.randn(30, 2, 6) * 4
    got = project_actions(candidates, ref)
    assert torch.all(got.abs() <= 1)
    assert torch.all(got[..., 3] == 0)
    for part in (slice(0, 3), slice(4, 6)):
        assert torch.all(torch.linalg.vector_norm(got[..., part], dim=-1)
                         <= torch.linalg.vector_norm(ref[..., part], dim=-1) + 1e-6)
    ref[:, 3] = 0
    assert torch.allclose(project_actions(ref, ref), ref)


def test_ranking_rejects_invalid_and_nonfinite_and_penalizes_instability():
    coverage = torch.tensor([[[.4] * 4] * 2, [[.3] * 4, [.5] * 4],
                             [[.8] * 4] * 2, [[float('nan')] * 4] * 2])
    valid = torch.ones_like(coverage, dtype=torch.bool)
    valid[2, 0, 0] = False
    scores = robust_scores(coverage, torch.zeros_like(coverage), valid)
    assert scores[0] > scores[1]
    assert torch.isneginf(scores[2:]).all()


def test_cem_learns_a_temporally_varying_plan_without_larger_motions():
    ref = torch.zeros(12, 6)
    ref[:, 0] = .8
    target = ref.clone()
    target[:6, 1], target[6:, 1] = .5, -.5
    target = project_actions(target, ref)
    cem = TrajectoryCEM(ref, 64, 4, seed=11)
    initial = float(torch.square(ref - target).sum())
    final = initial
    for _ in range(15):
        knots, plans = cem.ask()
        scores = -torch.square(plans - target).sum(dim=(1, 2))
        assert cem.tell(knots, scores)
        final = -float(scores.max())
    assert final < initial * .55
    before = cem.mean.clone()
    assert not cem.tell(knots, torch.full_like(scores, -torch.inf))
    assert torch.equal(before, cem.mean)


def test_evaluator_preserves_slot_identity_and_counts_physical_work(monkeypatch):
    from uipc_manip import physics_gradient_probe as probe

    class Env:
        num_envs = 3
        cfg = SimpleNamespace(horizon=20)
        cells = [SimpleNamespace(shoulder=np.array([1., 0., 0.]), elbow=np.zeros(3), opening_idx=[0]) for _ in range(3)]

        def step(self, actions):
            self.x += actions[:, 0]
            rows = [dict(upperarm_ratio=x, tracking_error=0., grasp_valid=True,
                         collision_rejected_substeps=0, tether_rejected_substeps=0) for x in self.x]
            return np.zeros((3, 1)), self.x.copy(), np.zeros(3, dtype=bool), rows

        def positions(self):
            return [np.array([[x, 0., 0.]]) for x in self.x]

    def restore(env, _):
        env.x = np.arange(3, dtype=float)
        return 0.

    monkeypatch.setattr(probe, 'restore', restore)
    plans = torch.zeros(5, 2, 6)
    plans[:, :, 0] = torch.arange(5)[:, None] / 10
    evaluator = IPCBatchEvaluator(Env(), dict(episode_step=0), torch.device('cpu'))
    _, record = evaluator.evaluate(plans, repeats=2)
    got = np.asarray(record['coverage'])[:, :, -1]
    expected = np.arange(5)[:, None] / 5 + np.tile(np.arange(3), 2)[None]
    assert np.allclose(got, expected)
    assert evaluator.decisions == record['physical_decisions'] == 60
    policy = SimpleNamespace(act=lambda obs, deterministic: np.tile([.4, 0., 0., 0., 0., 0.], (len(obs), 1)))
    evaluator.env.observation = lambda: np.zeros((3, 1))
    _, closed = evaluator.evaluate(plans, tail_agent=policy, tail_steps=1, closed_loop_candidates=(0,))
    got = np.asarray(closed['coverage'])[:, :, -1]
    expected = np.arange(5)[:, None] / 5 + np.arange(3)[None] + .4
    expected[0] = np.arange(3) + 1.2
    assert np.allclose(got, expected)
    assert closed['closed_loop_candidates'] == [0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_projection_and_ranking_agree_with_cpu():
    generator = torch.Generator().manual_seed(9)
    ref = torch.randn((12, 6), generator=generator).clamp(-1, 1)
    knots = torch.randn((8, 4, 6), generator=generator)
    cpu = expand_knots(knots, ref)
    gpu = expand_knots(knots.cuda(), ref.cuda())
    assert gpu.is_cuda
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-6)
    coverage = torch.rand((8, 4, 12), generator=generator)
    valid = torch.ones_like(coverage, dtype=torch.bool)
    assert torch.allclose(robust_scores(coverage, coverage, valid),
                          robust_scores(coverage.cuda(), coverage.cuda(), valid.cuda()).cpu(), atol=1e-6)
