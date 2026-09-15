import pytest
import torch

from uipc_manip.counterfactual import response_objective, symmetric_radius
from uipc_manip.response_experiment import ResponseModel, split


def test_feasible_symmetric_radius_preserves_direction_at_bounds():
    a = torch.tensor([[.99, 0., 0.], [1., 0., 0.]])
    v = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    radius = symmetric_radius(a, v, .1)
    torch.testing.assert_close(radius, torch.tensor([.01, 0.]))
    assert ((a + radius[:, None] * v).abs() <= 1).all()
    with pytest.raises(ValueError):
        symmetric_radius(a, v, 0)


@pytest.mark.parametrize('mode', ['response', 'jacobian', 'counterfactual', 'difference'])
def test_linear_teacher_zero_loss_and_no_teacher_grad(mode):
    torch.manual_seed(5)
    matrix = torch.randn(2, 4, 3, 3)
    a = torch.rand(2, 3) - .5
    predict = lambda x: torch.einsum('bnca,ba->bnc', matrix, x)
    teacher = predict(a).requires_grad_()
    d = matrix.clone().requires_grad_()
    v = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    loss, _ = response_objective(predict, a, teacher, d, v, mode=mode)
    assert loss.item() < 1e-10
    assert teacher.grad is None and d.grad is None


def test_difference_keeps_slope_strength_while_endpoint_loss_shrinks():
    weight = torch.tensor(2., requires_grad=True)
    a = torch.zeros(1, 3)
    v = torch.tensor([[1., 0., 0.]])
    predict = lambda x: (weight * x[:, :1])[:, :, None]
    response = torch.zeros(1, 1, 1)
    d = torch.ones(1, 1, 1, 3)
    losses = {}
    for mode in ('counterfactual', 'difference', 'jacobian'):
        losses[mode] = [response_objective(predict, a, response, d, v, mode=mode, epsilon=e)[0]
                        for e in (.1, .01)]
    torch.testing.assert_close(losses['counterfactual'][0], 100 * losses['counterfactual'][1])
    torch.testing.assert_close(losses['difference'][0], losses['difference'][1])
    torch.testing.assert_close(losses['difference'][0], losses['jacobian'][0])


@pytest.mark.parametrize('mode', ['jacobian', 'counterfactual', 'difference'])
def test_differential_loss_reaches_encoder_and_history(mode):
    torch.manual_seed(7)
    model = ResponseModel()
    points = torch.randn(2, 3, 16, 3) * .03
    past = torch.randn(2, 3, 3)
    h = model.encode(points, past)
    a = torch.zeros(2, 3)
    q = torch.randn(2, 5, 3)
    predict = lambda x: model.respond(h, x, q)
    # Perfect nominal target ensures gradients come only from differential supervision.
    y = predict(a).detach()
    d = torch.randn(2, 5, 3, 3)
    v = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    loss, _ = response_objective(predict, a, y, d, v, mode=mode)
    loss.backward()
    for module in (model.encoder, model.history):
        total = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
        assert total > 0


def test_split_keeps_all_slots_and_rows_of_episode_together():
    import numpy as np
    data = {'episode': np.repeat(np.arange(8), 20)}
    train, val = split(data, 1)
    assert not set(data['episode'][train]) & set(data['episode'][val])
    assert len(train) + len(val) == 160


def test_offline_experiment_uses_ood_only_for_evaluation(tmp_path):
    import argparse
    import json
    import numpy as np
    from uipc_manip.response_experiment import run

    rng = np.random.default_rng(4)
    query = rng.normal(size=(16, 3)).astype('float32') * .03
    for name in ('train', 'ood'):
        root = tmp_path / name
        root.mkdir()
        n = 12
        action = rng.uniform(-.5, .5, (n, 3)).astype('float32')
        direction = np.ones((n, 3), dtype='float32') / np.sqrt(3)
        tangent = np.broadcast_to(np.eye(3), (n, 16, 3, 3)).copy().astype('float32') * .001
        response = np.einsum('bnca,ba->bnc', tangent, action)
        dv = np.einsum('bnca,ba->bnc', tangent, direction)
        np.savez(root / 'data.npz', query=query, visible_ids=np.arange(8),
                 points=rng.normal(size=(n, 2, 8, 3)).astype('float32')*.03,
                 past_actions=np.zeros((n, 2, 3)), action=action, tangent=tangent,
                 response=response, episode=np.repeat(np.arange(4), 3),
                 velocity=response.mean(1), goal=np.zeros((n, 3)),
                 returns=rng.normal(size=n), reward=np.zeros(n),
                 remaining_time=np.ones((n, 1)), probe_valid=np.ones(n),
                 probe_direction=direction, probe_plus=response+.05*dv,
                 probe_minus=response-.05*dv)
        (root / 'dataset.json').write_text(json.dumps({'probe_epsilon': .05}))
    args = argparse.Namespace(data=str(tmp_path/'train'), ood_data=str(tmp_path/'ood'),
        out=str(tmp_path/'result'), device='cpu', seeds=[0], variants=['jacobian', 'difference'],
        steps=2, probe_steps=2, batch_size=2, lr=.001, weight=1., epsilon=.05,
        target_sampling='dense', target_points=8, split_seed=5, probe_seed=6)
    run(args)
    protocol = json.loads((tmp_path/'result'/'protocol.json').read_text())
    assert max(protocol['train_episodes']) < min(protocol['test_episodes'])
    assert protocol['ood_dataset_sha256']
    results = json.loads((tmp_path/'result'/'results.json').read_text())
    assert len(results) == 2
    assert all(np.isfinite(r['response_q_mse']) for r in results)
    with pytest.raises(FileExistsError):
        run(args)
