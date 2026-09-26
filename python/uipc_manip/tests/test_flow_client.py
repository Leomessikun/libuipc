"""Deployment history boundaries, command contract and real subprocess framing."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uipc_manip.flow_client import FlowController, FlowPolicyClient
from uipc_manip.fql import FQLConfig
from uipc_manip.obs import ObsSpec
from uipc_manip.tests.test_sac_agent import _small_cfg
from uipc_manip.train_flow_bc import ChunkFlowPolicy


CONTRACT = dict(point_budget=10, obs_mode="wang_static_arm", collision_geometry="full_body",
                max_translation_m=.008, max_rotation_rad=.08, decision_dt_s=.1)


class RecordingModel:
    spec = SimpleNamespace(dim=3)
    history_length, horizon = 3, 2

    def sample(self, observations, valid, noise):
        self.last = observations.clone(), valid.clone(), noise.clone()
        return noise.reshape(1, 2, 6).clamp(-1, 1)


def test_history_is_causal_separate_per_slot_and_resettable():
    model = RecordingModel()
    controller = FlowController(model, dict(contract=CONTRACT))
    controller.reset(17, CONTRACT)
    first = controller.act(np.ones(3), 0)
    torch.testing.assert_close(model.last[1], torch.tensor([[False, False, True]]))
    controller.act(np.full(3, 2), 0)
    torch.testing.assert_close(model.last[0][0, :, 0], torch.tensor([1., 1., 2.]))
    controller.act(np.full(3, 9), 1)
    torch.testing.assert_close(model.last[1], torch.tensor([[False, False, True]]))
    assert (model.last[0] == 9).all()
    controller.reset(17, CONTRACT)
    np.testing.assert_array_equal(controller.act(np.ones(3), 0), first)
    with pytest.raises(ValueError, match="decision_dt_s"):
        controller.reset(17, dict(CONTRACT, decision_dt_s=.2))


def test_checkpoint_subprocess_reproduces_direct_controller(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(18)
    spec = ObsSpec(10)
    model = ChunkFlowPolicy(spec, FQLConfig(encoder=_small_cfg().encoder, hidden_dim=16, flow_steps=2), 3, 2).eval()
    path = tmp_path / "policy.pt"
    metadata = dict(contract=CONTRACT)
    model.save(path, metadata=metadata, step=0)
    observation = spec.pack(np.array([[.02, 0, 0], [.04, .01, 0]]), np.array([False, True]),
                            np.array([.1, 0, 0]), np.zeros(3), True)
    direct = FlowController(model, metadata)
    direct.reset(5, CONTRACT)
    client = FlowPolicyClient(path)
    try:
        client.reset(5, CONTRACT)
        for slot in (0, 1, 0):
            np.testing.assert_allclose(client.act(observation, slot), direct.act(observation, slot), rtol=0, atol=0)
        client.reset(5, CONTRACT)
        direct.reset(5, CONTRACT)
        np.testing.assert_allclose(client.act(observation, 0), direct.act(observation, 0), rtol=0, atol=0)
    finally:
        client.close()
    assert client.proc.returncode == 0
