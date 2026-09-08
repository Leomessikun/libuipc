"""GPU smoke test: build the Genesis + libuipc environment and run the scripted policy."""

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

genesis = pytest.importorskip("genesis")
torch = pytest.importorskip("torch")

from uipc_manip.genesis_env import EnvConfig, GenesisIPCManipEnv  # noqa: E402
from uipc_manip.obs import goal_rel, marker_centroid_rel  # noqa: E402
from uipc_manip.tasks import heuristic_action  # noqa: E402


def test_cloth_drag_heuristic_progress():
    cfg = EnvConfig(task="cloth_drag", seed=3, point_budget=64, horizon=30, settle_steps=20)
    env = GenesisIPCManipEnv(cfg)
    assert env.spec.deformable_budget == 62 and env._obs_subset.size == 62
    obs = env.reset(seed=0)
    assert obs.shape == (env.spec.dim,) and np.isfinite(obs).all()
    assert env.reset_restore_error < 1e-9
    first = env._prev_distance
    info = {}
    for _ in range(cfg.horizon):
        action = heuristic_action(marker_centroid_rel(obs, env.spec), goal_rel(obs, env.spec), cfg.max_translation)
        obs, reward, done, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
    assert done and info["time_limit"]
    assert info["distance"] < first
    assert info["tracking_error"] < 0.03
    second = env.reset(seed=1)
    assert second.shape == obs.shape and env._episode_step == 0
    env.close()
