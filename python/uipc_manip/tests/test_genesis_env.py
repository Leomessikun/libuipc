"""GPU smoke test: build the batched Genesis + libuipc environment and run the scripted policy."""

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

genesis = pytest.importorskip("genesis")
torch = pytest.importorskip("torch")

from uipc_manip.genesis_env import EnvConfig, GenesisIPCManipEnv  # noqa: E402
from uipc_manip.obs import goal_rel, marker_centroid_rel  # noqa: E402
from uipc_manip.tasks import heuristic_action  # noqa: E402


def test_batched_cloth_drag_heuristic_progress():
    cfg = EnvConfig(task="cloth_drag", seed=3, point_budget=64, horizon=30, settle_steps=20)
    env = GenesisIPCManipEnv(cfg, num_envs=2)
    assert env.spec.deformable_budget == 62 and env._obs_subset.size == 62
    assert len(env.slots) == 2 and env.positions().shape == (2, env.deformable.vertex_count, 3)
    obs = env.reset([0, 1])
    assert obs.shape == (2, env.spec.dim) and np.isfinite(obs).all()
    assert env.reset_restore_error < 1e-9
    # Different seeds must give different goals; the physics start is shared.
    assert not np.allclose(env.goals[0], env.goals[1])
    first = env._prev_distance.copy()
    infos = []
    for _ in range(cfg.horizon):
        actions = np.stack(
            [heuristic_action(marker_centroid_rel(o, env.spec), goal_rel(o, env.spec), cfg.max_translation) for o in obs]
        )
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (2, env.spec.dim) and np.isfinite(obs).all() and np.isfinite(rewards).all()
    assert dones.all() and all(info["time_limit"] for info in infos)
    assert all("terminal_obs" in info for info in infos)
    assert all(info["distance"] < first[i] for i, info in enumerate(infos))
    assert all(info["tracking_error"] < 0.03 for info in infos)
    assert env._episode_step == 0
    env.close()
