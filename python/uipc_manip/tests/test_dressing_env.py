"""GPU smoke test: build the batched dressing environment from the Newton cache and step it."""

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

genesis = pytest.importorskip("genesis")
torch = pytest.importorskip("torch")

from uipc_manip.dressing_assets import DressingCache, DressingCacheConfig  # noqa: E402
from uipc_manip.dressing_env import DressingConfig, GenesisIPCDressingEnv  # noqa: E402


def test_batched_dressing_env_steps():
    cache_cfg = DressingCacheConfig()
    if not cache_cfg.cache_path.exists():
        pytest.skip("Newton dressing cache is not available on this machine")
    assert ("tshirt_26", 0) in DressingCache(cache_cfg).cells
    cfg = DressingConfig(human=0, garments=("tshirt_26",), horizon=4, settle_steps=5, seed=2, point_budget=256)
    env = GenesisIPCDressingEnv(cfg, num_envs=2)
    assert env.action_dim == 6 and len(env.slots) == 2
    obs = env.reset([0, 1])
    assert obs.shape == (2, env.spec.dim) and np.isfinite(obs).all()
    assert env.reset_restore_error < 1e-9
    # The opening starts in front of the fingertips: pre-insertion reward, no progress yet.
    actions = env.scripted_actions()
    assert actions.shape == (2, 6) and np.all(np.abs(actions) <= 1.0)
    infos = []
    for _ in range(cfg.horizon):
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (2, env.spec.dim) and np.isfinite(obs).all() and np.isfinite(rewards).all()
    assert dones.all() and all(info["time_limit"] for info in infos)
    assert all(0.0 <= info["upperarm_ratio"] <= 1.0 for info in infos)
    assert all(info["garment"] == "tshirt_26" for info in infos)
    assert all("terminal_obs" in info for info in infos)
    env.close()


def test_cuff_hold_tracks_the_commanded_anchor():
    """The held cuff must follow the tool through free air: a hold that lags by centimetres detaches
    the garment from the policy's actions once the sleeve touches the hand."""
    cache_cfg = DressingCacheConfig()
    if not cache_cfg.cache_path.exists():
        pytest.skip("Newton dressing cache is not available on this machine")
    cfg = DressingConfig(human=0, garments=("tshirt_26",), horizon=60, settle_steps=10, seed=3, point_budget=256)
    env = GenesisIPCDressingEnv(cfg, num_envs=1)
    env.reset([0])
    assert env.snapshot_tracking_error < 2.0e-3
    # Raise the anchor at the reference's top speed, away from the arm, for 40 decisions.
    lift = np.zeros((1, 6))
    lift[0, 2] = 1.0
    worst = 0.0
    for _ in range(40):
        _, _, _, infos = env.step(lift)
        worst = max(worst, infos[0]["tracking_error"])
    assert worst < 2.0e-2, f"cuff lagged the tool by {worst:.3f} m"
    env.close()
