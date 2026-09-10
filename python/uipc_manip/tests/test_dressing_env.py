"""GPU smoke test: build the batched dressing environment from the Newton cache and step it."""

from pathlib import Path

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


def _require_live_assets():
    """Skip unless the raw garments, the SMPL-X model and Newton's offline drapes are on disk."""
    from uipc_manip.dressing_body import BodyConfig
    from uipc_manip.dressing_live import OFFLINE_DRAPE_DIR, available_garments

    if not Path(BodyConfig().model_dir).exists():
        pytest.skip("SMPL-X model files are not available on this machine")
    if not {"tshirt_26", "tshirt_4"} <= set(available_garments()):
        pytest.skip("raw garment meshes are not available on this machine")
    if not Path(OFFLINE_DRAPE_DIR).exists():
        pytest.skip("Newton's offline drapes, which tshirt_4 falls back to, are not available")


def test_live_multicell_world_settles_and_steps(tmp_path):
    """Two live cells on two generated bodies, settled for the pre-flight's ten steps and then
    driven by the scripted expert. The copies' garments occupy the same volume, so this holds
    only while each copy's garment and arm share an IPC subscene the other copy cannot reach."""
    _require_live_assets()
    cells = (("tshirt_26", 0), ("tshirt_4", 1))
    cfg = DressingConfig(
        cell_source="live", cells=cells, horizon=8, action_repeat=6, settle_steps=10,
        seed=1, point_budget=256, augment_obs=False, workspace=str(tmp_path),
    )
    env = GenesisIPCDressingEnv(cfg, num_envs=len(cells))
    assert env.cell_labels == ["tshirt_26__human_0", "tshirt_4__human_1"]
    assert 0.0 < env.settle_displacement < 1.0
    clearances = env.clearances()
    assert len(clearances) == 2 and all(c["cloth_to_arm_m"] > cfg.cloth_thickness for c in clearances)
    assert np.isfinite(env.reset()).all()
    worst = 0.0
    for _ in range(6):
        obs, rewards, _, infos = env.step(env.scripted_actions())
        assert not any(info.get("sim_error") for info in infos), infos[0].get("error")
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        worst = max(worst, max(info["tracking_error"] for info in infos))
    assert worst < env.grasp_tracking_tolerance_m, f"cuff lagged the tool by {worst:.3f} m"
    env.close()
