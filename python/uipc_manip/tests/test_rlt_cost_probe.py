"""Cost accounting covers full optimizer schedules and counts work, not input slots."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from uipc_manip.rlt_cost_probe import count_spatial_clouds, measure_updates  # noqa: E402


class ScheduledAgent:
    def __init__(self):
        self.cfg = SimpleNamespace(actor_update_freq=4, critic_target_update_freq=2, batch_size=64)
        self.actor, self.critic, self.critic_target = [
            SimpleNamespace(encoder=torch.nn.Linear(2, 2)) for _ in range(3)
        ]
        self.updates = 0

    def update(self, replay):
        self.updates += 1
        self.actor.encoder(torch.zeros(5, 2))
        self.critic.encoder(torch.zeros(7, 2))
        if self.updates % 2 == 0:
            self.critic_target.encoder(torch.zeros(11, 2))
        stats = {"learning_positions": 3 + self.updates % 2}
        if self.updates % 4 == 0:
            self.actor.encoder(torch.zeros(5, 2))
            stats["actor_loss"] = 0.0
        return stats


def test_update_measurement_covers_actor_cadence_and_actual_positions():
    agent = ScheduledAgent()
    result = measure_updates(agent, None, torch.device("cpu"), cycles=2)
    assert agent.updates == 12
    assert result["warmup_updates"] == 4
    assert result["measured_updates"] == 8
    assert result["actor_updates"] == 2
    assert result["learning_positions_total"] == 28
    assert result["learning_positions_per_update"] == 3.5
    assert result["spatial_clouds_by_encoder"] == {"actor": 50, "critic": 56, "target": 44}
    assert result["spatial_clouds_total"] == 150
    assert result["per_position_s"] == pytest.approx(result["update_s"] * 8 / 28)
    assert result["cuda_peak_allocated_bytes"] is None
    assert not agent.actor.encoder._forward_pre_hooks


def test_counter_cleans_up_hooks_when_measurement_fails():
    agent = ScheduledAgent()
    with pytest.raises(RuntimeError, match="interrupted"):
        with count_spatial_clouds(agent):
            raise RuntimeError("interrupted")
    for module in (agent.actor, agent.critic, agent.critic_target):
        assert not module.encoder._forward_pre_hooks


@pytest.mark.parametrize("cycles,warmup", [(0, 1), (1, 0)])
def test_measurement_refuses_empty_cycles(cycles, warmup):
    with pytest.raises(ValueError, match="complete cycle"):
        measure_updates(ScheduledAgent(), None, torch.device("cpu"), cycles=cycles, warmup_cycles=warmup)
