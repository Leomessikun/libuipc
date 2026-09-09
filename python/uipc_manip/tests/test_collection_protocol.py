"""Collection must construct the teacher's task before playing its actions."""

import pytest

from uipc_manip import collect_rollouts
from uipc_manip.sac import SACAgent, SACConfig


def test_collector_inherits_teacher_environment_before_build(monkeypatch, tmp_path):
    payload = {
        "sac_config": SACConfig().to_dict(),
        "metadata": {
            "task": "dressing",
            "env": {"human": 6, "garments": ["tshirt_68"], "horizon": 150, "action_repeat": 6,
                    "point_budget": 256, "constraint_strength": 1e6},
        },
    }
    monkeypatch.setattr(SACAgent, "read_checkpoint", lambda path: payload)

    class BuiltExpectedEnvironment(Exception):
        pass

    def make_env(args):
        assert args.human == 6 and args.garments == ["tshirt_68"]
        assert args.horizon == 150 and args.action_repeat == 6
        assert args.cuff_strength == 1e6 and args.point_budget == 256
        raise BuiltExpectedEnvironment

    monkeypatch.setattr(collect_rollouts, "make_env", make_env)
    with pytest.raises(BuiltExpectedEnvironment):
        collect_rollouts.main(["--checkpoint", "teacher.pt", "--out-dir", str(tmp_path)])


def test_collector_rejects_another_task_before_build(monkeypatch, tmp_path):
    payload = {"sac_config": SACConfig().to_dict(), "metadata": {"task": "dressing"}}
    monkeypatch.setattr(SACAgent, "read_checkpoint", lambda path: payload)
    with pytest.raises(ValueError, match="Checkpoint task"):
        collect_rollouts.main(["--checkpoint", "teacher.pt", "--task", "cloth_drag", "--out-dir", str(tmp_path)])
