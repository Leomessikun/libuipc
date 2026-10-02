"""Focused causal-input, matched-candidate and history-boundary checks (CPU)."""
import sys
import io
import struct
import subprocess
import tempfile
import json
from unittest.mock import patch
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_observation import ObservedArmMotion, history_inputs


class CausalInputsTest(unittest.TestCase):
    def test_robot_motion_is_removed_before_registration(self):
        rng = np.random.default_rng(3)
        world = rng.normal(size=(30, 3)) * .1
        flags = np.zeros((30, 4)); flags[:, 1] = 1
        def register(source, target):
            transform = np.eye(4)
            transform[:3, 3] = target.mean(0) - source.mean(0)
            return transform, dict(valid=True, rms_m=0.)
        motion = ObservedArmMotion(register, lambda p, t: p, dt=.1)
        tool0, tool1 = np.zeros(3), np.array([.08, -.1, .02])
        motion.update(world - tool0, flags, np.ones(30, bool), tool0)
        motion.update(world - tool1, flags, np.ones(30, bool), tool1)
        np.testing.assert_allclose(motion.displacement(world), 0., atol=1e-12)
        shift = np.array([.01, .002, -.005])
        motion.update(world + shift - tool1, flags, np.ones(30, bool), tool1)
        np.testing.assert_allclose(motion.displacement(world), np.broadcast_to(shift, world.shape), atol=1e-12)

    def test_missing_or_invalid_cloud_cannot_keep_stale_velocity(self):
        def register(source, target):
            t = np.eye(4); t[0, 3] = .01
            return t, dict(valid=len(target) >= 12, rms_m=0.)
        motion = ObservedArmMotion(register, lambda p, t: p, dt=.1)
        points = np.zeros((20, 3)); flags = np.zeros((20, 4)); flags[:, 1] = 1
        motion.update(points, flags, np.ones(20, bool), np.zeros(3))
        motion.update(points, flags, np.ones(20, bool), np.zeros(3))
        self.assertGreater(float(motion.displacement(points)[0, 0]), 0)
        motion.update(points, flags, np.zeros(20, bool), np.zeros(3))
        np.testing.assert_array_equal(motion.transform, np.eye(4))

    def test_history_has_no_future_or_other_episode_frames(self):
        f = np.arange(20, dtype=np.float32).reshape(5, 4)
        tool = np.arange(15, dtype=np.float32).reshape(5, 3)
        x = history_inputs(f, tool, 3)
        altered = f.copy(); altered[3:] = -999
        np.testing.assert_array_equal(x[:3], history_inputs(altered, tool, 3)[:3])
        first = x[0].reshape(3, 7)
        np.testing.assert_array_equal(first[:, :4], np.repeat(f[:1], 3, axis=0))
        np.testing.assert_array_equal(first[:, 4:], 0)
        fresh = history_inputs(f[-1:], tool[-1:], 3)
        np.testing.assert_array_equal(fresh[0].reshape(3, 7)[:, :4], np.repeat(f[-1:], 3, axis=0))

    def test_no_history_control_has_same_shape_and_only_current_data(self):
        f = np.arange(20, dtype=np.float32).reshape(5, 4)
        tool = np.arange(15, dtype=np.float32).reshape(5, 3)
        x = history_inputs(f, tool, 3, current_only=True)
        self.assertEqual(x.shape, history_inputs(f, tool, 3).shape)
        np.testing.assert_array_equal(x.reshape(5, 3, 7)[:, :, :4], np.repeat(f[:, None], 3, axis=1))
        np.testing.assert_array_equal(x.reshape(5, 3, 7)[:, :, 4:], 0)

    def test_candidate_actions_do_not_depend_on_planning_belief(self):
        from motion_lookahead_probe import MotionPlanner
        p = MotionPlanner.__new__(MotionPlanner)
        p.env = SimpleNamespace(cfg=SimpleNamespace(max_translation=.02), _anchor=np.zeros((1, 3)))
        p.candidate_set = "observed_common"
        p.observed = SimpleNamespace(displacement=lambda x: np.array([.01, 0, 0]),
                                     diagnostics={}, transform=np.eye(4))
        p.finger_shift = lambda mode: (_ for _ in ()).throw(AssertionError("Privileged candidate shift accessed"))
        p.snapshot = lambda: None
        p.restore = lambda x: None
        p.replay_checked, p.noise_margin = True, .0002
        p.evaluate = lambda action, nominal, mode: (dict(score=float(action[0])), None)
        nominal = np.linspace(-.3, .3, 6)
        actions = []
        for mode in ("current", "causal", "true", "observed"):
            _, diag = p.improve(nominal, mode)
            actions.append([r["action"] for r in diag["candidates"]])
        for value in actions[1:]:
            np.testing.assert_array_equal(value, actions[0])

    def test_student_banner_cannot_corrupt_binary_actions(self):
        # The reference FiLM module prints on load. Test the actual binary
        # server with an intentionally noisy policy in unbuffered Python.
        source = """
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import numpy as np
import dynamic_student as module
class NoisyPolicy:
    def __init__(self, *args): print('model banner')
    def reset(self): pass
    def act(self, *args):
        print('inference banner')
        return np.arange(6, dtype=np.float32) / 10
module.StudentPolicy = NoisyPolicy
module.serve(SimpleNamespace(checkpoint=None, package_root=None, yaw=267))
"""
        payload = io.BytesIO()
        np.savez(payload, pos=np.zeros((1, 3)), flags=np.zeros((1, 4)), tool=np.zeros(3))
        data = payload.getvalue()
        result = subprocess.run([sys.executable, "-u", "-c", source, str(Path(__file__).resolve().parent)],
                                input=struct.pack("<Q", len(data)) + data, capture_output=True, check=True)
        self.assertEqual(len(result.stdout), 24)
        np.testing.assert_allclose(np.frombuffer(result.stdout, np.float32), np.arange(6) / 10)
        self.assertIn(b"model banner", result.stderr)
        self.assertIn(b"inference banner", result.stderr)

    def test_candidate_error_does_not_reset_real_episode(self):
        from motion_lookahead_probe import make_env_class
        from uipc_manip.dressing_motion import MotionDressingEnv
        cls = make_env_class()
        env = cls.__new__(cls)
        env.plan_mode = "observed"
        env._obs_placeholder = np.zeros((1, 12))
        with patch.object(MotionDressingEnv, "reset", side_effect=AssertionError("Real reset inside branch")):
            result = env.reset()
        np.testing.assert_array_equal(result, env._obs_placeholder)
        self.assertIsNot(result, env._obs_placeholder)
        env.plan_mode = None
        with patch.object(MotionDressingEnv, "reset", return_value="real reset") as reset:
            self.assertEqual(env.reset([7]), "real reset")
            reset.assert_called_once_with([7])

    def run_mock_pipeline(self, root, observed_success):
        from run_dynamic_pipeline import Pipeline, save
        pipeline = Pipeline(root, 2)
        (root / "motions").mkdir()
        calls = []
        def run_jobs(jobs, gpu=True):
            for job in jobs:
                calls.append(job)
                output = Path(job["complete"])
                output.parent.mkdir(parents=True, exist_ok=True)
                if gpu:
                    cmd = [str(x) for x in job["command"]]
                    condition = cmd[cmd.index("--conditions") + 1]
                    success = condition == "observed" and observed_success
                    save(output, [dict(condition=condition, final_success=success, failure=None)])
                    np.savez(output.parent / f"{condition}_initial.npz",
                             positions=np.zeros((3, 3)), human_vertices=np.zeros((3, 3)), tcp=np.zeros(3))
                elif "_train" in pipeline.status["stage"]:
                    save(output, dict(feature_manifest_sha256="same", base_sha256="same", updates=1000,
                                      batch=128, parameters=87942, validation_bodies=[14052]))
                else:
                    output.write_text("{}")
        pipeline.run_jobs = run_jobs
        pipeline.run()
        return pipeline, calls

    def test_teacher_gate_stops_scaling_on_failed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline, calls = self.run_mock_pipeline(Path(directory), False)
        self.assertEqual(pipeline.status["stage"], "teacher_gate_failed")
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(job["name"].startswith("validate_") for job in calls))

    def test_pipeline_aggregates_both_rollins_and_preserves_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline, calls = self.run_mock_pipeline(Path(directory), True)
        self.assertEqual(pipeline.status["stage"], "complete")
        feature = next(job for job in calls if job["name"] == "round1_features")
        paths = [str(x) for x in feature["command"]]
        self.assertEqual(sum("/dagger_" in x for x in paths), 4)
        self.assertFalse(any("/test_" in x for x in paths))
        dagger = [job for job in calls if job["name"].startswith("dagger_")]
        self.assertEqual(len(dagger), 4)
        for job in dagger:
            cmd = job["command"]
            self.assertEqual(cmd[cmd.index("--teacher-execution-probability") + 1], "0.0")
        train = [job for job in calls if "_seed" in job["name"] and job["name"].startswith("round")]
        self.assertEqual(len(train), 12)


if __name__ == "__main__":
    unittest.main()
