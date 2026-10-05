"""Integrity checks for causal execution and finite-pilot fitting, without IPC."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compile_feedback_repairs import compile_gain_bank
from feedback_repair_control import FeedbackRepair, cloud_features, pilot_settings
from fit_feedback_repairs import fit, fit_router
from run_feedback_repair_pilot import source_controllers


def observation(shift=0.):
    p = np.zeros((12, 7), np.float32)
    p[:6, :3] = np.arange(18).reshape(6, 3) * .001 + shift
    p[6:, :3] = np.arange(18).reshape(6, 3) * .002 + .03
    p[:6, 3] = 1
    p[6:, 4] = 1
    return np.r_[p.ravel(), np.zeros(7)]


class PilotIntegrity(unittest.TestCase):
    def test_camera_features_ignore_privileged_extras(self):
        obs = observation()
        before = cloud_features(obs, np.zeros(3), np.zeros(3))
        obs[-7:] = [999, -45, 85, 234, 1, 0, -92]
        self.assertEqual(before, cloud_features(obs, np.zeros(3), np.zeros(3)))

    def test_camera_tool_frame_compensation(self):
        obs = observation()
        translation = np.array([.1, -.2, .3])
        moved = obs.copy()
        moved[:-7].reshape(-1, 7)[:, :3] -= translation
        np.testing.assert_allclose(cloud_features(obs, np.zeros(3), np.zeros(3)),
                                   cloud_features(moved, translation, np.zeros(3)), atol=3e-8)

    def test_shared_prefix_and_fallback(self):
        settings = pilot_settings()
        a, b = [FeedbackRepair(settings, dict(kind="fixed", prefix=0, suffix=s), 0) for s in (0, 1)]
        nominal = np.array([.2, .4, .1, 0., 0., .3], np.float32)
        for step in range(153):
            aa, _ = a.act(step=step, observation=observation(), tool=np.zeros(3), nominal=nominal)
            bb, _ = b.act(step=step, observation=observation(), tool=np.zeros(3), nominal=nominal)
            if step < 72:
                np.testing.assert_array_equal(aa, bb)
            elif step < 152:
                self.assertGreater(np.linalg.norm(aa - bb), .1)
            else:
                np.testing.assert_array_equal(aa, nominal)
                np.testing.assert_array_equal(bb, nominal)
        self.assertEqual(len(a.record()["response_feature"]), 39)

    def test_observable_router_does_not_read_privileged_state(self):
        settings = dict(pilot_settings(), start=0, prefix_steps=1, suffix_steps=2)
        table = dict(repairs=[dict(prefix=0, assignment=[0, 1])], weights=[1.],
                     routers=[dict(feature=None, left=1, right=1)] * 3)
        outputs = []
        for secret in ([-100.] * 5, [100.] * 5):
            c = FeedbackRepair(settings, dict(kind="observable"), 1, dict(tables=dict(observable=table)))
            for step in (0, 1):
                result, _ = c.act(step=step, observation=observation(), tool=np.zeros(3),
                                  nominal=np.ones(6), privileged=secret)
            outputs.append(result)
        np.testing.assert_array_equal(*outputs)

    def test_lp_prefers_no_harm_gain_over_ceiling_tie(self):
        r = compile_gain_bank([[0., .5], [0., 0.]])
        self.assertGreater(r["mixture_weights"][1], .999)
        self.assertEqual(compile_gain_bank(np.zeros((3, 5)))["mixture_weights"], [1., 0., 0., 0., 0.])

    def test_empty_router_and_no_repair_source_gate(self):
        self.assertEqual(fit_router([], "response_feature")["samples"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "completed").mkdir()
            rows = []
            for repeat in (0, 1):
                for condition in ("nominal", "bending_x2", "density_x1p5"):
                    for body in (1032, 1041):
                        for c in source_controllers():
                            rows.append(dict(phase="source", body=body, condition=condition,
                                             repeat=repeat, accepted=body == 1041,
                                             feedback_repair=dict(controller=c, branch_reached=True,
                                                 initial_feature=[.1] * 18, response_feature=[.1] * 39,
                                                 privileged_feature=[.1] * 5)))
            (out / "completed/000.json").write_text(json.dumps(dict(rows=rows)))
            result = fit(out)
            self.assertFalse(result["source_gate"]["proceed_to_target"])
            self.assertEqual(result["source_gate"]["new_repair_successes"], 0)
            self.assertEqual(result["tables"]["observable"]["weights"][0], 1.)


if __name__ == "__main__":
    unittest.main()
