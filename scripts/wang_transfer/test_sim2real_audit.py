"""Measurement integrity: sensor boundaries, voxel equivalence, paired statistics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sim2real_audit import CONDITIONS, POLICIES, ROOT, exact_mcnemar, summarize
from sim2real_perturbations import perturb_observation

sys.path.insert(0, str(ROOT / ".claude/worktrees/residual-rl/python"))
from uipc_manip.obs import ObsSpec
from uipc_manip.wang_bridge import to_reference_cloud


class MeasurementIntegrity(unittest.TestCase):
    def setUp(self):
        self.spec = ObsSpec(128)
        rng = np.random.default_rng(10)
        points = rng.normal(0., .1, (100, 3)).astype(np.float32)
        flags = np.zeros((100, 4), np.float32)
        flags[:50, 0], flags[50:, 1] = 1, 1
        self.obs = self.spec.pack_labeled(points, flags, np.array([.2, 0., 0.]), np.array([1., 2., 3.]), True)

    def test_sensor_changes_preserve_truth_proprioception_tool_and_goal(self):
        clean = self.obs.copy()
        noisy = perturb_observation(clean, self.spec, noise_m=.003, dropout=.3,
                                    body=1032, seed=52, frame=4)
        np.testing.assert_array_equal(self.obs, clean)
        np.testing.assert_array_equal(clean[-7:], noisy[-7:])
        pos, feat, valid, _ = self.spec.unpack_numpy(noisy)
        clean_pos, clean_feat, _, _ = self.spec.unpack_numpy(clean)
        protected = (clean_feat[:, 2] > .5) | (clean_feat[:, 3] > .5)
        np.testing.assert_array_equal(pos[protected], clean_pos[protected])
        np.testing.assert_array_equal(feat[protected], clean_feat[protected])
        self.assertLess(valid.sum(), 102)
        self.assertGreater(valid.sum(), 2)
        np.testing.assert_array_equal(noisy, perturb_observation(clean, self.spec, noise_m=.003,
                                                               dropout=.3, body=1032, seed=52, frame=4))
        self.assertFalse(np.array_equal(noisy, perturb_observation(clean, self.spec, noise_m=.003,
                                                                  dropout=.3, body=1032, seed=52, frame=5)))

    def test_shift_zero_exactly_matches_bridge_and_shift_never_translates(self):
        pos, feat, valid, _ = self.spec.unpack_numpy(self.obs)
        expected = to_reference_cloud(pos[valid], feat[valid], yaw_deg=267., voxel=.0625)
        zero = perturb_observation(self.obs, self.spec, voxel_shift=[0, 0, 0])
        p, f, v, _ = self.spec.unpack_numpy(zero)
        actual = to_reference_cloud(p[v], f[v], yaw_deg=267., voxel=0.)
        for x, y in zip(expected, actual):
            np.testing.assert_array_equal(x, y)
        shifted = perturb_observation(self.obs, self.spec, voxel_shift=[.03125] * 3)
        p, f, v, _ = self.spec.unpack_numpy(shifted)
        np.testing.assert_array_equal(pos, p)
        self.assertFalse(np.array_equal(f, self.spec.unpack_numpy(zero)[1]))
        self.assertEqual(float(f[:, 3].sum()), 1.)

    def test_pairing_counts_grasp_failure_and_detects_interaction(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "completed").mkdir()
            rows = []
            for c in ("nominal", "friction_x2", "nominal_repeat"):
                for p in POLICIES:
                    for b in (1032, 1041):
                        ok = (p != "fmvp_sim") if c != "friction_x2" else (p == "fmvp_sim")
                        rows.append(dict(condition=c, policy=p, body=b, accepted=ok,
                                         initial_geometry_sha256="same", sim_error=None if ok else "grasp tracking exceeded validity limit"))
            (out / "completed/000.json").write_text(json.dumps(dict(rows=rows)))
            report = summarize(out, dict(planned_episodes=18, measured_bodies=[1032, 1041], jobs=[0]))
            comp = report["conditions"]["friction_x2"]["comparisons"]["r1"]
            self.assertEqual(report["conditions"]["friction_x2"]["paired_units"], 2)
            self.assertEqual(comp["gain"]["mean"], -1.)
            self.assertEqual(comp["gain_change_from_nominal"]["mean"], -2.)
            self.assertEqual(report["repeatability"]["r1"]["agreement"]["mean"], 1.)
            self.assertEqual(exact_mcnemar(6, 0), .03125)


if __name__ == "__main__":
    unittest.main()
