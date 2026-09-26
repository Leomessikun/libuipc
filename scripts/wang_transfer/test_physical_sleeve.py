"""Geometry regressions: use the sleeve side, and reject an off-arm tube."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from physical_sleeve import SleeveSections, first_stationary_window, measure
from prepare_flow_dataset import TerminalSleeveAudit


class PhysicalSleeveTests(unittest.TestCase):
    def setUp(self):
        # A sleeve ends at x=0; the mesh continues into a torso at x=3.
        n = 24
        xs = np.linspace(0., 3., 31)
        angles = np.arange(n) * 2 * np.pi / n
        self.vertices = np.array([[x, .07 * np.cos(a), .07 * np.sin(a)] for x in xs for a in angles])
        faces = []
        for j in range(len(xs) - 1):
            for k in range(n):
                a, b = j * n + k, j * n + (k + 1) % n
                faces.extend([[a, b, b + n], [a, b + n, a + n]])
        # Armhole at x=1.1. The closest free boundary must be x=0.
        self.sections = SleeveSections(self.vertices, np.asarray(faces), np.arange(11 * n, 12 * n))
        self.landmarks = np.array([[-.1, 0., 0.], [.55, 0., 0.], [1.2, 0., 0.]])

    def test_sections_follow_sleeve_not_torso(self):
        centers = np.array([p.mean(0) for p in self.sections.points(self.vertices)])
        np.testing.assert_allclose(centers[:, 0], [0., .275, .55, .825], atol=1e-10)
        self.assertTrue(measure(self.sections, self.vertices, self.landmarks)['sleeve_wrapped'])

    def test_off_arm_is_not_threaded(self):
        shifted = self.vertices + [0., .3, 0.]
        self.assertFalse(measure(self.sections, shifted, self.landmarks)['sleeve_wrapped'])

    def test_only_one_section_threaded_is_not_a_complete_sleeve(self):
        displaced = self.vertices.copy()
        displaced[:, 1] += np.clip(displaced[:, 0] * .6, 0., .5)
        result = measure(self.sections, displaced, self.landmarks)
        self.assertTrue(result['rings'][0]['wrapped'])
        self.assertFalse(result['sleeve_wrapped'])

    def test_moving_past_target_is_not_a_stationary_hold(self):
        mask = np.ones(6, dtype=bool)
        moving = np.ones((5, 6)) * .1
        self.assertIsNone(first_stationary_window(mask, moving, 2))
        moving[3:] = 0.
        self.assertEqual(first_stationary_window(mask, moving, 2), 3)
        mask[-1] = False
        self.assertIsNone(first_stationary_window(mask, moving, 2))

    def test_common_audit_recomputes_meshes_and_allows_cuff_beyond_fingers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'episode'
            episode.mkdir()
            path = episode / 'baseline.npz'
            run = dict(success_geometry='physical_sleeve', armhole_endpoint=True,
                       stop_proximal_upper=.7, hold=20, collision_geometry='full_body',
                       environment_sha256='test', collector_sha256='test', sections_wrap=True)
            (root / 'run.json').write_text(json.dumps(run))
            (episode / 'config.json').write_text(json.dumps(dict(
                config=dict(obs=dict(mode='wang_static_arm')), placement={})))
            # The cuff at x=0 lies beyond the fingertip at x=.1. Interior
            # sections still enclose the arm and the armhole is above 0.7.
            landmarks = self.landmarks.copy()
            landmarks[0, 0] = .1
            cloth = np.repeat(self.vertices[None], 21, axis=0)

            def save():
                np.savez(path, positions=cloth, faces=self.sections.faces,
                         opening_idx=self.sections.armhole, finger=landmarks[0],
                         elbow=landmarks[1], shoulder=landmarks[2],
                         sleeve_wrapped=np.ones(21, bool),
                         metadata_json=json.dumps(dict(success_state=0)))

            save()
            audit = TerminalSleeveAudit()
            audit.template = lambda garment, opening: (self.sections, 'test')
            record = dict(source_path=str(path), garment='synthetic')
            result = audit(record, np.zeros((20, 6)))
            self.assertFalse(result['cuff_also_wrapped'])
            self.assertEqual(result['held_states'], 21)
            with self.assertRaisesRegex(ValueError, 'nonzero commands'):
                audit(record, np.ones((20, 6)))
            # Recorded success flags remain true; a displaced final mesh
            # must nevertheless be rejected by the common geometry audit.
            cloth[-1, :, 1] += .3
            save()
            with self.assertRaisesRegex(ValueError, 'Terminal mesh fails'):
                audit(record, np.zeros((20, 6)))


if __name__ == '__main__':
    unittest.main()
