"""Trajectory rendering writes a playable video (CPU; matplotlib and ffmpeg only)."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from uipc_manip.preview import _ffmpeg, render


def _episode(tmp_path, frames=4, verts=6):
    """A tiny two-triangle sheet over four decisions; enough to exercise every npz key."""
    rng = np.random.default_rng(0)
    path = tmp_path / "episode_000.npz"
    np.savez_compressed(
        path,
        positions=rng.normal(scale=0.05, size=(frames, verts, 3)),
        tcp=rng.normal(scale=0.05, size=(frames, 3)),
        goal=rng.normal(scale=0.05, size=(frames, 3)),
        marker_centroid=rng.normal(scale=0.05, size=(frames, 3)),
        qpos=np.zeros((frames, 0)),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        edges=np.array([[0, 1], [1, 2]], dtype=np.int32),
        radius=0.01,
        deformable="cloth",
        task="dressing",
        static_vertices=rng.normal(scale=0.05, size=(3, 3)),
        static_faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    return path


def test_mp4_is_the_default_and_carries_the_still(tmp_path):
    if _ffmpeg() is None:
        pytest.skip("neither a system ffmpeg nor imageio-ffmpeg")
    out = render(_episode(tmp_path), stride=1)
    assert out.suffix == ".mp4" and out.stat().st_size > 0
    assert out.with_suffix(".png").exists()
    # An MP4 begins with an ISO base-media box; a GIF would start with GIF8.
    assert out.read_bytes()[4:8] == b"ftyp"


def test_gif_still_works_for_a_machine_without_ffmpeg(tmp_path):
    out = render(_episode(tmp_path), stride=1, fmt="gif")
    assert out.suffix == ".gif" and out.read_bytes()[:4] == b"GIF8"


def test_an_unknown_format_is_refused_before_any_frame_is_drawn(tmp_path):
    with pytest.raises(ValueError, match="neither"):
        render(_episode(tmp_path), stride=1, fmt="webm")


def test_the_bundled_binary_stands_in_for_a_missing_system_ffmpeg(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    exe = _ffmpeg()
    pytest.importorskip("imageio_ffmpeg")
    assert exe is not None and "ffmpeg" in exe
