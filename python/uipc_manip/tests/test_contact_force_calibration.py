"""Physical GPU regression: exported contact forces support a known particle load."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cuda
pytest.importorskip("uipc")


@pytest.mark.parametrize("dt", [1.0 / 60.0, 1.0 / 120.0])
@pytest.mark.parametrize("gx", [0.0, 2.0])
@pytest.mark.parametrize("tight", [False, True], ids=["default_tolerance", "tight_tolerance"])
def test_particle_force_calibration(tmp_path, dt, gx, tight):
    # The backend maintains process-global state, so isolate each physical configuration.
    out = tmp_path / "calibration"
    command = [sys.executable, "-m", "uipc_manip.calibrate_contact_force", "--dt", str(dt), "--gx", str(gx), "--out", str(out)]
    if tight:
        command.extend(["--tol", "1e-7"])
    environment = dict(os.environ)
    python_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = python_root + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((out / "result.json").read_text())
    summary = report["summary"]
    assert report["provenance"]["binaries"]
    assert {"PH+N", "PH+F"}.issubset(report["contact_primitive_types"])
    assert np.isfinite(summary["balance_residual_max_n"])
    assert summary["normal_mean_n"][1] == pytest.approx(summary["expected_normal_n"], rel=1e-3)
    if tight:
        assert summary["friction_mean_n"][0] == pytest.approx(summary["expected_friction_n"], abs=1e-6)
        assert summary["balance_residual_max_n"] < 1e-6
    # Default tolerance is measured, not asserted force-accurate: its single-iterate
    # readout can precede a nonzero accepted tangential correction. Keep the raw evidence.
    for row in report["rows"][-summary["tail_frames"]:]:
        assert row["force_balance_residual"] is not None
        assert np.isfinite(row["force_balance_residual"]).all()
