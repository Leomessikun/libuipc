# Local cloth and cable environment

The isolated `.venv` uses Python 3.11 and the released `pyuipc==0.0.28`
CUDA wheel, rather than a native build of this checkout.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r python/examples/cloth_cable_requirements.txt
.venv/bin/python -m uipc doctor --probe-cuda
.venv/bin/python python/examples/cloth_cable_manipulation.py --frames 180 --preview
.venv/bin/python python/examples/cloth_cable_manipulation.py --gui
```

The standalone example generates a 289-vertex cloth and 41-vertex elastic rod,
with soft position constraints at two grips per object. In the GUI, use Run or
Step and adjust each object's XYZ sliders. Scripted motion can be disabled.
The headless path checks world validity, finite positions, target tracking,
and grip displacement. It writes metrics, trajectories, and an optional GIF
and PNG to `output/cloth_cable`. The GUI has not been interactively verified.

A 180-step headless CUDA run passed on RTX PRO 6000 Blackwell, driver 595.84:
maximum grip error was 1.30 mm for cloth and 0.086 mm for cable. Both moving
grips displaced about 16 cm. These are prescribed soft grips, not physical
robot grasps. The separated layout does not test cloth/cable mutual contact.

The neighboring Genesis checkout is version 1.1.2 and has also been set up with
pyuipc 0.0.28. Its `examples/IPC_Solver/ipc_cloth_cable.py` advances cloth and
an experimental native IPC cable through Genesis's existing coupler. See that
checkout's IPC example README for commands and integration limitations.
