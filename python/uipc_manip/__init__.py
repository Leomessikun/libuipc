"""Robot cloth and cable manipulation pretraining on Genesis with the libuipc IPC solver.

The package is intentionally import-light: the Genesis environment and the
PyTorch agent are imported lazily by the modules that need them, so the
observation layout and task definitions can be used without a GPU.
"""

__version__ = "0.1.0"
