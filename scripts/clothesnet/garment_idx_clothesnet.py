"""Wang's garment index tables plus the ClothesNet garments written by ``prepare_garment.py``.

Point ``UIPC_MANIP_GARMENT_INDEX_MODULE`` at this file and ``UIPC_MANIP_RAW_GARMENT_DIR`` at the
asset directory's ``raw/`` (which also links Wang's raw meshes). ``CLOTHESNET_ASSETS`` selects the
asset directory; its ``index.json`` holds the per-garment tables.
"""
import importlib.util
import json
import os
from pathlib import Path

_WANG = Path("/home/ge47gax/kun/ppf-contact-solver/tools/dressing_bake/garment_idx_utils.py")
_spec = importlib.util.spec_from_file_location("wang_garment_idx_utils", _WANG)
_wang = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wang)
for _k, _v in vars(_wang).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

_ASSETS = Path(os.environ.get("CLOTHESNET_ASSETS", "/home/ge47gax/kun/libuipc/output/uipc_manip/clothesnet_assets"))
pull_schedules = {}
if (_ASSETS / "index.json").exists():
    for _name, _t in json.loads((_ASSETS / "index.json").read_text()).items():
        grasping_particle_indices[_name] = _t["grasping_particle_indices"]
        picker_picking_particle_indices[_name] = _t["picker_picking_particle_indices"]
        shoulder_polygon_particle_indices[_name] = _t["shoulder_polygon_particle_indices"]
        alignment_line_indices[_name] = _t["alignment_line_indices"]
        cloth_scales[_name] = _t["cloth_scales"]
        pull_schedules[_name] = _t["pull_schedules"]
