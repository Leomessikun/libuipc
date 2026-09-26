"""Where do failed rollouts jam? Garment part against body part at the last recorded state.

For each episode, cloth vertices within ``--gap`` of the body surface (nearest body vertex) are the
contacts. Body side: arm coordinate along fingertip -> elbow -> shoulder of the nearest body vertex
(hand past the wrist, forearm, elbow band, upper arm) or torso when it is far from that polyline.
Garment side, on the rest mesh: cuff and armhole rings (within 3 cm), sleeve (between them), torso panel.
Also records where the gripper is along the arm and whether the arm is inside the sleeve.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CLOTH3D = Path("/home/ge47gax/kun/fmvp_pb/dressing_pb/assistive_gym/assets/data/cloth3d/train/Tshirt")
_parts = {}


def garment_parts(garment):
    if garment in _parts:
        return _parts[garment]
    import importlib.util
    from physical_sleeve import SleeveSections, read_obj
    spec = importlib.util.spec_from_file_location("g", "/home/ge47gax/kun/dressing/garment_idx_utils.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    v, f = read_obj(CLOTH3D / f"{garment}.obj")
    v = v * 4.0
    arm = np.asarray(g.shoulder_polygon_particle_indices[garment])
    s = SleeveSections(v, f, arm)
    cuff_c, arm_c = v[s.cuff].mean(0), v[arm].mean(0)
    axis = arm_c - cuff_c
    length = np.linalg.norm(axis); axis /= length
    t = (v - cuff_c) @ axis
    radial = np.linalg.norm((v - cuff_c) - np.outer(t, axis), axis=1)
    r_sleeve = np.linalg.norm(v[s.cuff] - cuff_c, axis=1).mean()
    label = np.full(len(v), "torso", dtype=object)
    sleeve = (t > -0.01) & (t < length) & (radial < 1.8 * max(r_sleeve, np.linalg.norm(v[arm] - arm_c, axis=1).mean()))
    label[sleeve] = "sleeve"
    ring_d = lambda ids: np.min(np.linalg.norm(v[:, None, :] - v[ids][None, :, :], axis=2), axis=1)
    label[ring_d(arm) < 0.03] = "armhole"
    label[ring_d(s.cuff) < 0.03] = "cuff"
    _parts[garment] = label
    return label


def arm_coordinate(points, finger, elbow, shoulder):
    """Normalised position along fingertip->elbow (0..1) ->shoulder (1..2) and distance to that polyline."""
    best_s, best_d = np.zeros(len(points)), np.full(len(points), np.inf)
    for k, (a, b) in enumerate(((finger, elbow), (elbow, shoulder))):
        ab = b - a
        t = np.clip((points - a) @ ab / (ab @ ab), 0, 1)
        d = np.linalg.norm(points - (a + np.outer(t, ab)), axis=1)
        take = d < best_d
        best_s[take], best_d[take] = k + t[take], d[take]
    return best_s, best_d


def body_part(s, d, wrist_s):
    if d > 0.14:
        return "torso"
    if s < wrist_s:
        return "hand"
    if s < 0.85:
        return "forearm"
    if s < 1.15:
        return "elbow"
    return "upper_arm"


def analyse(job):
    path, garment, accepted = job
    try:
        with np.load(path) as d:
            x = d["positions"][-1]; hv = d["human_vertices"]; fi, el, sh = d["finger"], d["elbow"], d["shoulder"]
            tcp = d["tcp"][-1]; wrapped = bool(np.asarray(d["sleeve_wrapped"])[-1]); T = len(d["positions"])
    except Exception:
        return None
    labels = garment_parts(garment)
    if len(labels) != len(x):
        return None
    dist, idx = cKDTree(hv).query(x)
    touching = np.flatnonzero(dist < 0.006)
    fe = np.linalg.norm(el - fi)
    wrist_s = min(0.2, 0.09 / fe)                   # about 9 cm of hand beyond the wrist
    s, dd = arm_coordinate(hv[idx[touching]], fi, el, sh)
    pairs = collections.Counter((labels[c], body_part(a, b, wrist_s)) for c, a, b in zip(touching, s, dd))
    gs, _ = arm_coordinate(tcp[None], fi, el, sh)
    return dict(path=str(path), garment=garment, accepted=accepted, T=T, wrapped=wrapped,
                grip_s=float(gs[0]), contacts={f"{a}|{b}": n for (a, b), n in pairs.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempts", type=Path, required=True)
    p.add_argument("--per-garment", type=int, default=60)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.attempts.parent
    rows = [json.loads(l) for l in a.attempts.read_text().splitlines() if l.strip()]
    jobs, count = [], collections.Counter()
    for r in rows:
        key = (r["garment"], bool(r["accepted"]))
        if count[key] >= a.per_garment:
            continue
        path = root / r["path"] if (root / r["path"]).exists() else Path(r["log"]).with_suffix("") / r["path"]
        jobs.append((path, r["garment"], bool(r["accepted"])))
        count[key] += 1
    with ProcessPoolExecutor(20) as ex:
        res = [r for r in ex.map(analyse, jobs, chunksize=4) if r]
    a.out.write_text(json.dumps(res))
    for acc in (False, True):
        sub = [r for r in res if r["accepted"] == acc]
        tot = collections.Counter()
        for r in sub:
            n = sum(r["contacts"].values()) or 1
            for k, v in r["contacts"].items():
                tot[k] += v / n
        print(f"\n{'ACCEPTED' if acc else 'FAILED'} n={len(sub)}  mean share of contact vertices by (garment part | body part):")
        for k, v in tot.most_common(10):
            print(f"   {k:22s} {v / len(sub):.2f}")
        print("   gripper position along arm (0 tip,1 elbow,2 shoulder): median", round(float(np.median([r["grip_s"] for r in sub])), 2),
              " arm inside sleeve at end:", round(float(np.mean([r["wrapped"] for r in sub])), 2))


if __name__ == "__main__":
    main()
