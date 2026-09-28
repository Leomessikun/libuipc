"""Turn ClothesNet sleeved tops into garments the dressing pipeline can bake, hang and dress.

ClothesNet ships each top as unwelded sewing panels (centimetres, Y up). Per garment this script:

* welds panel seams (vertices within ``--weld-cm``), skipping a panel that would make an edge
  non-manifold (collars and pockets sewn on top), and keeps the largest connected piece;
* converts to metres in Wang's raw frame (Z up, sleeves along X);
* finds the two cuffs (the boundary loops at the sleeve ends) and takes the one at -X, the side
  Wang's tables use, as the dressed sleeve;
* finds the armhole seam from the panels: the welded vertices between the panels that carry that
  cuff and the rest of the garment; when that fails (sleeves cut with the torso), marches
  cross-sections from the cuff and takes the last closed sleeve section;
* writes Wang-style index tables: six armhole vertices (``shoulder_polygon``), a grasp patch on
  top of the shoulder medial to the armhole, two picker vertices at its top, and a two-vertex
  alignment line pointing from the hand side to the shoulder side.

The rules copy where Wang's tables sit on tshirt_26/68/392, in metres: the grasp centre is
11-14 cm medial of the armhole centre at the armhole's top height, the patch spans about
19 x 13 x 8 cm over the shoulder, and the pickers are its top vertices. Outputs go to
``--out`` as ``raw/<name>.obj`` plus ``index.json``; ``garment_idx_clothesnet.py`` merges them
with Wang's tables for ``UIPC_MANIP_GARMENT_INDEX_MODULE``.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def read_panels(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, int)
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    labels = trimesh.graph.connected_component_labels(trimesh.graph.face_adjacency(f), node_count=len(f))
    return v, f, labels


def edge_counts(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0, return_counts=True)


def weld(v, f, labels, weld_cm):
    """Weld panel seams panel by panel (largest first); skip panels that would break manifoldness."""
    order = np.argsort(-np.bincount(labels))
    tree = cKDTree(v)
    rep = np.arange(len(v))
    for a, b in tree.query_pairs(weld_cm):     # union-find over coincident seam vertices
        ra, rb = a, b
        while rep[ra] != ra:
            ra = rep[ra]
        while rep[rb] != rb:
            rb = rep[rb]
        if ra != rb:
            rep[max(ra, rb)] = min(ra, rb)
    for i in range(len(rep)):
        r = i
        while rep[r] != r:
            r = rep[r]
        rep[i] = r
    kept, kept_panel, skipped = [], [], []
    for panel in order:
        own = f[labels == panel]
        if not kept:
            # A panel that is non-manifold by itself loses the faces on its over-shared edges.
            edges, counts = edge_counts(rep[own])
            bad = {tuple(e) for e in edges[counts > 2]}
            if bad:
                tri = rep[own]
                keep = [not ({tuple(sorted((t[0], t[1]))), tuple(sorted((t[1], t[2]))), tuple(sorted((t[2], t[0])))} & bad)
                        for t in tri]
                own = own[np.asarray(keep, bool)]
            kept.append(own)
            kept_panel.append(np.full(len(own), panel))
            continue
        _, counts = edge_counts(rep[np.concatenate(kept + [own])])
        if (counts > 2).any():
            skipped.append(int(panel))
            continue
        kept.append(own)
        kept_panel.append(np.full(len(own), panel))
    faces = rep[np.concatenate(kept)]
    face_panel = np.concatenate(kept_panel)
    # Largest connected piece after welding.
    comp = trimesh.graph.connected_component_labels(trimesh.graph.face_adjacency(faces), node_count=len(faces))
    main = np.argmax(np.bincount(comp))
    faces, face_panel = faces[comp == main], face_panel[comp == main]
    used = np.unique(faces)
    remap = -np.ones(len(v), int)
    remap[used] = np.arange(len(used))
    return v[used], remap[faces], face_panel, skipped


def boundary_loops(faces):
    edges, counts = edge_counts(faces)
    adj = collections.defaultdict(set)
    for a, b in edges[counts == 1]:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    loops, seen = [], set()
    for s in adj:
        if s in seen:
            continue
        comp, stack = [], [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        loops.append(np.asarray(comp, int))
    return loops


def ring_order(points, centre, axis):
    ref = points[0] - centre
    ref -= (ref @ axis) * axis
    ref /= np.linalg.norm(ref)
    other = np.cross(axis, ref)
    rel = points - centre
    return np.argsort(np.arctan2(rel @ other, rel @ ref))


def geometric_armhole(V, faces, cuff, step=0.015):
    """March cross-sections from the cuff along the sleeve; the armhole is the last closed sleeve section
    before a section leaves the sleeve (no closed loop near the centreline, or one much longer)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wang_transfer"))
    from physical_sleeve import SleeveSections

    centre = V[cuff].mean(0)
    axis = np.array([1.0, 0.0, 0.0])                     # the -X sleeve runs towards +X
    ring = np.linalg.svd(V[cuff] - centre)[2][2]
    axis = ring if ring @ axis > 0 else -ring
    perimeters, last = [], None
    for k in range(1, 80):
        section = None
        for jitter in (0.0, 1e-4, -1e-4, 3e-4):
            try:
                section = SleeveSections._section(V, faces, centre, axis, 1.0, step + jitter)
                break
            except ValueError:
                continue
        if section is None:
            break
        edges, weight = section
        pts = V[edges[:, 0]] * (1 - weight[:, None]) + V[edges[:, 1]] * weight[:, None]
        perim = float(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum())
        if (np.linalg.norm(pts.mean(0) - (centre + step * axis)) > 0.05
                or (len(perimeters) >= 2 and perim > 1.5 * np.median(perimeters[-3:]))):
            break
        perimeters.append(perim)
        new_centre = pts.mean(0)
        direction = new_centre - centre
        if np.linalg.norm(direction) > 1e-6:
            axis = 0.7 * axis + 0.3 * direction / np.linalg.norm(direction)
            axis /= np.linalg.norm(axis)
        centre, last = new_centre, edges
    if last is None or len(perimeters) < 3:
        raise ValueError("no closed sleeve section beyond the cuff")
    # The cuff-side endpoint of every crossed edge of the last sleeve section.
    signed = (V - centre) @ axis
    return np.unique(np.where(signed[last[:, 0]] < signed[last[:, 1]], last[:, 0], last[:, 1]))


def ring_shape(points):
    """Mean radius and flatness (smallest over largest principal extent) of a ring of points."""
    c = points.mean(0)
    sv = np.linalg.svd(points - c, compute_uv=False)
    return float(np.linalg.norm(points - c, axis=1).mean()), float(sv[2] / max(sv[0], 1e-12))


def check_semantics(V, cuff, seam, grasp_centre, picker):
    """Reject annotations that visual QA showed to be wrong: an 'armhole' that is a torso seam, a hem
    taken for a cuff, a grasp below or lateral of the armhole."""
    cuff_r, cuff_flat = ring_shape(V[cuff])
    arm_r, arm_flat = ring_shape(V[seam])
    arm_c = V[seam].mean(0)
    far = float(np.linalg.norm(V[seam] - arm_c, axis=1).max())
    if cuff_r > 0.12:
        raise ValueError(f"QA: cuff radius {cuff_r:.3f} m (hem or slit taken for a cuff)")
    if not (0.03 < arm_r < 0.13) or far > 0.2 or arm_flat > 0.35:
        raise ValueError(f"QA: armhole ring radius {arm_r:.3f} m, max {far:.3f} m, flatness {arm_flat:.2f}")
    if grasp_centre[2] < arm_c[2] + 0.03 or grasp_centre[0] < arm_c[0] + 0.04:
        raise ValueError("QA: grasp patch not above and medial of the armhole")
    if V[picker][:, 2].min() < arm_c[2]:
        raise ValueError("QA: pickers below the armhole centre")


def prepare(path, name, out, weld_cm=0.05, target_edge_m=None):
    v, f, labels = read_panels(path)
    vw, fw, face_panel, skipped = weld(v, f, labels, weld_cm)
    V = np.column_stack([vw[:, 0], -vw[:, 2], vw[:, 1]]) / 100.0          # cm, Y up -> m, Z up
    edges, counts = edge_counts(fw)
    report = dict(source=str(path), name=name, panels=int(labels.max() + 1), skipped_panels=skipped,
                  vertices=len(V), faces=len(fw), nonmanifold_edges=int((counts > 2).sum()),
                  median_edge_m=float(np.median(np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1))))
    loops = boundary_loops(fw)
    report["loops"] = len(loops)
    centres = np.array([V[l].mean(0) for l in loops])
    mid_x = 0.5 * (V[:, 0].min() + V[:, 0].max())
    # Cuffs: the loop farthest out on each side, both well outside the torso.
    left = int(np.argmin(centres[:, 0])); right = int(np.argmax(centres[:, 0]))
    width = V[:, 0].max() - V[:, 0].min()
    if not (centres[left, 0] < mid_x - 0.25 * width and centres[right, 0] > mid_x + 0.25 * width):
        raise ValueError(f"no sleeve cuffs: loop centres x {np.round(centres[:, 0] - mid_x, 3).tolist()}")
    cuff = loops[left]
    cuff_c = V[cuff].mean(0)
    # Armhole: welded seam between the cuff's panels and the rest.
    vert_panels = collections.defaultdict(set)
    for tri, p in zip(fw, face_panel):
        for x in tri:
            vert_panels[int(x)].add(int(p))
    sleeve_panels = set().union(*(vert_panels[int(x)] for x in cuff))
    seam = np.array([x for x, ps in vert_panels.items() if ps & sleeve_panels and ps - sleeve_panels], int)
    method = "panels"
    ok = len(seam) >= 12
    if ok:
        # A seam that runs on past the armhole (into a shoulder seam) has loose ends in the edge graph.
        edges_all, _ = edge_counts(fw)
        inside = np.isin(edges_all, seam).all(1)
        degree = collections.Counter(edges_all[inside].ravel().tolist())
        ok = sum(degree[int(x)] <= 1 for x in seam) < 2
    if ok:
        seam_c = V[seam].mean(0)
        sleeve_len = np.linalg.norm(seam_c - cuff_c)
        ok = 0.08 < sleeve_len < 0.9 and seam_c[0] > cuff_c[0]
    if not ok:
        seam = geometric_armhole(V, fw, cuff)
        method = "sections"
        seam_c = V[seam].mean(0)
        if not (0.08 < np.linalg.norm(seam_c - cuff_c) < 0.9 and seam_c[0] > cuff_c[0]):
            raise ValueError(f"armhole not found (panels: {len(seam)} seam vertices; sections gave a bad seam)")
    axis = (seam_c - cuff_c) / np.linalg.norm(seam_c - cuff_c)
    order = seam[ring_order(V[seam], seam_c, axis)]
    polygon = order[np.linspace(0, len(order), 6, endpoint=False).astype(int)]
    # Grasp patch over the shoulder ridge, 13 cm medial of the armhole centre (Wang: 11.6-14 cm), at
    # the ridge's height there; the ridge is taken in the armhole's own front-back position.
    medial = np.array([1.0, 0.0, 0.0])
    for inset in (0.13, 0.115, 0.10, 0.085, 0.07):      # a wide neckline leaves no ridge at 13 cm
        tx = seam_c[0] + inset
        band = (np.abs(V[:, 0] - tx) < 0.02) & (np.abs(V[:, 1] - seam_c[1]) < 0.04)
        if band.any():
            break
    else:
        raise ValueError("no shoulder ridge above the armhole")
    centre = V[band][np.argmax(V[band][:, 2])]
    dist = np.linalg.norm(V - centre, axis=1)
    patch = np.argsort(dist)[:150]
    patch = patch[dist[patch] < 0.10]
    # Pickers: Wang's sit 4-5 cm medial of and 5 cm above the patch centre, 2 cm apart.
    pick_target = centre + np.array([0.045, 0.0, 0.05])
    ranked = patch[np.argsort(np.linalg.norm(V[patch] - pick_target, axis=1))]
    first = ranked[0]
    second = ranked[1:][np.argmin(np.abs(np.linalg.norm(V[ranked[1:]] - V[first], axis=1) - 0.02))]
    picker = np.array([first, second])
    shoulder_end = patch[np.argmin(np.linalg.norm(V[patch] - (centre + 0.02 * medial), axis=1))]
    hand_end = patch[np.argmin(np.linalg.norm(V[patch] - (centre - 0.045 * medial), axis=1))]
    check_semantics(V, cuff, seam, centre, picker)
    report.update(cuff_vertices=len(cuff), armhole_vertices=len(seam), armhole_method=method,
                  sleeve_length_m=float(np.linalg.norm(seam_c - cuff_c)),
                  armhole_radius_m=float(np.linalg.norm(V[seam] - seam_c, axis=1).mean()),
                  cuff_radius_m=float(np.linalg.norm(V[cuff] - cuff_c, axis=1).mean()),
                  grasp_vertices=len(patch), grasp_centre_minus_armhole_m=(centre - seam_c).round(4).tolist())
    (out / "raw").mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(V, fw, process=False).export(out / "raw" / f"{name}.obj")
    tables = dict(grasping_particle_indices=patch.tolist(), picker_picking_particle_indices=picker.tolist(),
                  shoulder_polygon_particle_indices=polygon.tolist(),
                  alignment_line_indices=[int(shoulder_end), int(hand_end)], cloth_scales=1.0,
                  pull_schedules=[[0.0, -1.0, 0.0], 0.15, 1.5, 1.5], cuff=cuff.tolist(), armhole=seam.tolist())
    return report, tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("objs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weld-cm", type=float, default=0.05)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    index_path = a.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    reports = []
    for obj in a.objs:
        name = "cn_" + obj.stem.lower().replace("-", "_").strip("_")
        try:
            report, tables = prepare(obj, name, a.out, a.weld_cm)
            index[name] = tables
            report["ok"] = True
        except Exception as exc:
            report = dict(source=str(obj), name=name, ok=False, error=repr(exc))
        reports.append(report)
        print(json.dumps(report), flush=True)
    index_path.write_text(json.dumps(index))
    with (a.out / "prepare_report.jsonl").open("a") as fh:
        for r in reports:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
