# Full-body FMVP rollouts: IPC versus Newton VBD sleeve audit

The recorded upper-arm travel ratio is **not** sufficient to label a rollout
as a complete dressing demonstration. A valid endpoint must retain several
successive semantic sleeve rings around a nonzero span of the arm. This audit
uses Newton's own pure `multi_section_topology` and
`evaluate_sleeve_retention` functions on saved cloth meshes; no simulator
replay or Matplotlib is involved. The IPC recordings use the fingertip as the
available wrist-side landmark, while Newton uses its wrist landmark. Both
checks use the same seven `tshirt_26` sleeve rings, whose cuff indices match
the IPC mesh exactly.

| Saved rollout group | Scalar or original success | Multi-ring retained at endpoint | Material and start |
| --- | ---: | ---: | --- |
| Genesis+IPC, FMVP actor, body 14045/14049, two replicas each | 4/4 held upper-arm ratio >=0.7 | **0/4**; none retained at any recorded state | Full SMPL-X body, gravity-hung start, `cloth_strain_rate=10`, 0.896 kg shirt |
| Genesis+IPC, earlier body 14046 baseline and half-speed actor | 2/2 held upper-arm ratio >=0.7 | **0/2**; none retained at any recorded state | Full body, default material and no actor rotation |
| Newton+VBD, evolved heuristic, body 24 | 19/30 accepted by its axial-occupancy success gate | **0/19** saved accepted endpoints | Full-body baked state; all 30 attempts started arm-axis-threaded |

At the IPC upper-arm ratio crossing, all seven ring centers have projected
arm-chain coordinate `s=1` at the shoulder. At most the cuff and the next
ring wind around the arm; the sleeve is bunched or draped near the shoulder.
The native Genesis front and rear views agree with this failure mode. IPC's
99th-percentile cloth-edge length at the crossing is 2.0–2.6 times the first
recorded state; peak simulated gripper loads are 442–554 N. These loads are
not calibrated real-world force readings.

Newton's 19 accepted endpoints are less deformed by its own rest-edge metric
(`max_edge_strain` 0.258–0.316), and the recorded sampled body penetration is
zero. But all 19 have only the cuff and second ring wrapped, with retained
ring span at most 0.024 along the arm chain; the strict gate requires at least
three contiguous rings and span 0.15. Newton's existing success gate checks
axial sleeve-vertex occupancy, which can pass when the semantic rings bunch
near the shoulder. The Newton data therefore does not establish that VBD
produces complete sleeve trajectories. Its baked, prethreaded start and
heuristic controller also differ from the hanging-start FMVP actor, so the
success fractions are not a matched solver benchmark.

Reducing the IPC shirt density from 3333 to 750 kg/m³ (roughly 0.896 to
0.202 kg) at `cloth_strain_rate=10` reduced peak gripper load to 210–258 N,
but neither of two body rollouts reached the upper-arm threshold. Lower mass
alone did not solve the geometry or provide a calibrated jersey material.
Both did hold the forearm-ratio milestone. Through that earlier milestone,
their peak loads were only 23–26 N, versus 186–307 N for the heavy-shirt
prefixes; 99th-percentile edge-length ratios stayed around 1.30–1.32 in
both cases. The lighter-shirt clips therefore look like the better *partial*
motion source, pending material calibration. The [native Genesis forearm
view](../../output/uipc_manip/fmvp_ipc_rot_strain10_mass020_pilot_20260923/genesis_body14049_forearm_ring_centers.png)
still shows much of the garment hanging behind the body, so it is not a clean
full-sleeve demonstration.

The four heavy-shirt IPC episodes contain **384 actions** through the first
held forearm-ratio milestone. At that point the cuff and next ring encircle
the arm and the cuff has reached `s≈0.29`, so these are partial checkpoint
motion probes. They are labeled `partial_cuff_progress`, **not complete
expert demonstrations**. No strict full-sleeve checkpoint trajectory has
been collected in the audited pilot.
The lighter-shirt pair adds **two** separately labeled partial clips with
**193 actions**. Do not merge the two material conditions into one dataset.

The next useful change is to correct the dressing geometry and endpoint
criterion before collecting a large dataset or fine-tuning on purported
successes. Keep Genesis+IPC as the current FMVP bridge: it already runs the
released actor from the hanging start with the full body. Use Newton+VBD as
a material/control comparison only after matching the start and running the
released actor through an observation/action adapter. In both solvers, require
multi-ring retention, acceptable deformation, grasp integrity, and a held
endpoint. Calibrate fabric mass and force-extension response against the
intended garment before using simulated force as a safety threshold.

Reproduce the saved-state audit (Newton's Python environment has Torch):

```bash
PY=/home/ge47gax/kun/newton/.venv/bin/python
IPC=output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923
"$PY" scripts/wang_transfer/audit_sleeve_topology.py \
  "$IPC"/body_14045_seed_1004/baseline_rep{0,1}.npz \
  "$IPC"/body_14049_seed_1004/baseline_rep{0,1}.npz \
  --newton-raw /home/ge47gax/kun/newton/runs/topo_ab_fc35b699/evolved_gen7/tshirt_26@h24/raw_states.pt \
  --out "$IPC"/topology_audit.json

python3 scripts/wang_transfer/extract_checkpoint_prefixes.py \
  --sources "$IPC" --topology-audit "$IPC"/topology_audit.json \
  --out output/uipc_manip/fmvp_ipc_partial_audited_example \
  --hold 5 --max-gripper-peak-N 350 --max-edge-p99-ratio 1.4
```

Raw audit: `output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923/topology_audit.json`.
Earlier 14046 audit: `output/uipc_manip/fmvp_full_body_14046_topology_audit_20260923.json`.
The [front](../../output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923/genesis_body14045_upperarm.png),
[rear](../../output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923/genesis_body14045_upperarm_back.png),
and [ring-center overlay](../../output/uipc_manip/fmvp_ipc_rot_strain10_pilot_20260923/genesis_body14045_upperarm_ring_centers.png)
are native Genesis captures; the interactive viewer is
`scripts/wang_transfer/view_rollouts_genesis.py`.
