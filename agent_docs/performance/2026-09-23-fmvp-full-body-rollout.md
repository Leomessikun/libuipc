# FMVP checkpoint with the complete SMPL-X body in Genesis+IPC

> Endpoint correction: the later [seven-ring sleeve audit](2026-09-23-ipc-vbd-sleeve-audit.md)
> found **zero** strict full-sleeve trajectories among the two 14046
> upper-arm-ratio crossings or four new crossings. The upper-arm prefixes
> reported below reached a scalar travel threshold; they are not verified
> complete dressing demonstrations. The current full-body extractor requires
> a topology audit to prevent that label from propagating into new datasets.

> Later same-day update: these rollouts used the default
> `cloth_strain_rate=100` with FMVP rotation suppressed. Follow-up experiments
> found upper-arm crossings on previously failed bodies at `strain_rate=10`
> with rotation enabled. See [the checkpoint and material audit](2026-09-23-wang-checkpoint-frame-and-placement.md#with-the-complete-body-every-controller-jams-at-the-shoulder-unless-the-cloth-may-stretch).
> The default-material failure below is therefore not evidence that the
> checkpoint itself needs retraining to pass the shoulder.

The IPC implementation calls `strain_rate` a Baraff–Witkin over-stretch
amplification rate: it multiplies a cubic energy term once in-plane stretch
exceeds rest length. It is not a measured fabric strain limit or an allowable
extension percentage. Lowering it can make the shirt pass without showing
that the simulated knit matches a real shirt.

The arm-only IPC trajectories cannot be treated as full-body dressing data.
This pilot put the complete fixed SMPL-X mesh (10,475 vertices, 20,908 faces)
into the cloth-contact solver, using the model in
`/home/ge47gax/kun/newton-fmvp/models_smplx_v1_1/models`. The checkpoint still
receives its original right-arm point-cloud observation. Arm progress and
right-arm force are computed from the corresponding vertices of the full-body
collider; separate whole-body contact force is recorded. The cloth and full
person are drawn in the native Genesis viewer.

Set `--collision-geometry full_body` in the collector or training launcher.
The historical `arm` setting remains the default, and training resume pins
the saved collision choice. The full-body initial placement checks the shirt
against every body triangle, rather than the isolated arm. The three preflighted
gravity-hung starts had 4.6–14.9 mm of initial full-body clearance, so the
rollout differences are from the ensuing contacts rather than an illegal
initial overlap.

## Matched pilot, original `fmvp_sim.pt`, zero FiLM force, `tshirt_26`

Both bodies used seed 1004, the same 5 mm upward held-garment offset and the
same 5-decision external hold check after upper-arm ratio first reached 0.7.
The 1,000 N gripper-load cutoff screens gross simulation outliers; it is not
a real-world safety threshold.

| Body | Arm-only | Full-body IPC | Full-body non-arm contact |
| --- | --- | --- | --- |
| 14045 | upper-arm ratio 0.994, valid 5-step hold | reached forearm 0.742, then lost grasp; upper arm 0; stopped at 1,023 N on decision 229 | first >10 N at decision 189, peak non-arm resultant 512 N |
| 14046 | upper-arm ratio 0.929, valid 5-step hold | crossed 0.7 at decision 207, minimum 0.713 during 5-step hold, valid grasp; gripper p90/peak 211/581 N | first >10 N at decision 182, peak non-arm resultant 405 N |

On body 14045, handing off to the scripted expert at forearm 0.503 (state 102)
did not rescue it: upper-arm coverage peaked at 0.126 and load reached 1,594 N
at decision 191. The hybrid is retained as a failed raw episode, not accepted
training data. Another candidate, body 5045, could not be placed legally at
this 5 mm offset under any tested 5-degree cloth rotation. A geometry search
found legal starts only after roughly 5 cm of sideways shift in the tested
direction, outside the small-start variation used for checkpoint transfer.

The separate selected dataset is
`output/uipc_manip/fmvp_full_body_prefixes_20260923/manifest.json`: three
checkpoint-only prefixes, **one upper-arm and two forearm**, totaling **412
aligned actions**. The two forearm clips come from the same body and nominal
start; this is a small pilot, not a diverse full-body demonstration set. The
arm-only dataset remains separate. The checkpoint does not autonomously stop
at the upper-arm threshold, and no full garment or bilateral dressing success
has been shown.

Replay all three selected clips in Genesis:

```bash
bash scripts/wang_transfer/view_checkpoint_prefixes.sh \
  output/uipc_manip/fmvp_full_body_prefixes_20260923/manifest.json
```

The recorded [successful upper-arm endpoint](../../output/uipc_manip/fmvp_full_body_prefixes_20260923/genesis_upperarm_body14046.png)
is a screenshot of the native Genesis viewer, not Matplotlib. The selected
NPZ files retain the full person mesh for interactive replay.

Validation: both full-body IPC rollouts built and stepped with arm-specific
force readout; the matched arm-only cached-cell smoke tests passed in the main
package (2 cases) and the checkpoint worktree (1 case). All selected NPZ
clips have checkpoint-only actions, valid grasps, finite states, exact
state/action alignment and one checkpoint hash. The native Genesis viewer
was opened at a failed contact frame and the successful upper-arm endpoint.

## Repeating collection across new bodies

Use the released `fmvp_sim.pt` directly. Training is not needed to collect
checkpoint trajectories. Keep the checkpoint's right-arm point cloud and zero
FiLM force input while testing transfer, and save the *whole* raw attempt before
selecting any training clips. The CPU preflight below checks the exact
full-body cloth placement before launching IPC. It reproduced the previously
measured turns and gaps for bodies 14045 and 14046 to within numerical noise.

```bash
PY=/home/ge47gax/kun/genesis-world/.venv/bin/python
HANG=output/uipc_manip/fmvp_better_rollouts_20260923/hang2.npz

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" \
  scripts/wang_transfer/preflight_full_body_starts.py \
  --hang "$HANG" --bodies 14055 14056 14057 14058 14059 \
  --out output/uipc_manip/full_body_preflight_example.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" \
  scripts/wang_transfer/collect_better_rollouts.py \
  --hang "$HANG" --collision-geometry full_body \
  --preflight-manifest output/uipc_manip/full_body_preflight_example.json \
  --placement-offset-mm 0 5 0 --variants baseline --steps 250 --hold 5 \
  --abort-gripper-force 1000 \
  --out output/uipc_manip/full_body_raw_example

"/home/ge47gax/kun/newton/.venv/bin/python" \
  scripts/wang_transfer/audit_sleeve_topology.py \
  output/uipc_manip/full_body_raw_example/body_*/*.npz \
  --out output/uipc_manip/full_body_raw_example/topology_audit.json

python3 scripts/wang_transfer/extract_checkpoint_prefixes.py \
  --sources output/uipc_manip/full_body_raw_example \
  --topology-audit output/uipc_manip/full_body_raw_example/topology_audit.json \
  --out output/uipc_manip/full_body_prefixes_example \
  --hold 5 --max-gripper-peak-N 1000

bash scripts/wang_transfer/view_checkpoint_prefixes.sh \
  output/uipc_manip/full_body_prefixes_example/manifest.json
```

`--bodies` can select a subset of legal body IDs from the preflight manifest.
The extractor keeps only the FMVP-controlled actions through the first
held forearm or upper-arm ratio milestone. For new full-body datasets it also
requires the saved-state ring audit: an upper-arm prefix needs multi-ring
retention throughout its five-decision hold; a forearm prefix is explicitly
labeled partial cuff progress. The validation decisions are excluded from the
training clip. The force cutoff is a simulation-only gross-outlier screen,
not a real-world safety value. Inspect selected clips in the **native Genesis
viewer**; no Matplotlib plot represents garment geometry here. Keep different
cloth meshes, checkpoints, or collision geometry in distinct datasets.

## Six-body expansion and controlled variants

A geometry preflight on body IDs 14045–14054 at the 5 mm upward offset found
legal starts for all ten tested bodies. The CPU preflight reproduced the two
earlier placements (turn and gap) to within numerical noise. This checks only
initial overlap, not dressing success.

The original FMVP actions on four **new** bodies (14047–14050) reached and
held the forearm threshold on all four, but reached upper-arm ratio 0.7 on
none. Every raw run was saved, including those stopped after invalid grasp or
the 1,000 N simulation-only load cutoff. Together with the earlier 14045 and
14046 baselines, this is six distinct full-body starts: **one verified
upper-arm prefix and five forearm prefixes**, 707 executed actions through
their milestones. Non-arm contact resultant is 0–5 N through the new forearm
milestones; the failure occurs later. The 14046 upper-arm baseline had a
405 N peak non-arm contact resultant through its milestone, so this geometric
success is not a safety-certified demonstration.

The deterministic `half` variant scales FMVP actions to 0.5 after the
gripper passes 90% of the finger-to-shoulder projection. On body 14046, it
still held upper-arm coverage for five decisions, at decision 283 rather
than 207. Relative to the original prefix, its simulated gripper peak fell
from 581 to 486 N (p90 from 212 to 167 N), and non-arm contact resultant
peak fell from 405 to 156 N (p90 from 107 to 30 N). It did **not** rescue
bodies 14047 or 14048; both stopped again without upper-arm coverage. The
selected `half` trajectory stores scaled executed `actions`, raw FMVP
`policy_actions`, and `speed_scale`.

Lowering cloth density from 3,333 to 750 kg/m³ changed the shirt mass from
about 0.896 to 0.202 kg and the initial gripper load from roughly 10 to 3 N
on body 14047. The run still stalled and exceeded the 1,000 N cutoff. Its
raw data remain separate from default-physics trajectories. Lower mass alone
did not solve this failure.

The broad [geometric manifest](../../output/uipc_manip/fmvp_full_body_curated_geometric_20260923/manifest.json)
has six distinct body starts (one upper-arm, five forearm), 783 selected
actions, with a 1,000 N peak screen. The more selective
[default-physics manifest](../../output/uipc_manip/fmvp_full_body_selected_20260923/manifest.json)
has four distinct starts (one upper-arm, three forearm), 589 actions, with a
600 N *simulation-only* peak screen. This threshold ranks gross outliers;
it is not a real-world allowable force. All selected clips have aligned
observations/actions, valid grasps, FMVP controller IDs, finite cloth states,
one checkpoint hash, and the full-body mesh. The modified `half` clip is
marked `action_scaling_applied` in the manifest.

The [half-speed upper-arm endpoint](../../output/uipc_manip/fmvp_full_body_half14046_prefix_20260923/genesis_half_upperarm_body14046.png)
was replayed and captured in the **native Genesis viewer**. Replay the
selected set interactively with:

```bash
bash scripts/wang_transfer/view_checkpoint_prefixes.sh \
  output/uipc_manip/fmvp_full_body_selected_20260923/manifest.json
```

These results support checkpoint-guided *forearm-prefix collection* and one
lower-load upper-arm variant under the default `strain_rate=100`. They do not
support claiming a large dataset of complete full-body dressing
demonstrations. At that material setting the original pull stalls after
forearm threading; force reduction and lower cloth mass alone did not resolve
the tested failures. The later material sweep shows why stretch calibration
must precede a decision to redesign the controller.
