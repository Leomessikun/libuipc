# Checkpoint collection: verified changes and rejected interventions

The checkpoint is capable of producing full-body-collision, single-sleeve
trajectories in Genesis+IPC. At this audit the original collection contains
75 accepted episodes, 28,387 transitions, and eight distinct SMPL-X body IDs:
14046 (25), 14047 (10), 14051 (5), 14052 (1), 14054 (5), 14055 (14),
14057 (3), and 14059 (12). These share one T-shirt and pose region 13; they
are not 75 independent body shapes or a complete two-arm dressing dataset.
An archive integrity audit confirms 75 distinct recording hashes, eight
distinct body mesh hashes, and aligned state/action arrays. There are 26,887
checkpoint-controlled transitions and 1,500 stationary-hold transitions;
no scripted-expert transitions occur in these accepted files. Speed variants
are baseline (22 episodes), half (23), and quarter (30).

The actor remains `/home/ge47gax/Desktop/fmvp_sim.pt`, SHA256
`eef38b84f95a0ca099d03cb92bbf00da0af29b58640837fe307d3d2635b4af97`.
No checkpoint weights were trained or modified in these experiments.

## What the controlled comparison established

`output/uipc_manip/fmvp_contract_pilot_v1_20260924` ran four quarter-speed
profiles on bodies 14045, 14053, and 14046, with the same seed and requested
placement. All used full SMPL-X collisions, the gravity-hung shirt, density
750 kg/m³ (approximately 0.2015 kg total mass), strain coefficient 10,
FMVP vertical rotation, and a 20-decision stationary hold.

The endpoint is explicitly changed: all four actual sleeve sections must wrap
the arm, with the proximal section at least 90% along elbow-to-shoulder.
Acceptance also requires a valid grasp throughout, no simulation/load-cutoff
failure, and peak edge ratios versus the initial state of p99 <= 2.25 and
maximum <= 4. These are simulation filters, not calibrated textile limits.
The independent audit verifies zero executed commands during the hold; merely
passing through the target does not qualify.

| Input/control profile | 14045 | 14053 | 14046 |
| --- | --- | --- | --- |
| Original inputs, revised stop | fails grasp | passes | passes |
| Remove second voxel filter | incomplete, fails grasp | passes | load cutoff, fails grasp |
| Single voxel + experimental force adapter | incomplete, fails grasp | passes | fails grasp |
| Single voxel + tracking guard | stalls | passes | stalls |

The five accepted pilot episodes are four variants on 14053 and the original
input profile on 14046. They are not five different bodies or repeated trials
of one configuration. The original-input 14053 rollout has 565 transitions,
51.8 N peak simulated gripper load, and edge p99/max 1.950/2.415. An earlier
independent seed on 14053 also passed at quarter speed; see the body-dependent
diagnostic document. Body 14045 remains unresolved.

The collection handoff therefore selects **original checkpoint inputs with
the revised stopping rule**. The single-voxel change is not promoted merely
because it resembles the source implementation more closely.

## Actor and observation checks

`audit_fmvp_contract.py` compares the bridge with the released actor on 40
saved PyBullet observations. Maximum action error is 3.0e-8. This rules out a
weight-loading or neural-network implementation discrepancy on those inputs;
it does not establish observation or dynamics equivalence between simulators.

On 54 IPC observations from six bodies, the bridge's extra mixed-category
voxel filter removes a median 32.8% of cloth points. Removing it changes the
normalized action by median L2 0.1516, maximum 0.8014. The rollout comparison
above demonstrates that removing it is not a universal performance fix.

The source PyBullet environment voxelizes cloth and arm separately, but it
also hides cloth while rendering the arm (`assistive_gym/dressing.py:320`).
Thus the prior claim that bare-arm visibility itself disagrees with the
released simulation checkpoint was incorrect. The paper's description and
the released implementation must be distinguished.

PyBullet contact-force inputs are not calibrated IPC newtons. The trial
adapter is explicitly labeled experimental. The selected production profile
continues to supply zero force; recording physical forces does not mean they
are being fed to or learned by the network.

## Grasp and lookahead experiments

Soft-position stiffness scales with vertex mass, so lowering density also
weakens the grasp at fixed constraint strength. Nevertheless, compensating
the stiffness by 4.444x did not rescue body 14045: all four grasp-pilot variants
hit the load cutoff, including one 12-anchor variant at 2.61 kN. These changes
are excluded from collection. The 12-anchor variant does not reproduce the
released PyBullet grasp, which uses a different, broad anchor patch.

An isolated, bounded IPC candidate-action experiment is saved separately at
`output/uipc_manip/fmvp_ipc_filter_pilot_20260924`. It changes executed actions,
not network weights, and is not included in the production recipe. Candidate
simulation restores IPC and Python state; nominal replay differed by 0.20 mm
in the first comparison, so scores use a measured noise margin. Lower local
force or higher predicted progress is not proof of a successful trajectory.
Its final independent audit, when available, is `independent_audit.json`.

## Continuing collection without an agent

`promote_verified_collection.py` independently audited the controlled pilot
and selected original inputs after passes on 14046 and 14053. It requests the
old collector to finish its current batch, then starts the revised collection
at `output/uipc_manip/fmvp_dataset_v2_500_20260924`:

- target 500 accepted episodes, maximum 3,000 attempts;
- body IDs 14045–14084, half and quarter speed, randomized legal placements;
- full-body collision and the revised, recorded endpoint;
- old recordings retained and independently re-audited before reuse;
- no model/API calls; progress in `status.json`, accepted files in
  `manifest.json`, all outcomes in `attempts.jsonl`.

Old and new counts must not be added: the new manifest can reference old files
and uses a different endpoint. Source profiles remain recorded for reused
pilot data. A 40-body sampling pool does not imply 40 successful bodies.

Newly launched workers also record checkpoint provenance, policy force input,
controller interventions, grasp indices, and relative virtual-tool rotation.
Older pilot files do not retroactively gain those fields. These are virtual
gripper trajectories, not Franka joint-space trajectories.

Eight regression tests pass: sleeve topology/wrapping, rejection of moving
target crossings as holds, force sign/clipping/smoothing, observation cropping,
rotation-aware grasp guarding, and virtual-tool rotation recovery. The multi-body simulations above provide
the end-to-end check. Visualization remains in the native Genesis viewer.
