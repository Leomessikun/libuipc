# Static flow baseline: reconciled v4/v5 recordings

## Authorized scope

On 2026-09-27 the owner approved the explained sequence: reconcile existing
FMVP data, establish static closed-loop competence, test the value and
learnability of future human motion, then collect targeted corrections.
This stage covers the first two steps on `research/anticipatory-dressing`.
GRAB is not included in this static training dataset. Preserve the original
workspace and its active collection; GPU sharing remains unauthorized under
the earlier no-overlap preference.

## Common data audit

`scripts/wang_transfer/prepare_flow_dataset.py` reuses `rollout_chunks.prepare`
and the existing `physical_sleeve` geometry. It freezes both source manifests,
copies observation/command arrays into a fresh cache and records source index,
collection flags, placement, source configuration hashes and audit code hashes.
It also verifies every terminal cloth state over the 20 zero-command decisions:
all three interior sleeve sections must wrap the arm and the armhole must reach
at least 0.7 of elbow-to-shoulder distance. Garment template topology is checked.
Recorded validity/acceptance and finite aligned arrays remain required.

The same criterion is applied to v4 and v5. A cuff beyond the fingertips may
fail the older cuff-inclusive rule while interior sections still surround the
arm. Keep that diagnostic rather than silently equating the original flags.
This audit independently recomputes terminal sleeve geometry; it does not
revalidate all collisions, grasp physics, forces, or realism during the path.

The frozen snapshot contains 810 v4 and 465 v5 episodes. All 1,275 pass, with
no rejected entries or exact observation/command-array duplicates. There are
397,729 commands in total. The terminal cuff-inclusive rule also passes for
all 810 v4 entries and 299 v5 entries; 166 v5 entries require the common
interior-only rule. These are acceptance counts, not success rates over attempts.

The cache is 8.01 GiB at
`output/anticipatory_dressing/flow_bc_v45_data`. The manifest was finalized at
2026-09-27 00:36:54 CEST. Collection continues outside this frozen snapshot.

| Split | Episodes | Body/pose IDs | Commands | v4 / v5 episodes |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,079 | 198 | 335,542 | 661 / 418 |
| Validation | 196 | 43 | 62,187 | 149 / 47 |

All old body assignments are preserved using the v4 cache manifest. New body
IDs receive a deterministic hash assignment that remains fixed as datasets
grow. Every replica and garment for a body shares that assignment. All five
garments appear on both sides. This remains a student body/pose holdout; the
FMVP teacher may have seen these bodies, and there is no garment holdout.
v5 also changes initial alignment, so a v4/v5 gain cannot be attributed solely
to data count. Original configurations and source arrays are unchanged.

Reproduce in the isolated worktree:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/ge47gax/kun/genesis-world/.venv/bin/python \
  scripts/wang_transfer/prepare_flow_dataset.py \
  --source /home/ge47gax/kun/libuipc/output/uipc_manip/fmvp_scaled_multigarment_v4_20260925 \
           /home/ge47gax/kun/libuipc/output/uipc_manip/fmvp_scaled_multigarment_v5_20260926 \
  --split-manifest output/anticipatory_dressing/flow_bc_v4_data/manifest.json \
  --out output/anticipatory_dressing/flow_bc_v45_data
```

Use a fresh output directory. Rerunning against the live source takes a newer
snapshot; the frozen `source_000_manifest.json` and `source_001_manifest.json`
identify this run's exact inputs.

## Static training

A bounded 20,000-update run was launched on CPU with two threads and low
process priority to avoid the occupied GPU. It uses the existing compact flow
BC model, random encoder initialization, 3 observation frames, 8 commands,
batch size 16 and seed 20260926. Evaluation samples 256 fixed validation windows
every 1,000 updates. Best checkpoint selection uses validation flow loss.
The output is `output/anticipatory_dressing/flow_bc_v45_train`.
Training completion and phase-specific errors still need inspection; no
closed-loop gain follows from lower offline loss.

## Verification and remaining gate

Fifteen CPU tests pass across `test_rollout_chunks.py`, `test_flow_client.py`
and `test_physical_sleeve.py`. New regressions preserve body splits when merging
and extending sources, reject conflicting reference splits, reject nonzero
terminal holds, and reject a displaced final cloth mesh even when saved success
flags remain true. A cuff beyond the fingers with wrapped interior sections
passes the chosen common criterion.

The next gate is matched FMVP/flow execution from preselected v4 validation
starts with the common endpoint rule for both controllers. Its five-case scope
is a conditional reproduction diagnostic with an external completion hold;
it cannot measure unbiased success, unseen garments, or autonomous stopping.
IPC execution must wait for the other GPU clients. Dynamic teacher/student
work remains conditional on competent static control and causal anticipation
evidence; see the [research pilot](2026-09-26-anticipatory-dressing-pilot.md).
