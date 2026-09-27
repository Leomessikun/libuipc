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
The run completed all 20,000 updates in 22,107 s (6.14 h); update 20,000 has
the best validation flow loss, 0.109342. Its sampled command MSE is 0.022145
versus 0.042839 for zero commands on the same fixed validation windows.
Checkpoint SHA256 is
`30476c7b64e5e502ab57da62b38ff7079106f0648a17c6dafd60ce5c1229c6ae`.
Lower offline loss does not establish closed-loop dressing success.

The follow-through also completed its old/new comparison on the same 196
validation episodes. The table reports RMS translation-command vector error
for the first predicted action, in millimetres, using recorded observations:

| Phase | Old v4 model | New v4/v5 model |
| --- | ---: | ---: |
| Start | 4.347 | 3.714 |
| Early | 3.262 | 3.237 |
| Middle | 1.559 | 1.359 |
| Before hold | 1.616 | 1.671 |
| Final hold | 3.187 | 2.763 |

The new model improves start/middle/hold errors, is nearly unchanged early,
and is slightly worse before hold. Both models still output nonzero commands
where the recorded terminal command is zero. This does not establish learned
autonomous stopping. These are not realized cloth/gripper tracking errors.
Data distribution and training duration both changed, so this comparison
cannot isolate the benefit of additional trajectories. Full source-specific
translation/yaw metrics are in `flow_bc_v45_stage/offline_phase_audit.json`.

## Verification and remaining gate

Fifteen CPU tests pass across `test_rollout_chunks.py`, `test_flow_client.py`
and `test_physical_sleeve.py`. New regressions preserve body splits when merging
and extending sources, reject conflicting reference splits, reject nonzero
terminal holds, and reject a displaced final cloth mesh even when saved success
flags remain true. A cuff beyond the fingers with wrapped interior sections
passes the chosen common criterion.

## Automatic follow-through and paired evaluation

`finish_static_flow_run.py` verified the final CPU training checkpoint and
completed the offline phase audit. Its current state is `ipc_runner` at
`output/anticipatory_dressing/flow_bc_v45_stage/status.json`; the nested
`ipc/status.json` reports `waiting_for_gpu`. Its sequence is:

1. `audit_flow_bc.py` compares the old v4 and new v4/v5 policies on identical
   held-out recordings. It checks that neither checkpoint trained on audited
   body IDs, evaluates start/early/middle/pre-hold/hold commands with seeded
   Gaussian sampling, and reports translation/yaw command errors by source.
   This is teacher-forced offline inference, not realized physical tracking.
2. `eval_static_flow.py` checks checkpoint/data identity and source environment,
   hang, actor and audited configuration hashes. It then queues five matched
   FMVP/flow cases, one per garment, with the same common interior-wrap and
   armhole endpoint. `collect_garment.py --sections-wrap` implements that rule
   for all controller slots and records the selected interpretation explicitly.
3. The IPC GPU guard waits until other non-desktop CUDA clients are absent.
   It never terminates or modifies collection jobs.

The training wait and initial GPU wait are each bounded at 24 hours. The IPC
execution budget is two hours after the initial GPU wait. An expired wait or
failed check sets an error state; it does not imply a completed evaluation.
Training and the offline audit are complete. No paired IPC case has executed
yet; all five remain pending while existing GPU clients finish.

Preflight resolves the same v4 starts as the original five-case diagnostic:
body 10040 for `tshirt_26`, `tshirt_4`, `tshirt_68`, `hospital_gown`, and 25040
for `tshirt_392`. They remain held out after merging. The common rule is
applied to both methods even though these source demonstrations used the
older cuff-inclusive rule. The runner explicitly rejects unsupported v5
alignment/material starts, preventing an accidental v4-placement substitution.
No v5-placement closed-loop claim is made. `--prepare-only` saves cases without
launching simulation.

Two new reproduction-guard tests pass (17 targeted CPU tests in total): reject
a checkpoint paired with another training manifest, and reject unsupported
placement flags. A real five-case CPU preflight also passes and confirms the
common endpoint flag on every command. A deliberate old-checkpoint/new-data
combination is rejected. Python compilation and `git diff --check` pass.

The five-case scope is a conditional reproduction diagnostic with an external
completion hold; it cannot measure unbiased success, unseen garments, or
autonomous stopping. Dynamic teacher/student work remains conditional on
competent static control and causal anticipation evidence; see the
[research pilot](2026-09-26-anticipatory-dressing-pilot.md).

## Moving from collection to evaluation

The owner asked whether the accumulated trajectories suffice to move on.
They suffice for the first static closed-loop diagnostic; no universal sample
threshold or dynamic/unseen-garment competence follows from the count.
At the scheduling snapshot, v4 has 810 accepted entries and v5 has 1,438:
2,248 entries / 706,276 commands across the same five garments. The extra
entries are not all re-audited or included in the trained frozen cache.
Do not restart training solely to consume every newly arriving demonstration
before evaluating the existing model.

To obtain a serial evaluation window without interrupting running collection
batches, `collection_eval_lease.py` temporarily SIGSTOPs only dispatcher PID
550064. Its six existing collector children continue writing their own logs
and archives. Their ledger entries will be processed when the dispatcher is
resumed. The selected collection path and readable manifest were verified
after the pause. Original source files and trajectory files are unchanged.

The active lease is recorded in `flow_bc_v45_stage/collection_eval_lease.json`
and currently reports `draining_collection`. The already queued evaluator is
PID 3029525. The lease restores the same dispatcher with SIGCONT after the
evaluation completes or errors. If evaluation does not start within one hour,
or exceeds a 7,500 s execution window, it cancels only our evaluation subtree
and restores collection. Other GPU users are never signaled; a new unrelated
GPU job can still delay this window. The dispatcher pause is a temporary
scheduling action, not termination of collection or its workers.

Two process tests passed: the dispatcher is stopped while its child completes,
and the dispatcher resumes after either normal evaluation completion or a
timeout. There are now 19 targeted CPU tests across the data, geometry,
inference, evaluation-protocol and scheduling checks. Read the lease and nested
IPC status before launching another evaluator or managing the dispatcher.
