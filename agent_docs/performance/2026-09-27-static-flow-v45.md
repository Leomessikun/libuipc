# Static flow baseline: reconciled v4/v5 recordings

## Completed five-case diagnostic

All five selected cases now have full results:

| Garment / body | FMVP | Flow |
| --- | --- | --- |
| tshirt_26 / 10040 | accepted | accepted |
| tshirt_4 / 10040 | accepted | accepted |
| tshirt_68 / 10040 | accepted | accepted |
| hospital_gown / 10040 | grasp failure | accepted |
| tshirt_392 / 25040 | grasp failure | decision limit without success |

The aggregate is `flow_bc_v45_stage/static_five_case_summary.json`, including
source-status paths and hashes, checkpoint identity and initial-state errors.
These four versus three accepted cases are not unbiased success-rate estimates:
starts were selected from accepted demonstrations, there are only two body/pose
IDs, and independently settled cloth differs by 3.12--7.91 mm between methods.
Do not infer superiority, dynamic competence, or new-garment generalization.

The last `tshirt_392` run took 2,019.14 s wall time. FMVP lost grasp at decision
170. Flow retained valid grasp throughout 750 decisions with no solver error,
but never met the three-interior-section wrap criterion; both legacy forearm
and upper-arm progress remained zero. Its final 100 TCP states span 80.2 mm,
and its last 100 mean executed translation norms are 1.57 mm per decision.
Thus the unsuccessful rollout continues moving while failing sleeve geometry;
this evidence does not identify a unique cause or prove a recovery solution.
Initial cloth mismatch in this pair is 3.12 mm. Preserve this failed rollout
for diagnosis and possible corrective labeling, not successful imitation.

## Authorized scope

On 2026-09-27 the owner approved the explained sequence: reconcile existing
FMVP data, establish static closed-loop competence, test the value and
learnability of future human motion, then collect targeted corrections.
This stage covers the first two steps on `research/anticipatory-dressing`.
GRAB is not included in this static training dataset. Preserve the original
workspace and its active collection. The owner later authorized shared GPU
evaluation on 2026-09-27; the continuation below supersedes the earlier
no-overlap preference. Pausing collection remains unauthorized.

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
completed the offline phase audit. The stage and nested IPC status files now
report `interrupted` after the owner requested uninterrupted collection (see
the scheduling correction below). That evaluator exited; the separately
authorized shared-GPU continuation is recorded below. Its sequence was:

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
Training and the offline audit are complete. Two IPC cases completed before
the evaluation was canceled; the third was interrupted and two remain pending.

| Garment / body | FMVP accepted / first success | Flow accepted / first success | Maximum initial cloth difference |
| --- | --- | --- | ---: |
| tshirt_26 / 10040 | yes / decision 292 | yes / decision 305 | 4.01 mm |
| tshirt_4 / 10040 | yes / decision 295 | yes / decision 291 | 5.52 mm |

Both methods completed the 20-decision hold with valid grasps and no simulator
error in these two cases. The initial TCP positions agree, but independently
settled cloth states do not. These are preliminary completions on two selected
static starts, not exactly matched evidence of improvement, an unbiased
success rate, unseen-garment transfer, or learned autonomous stopping.

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

## Collection continues; evaluation scheduling correction

The owner asked whether the accumulated trajectories suffice to move on.
They suffice for the first static closed-loop diagnostic; no universal sample
threshold or dynamic/unseen-garment competence follows from the count.
At the scheduling snapshot, v4 has 810 accepted entries and v5 has 1,438:
2,248 entries / 706,276 commands across the same five garments. The extra
entries are not all re-audited or included in the trained frozen cache.
Do not restart training solely to consume every newly arriving demonstration
before evaluating the existing model.

The agent used `collection_eval_lease.py` to pause dispatcher PID 550064 while
its six active children finished, then ran the queued evaluator. The owner
objected: the authorization to advance the research did not authorize pausing
collection. The agent terminated only its evaluation guardian and evaluation
subtree, and the guardian resumed the original dispatcher. Verification found
the same dispatcher identity in state S and six new `collector.py` processes.
The lease record reports `collector_resumed: true`; its interruption error is
retained as history. No collection worker was signaled and no trajectory file
was deleted. Parent and nested evaluation statuses explicitly record this
interruption and preserve both completed cases and partial output.

After resumption, v5's ledger contains 1,464 accepted entries / 463,236 commands,
giving 2,274 v4/v5 entries / 715,142 commands. This growing raw count is not the
audited training cache. More varied trajectories remain useful; the evidence
does not identify a data-sufficiency threshold for the final research task.

Do not restart this lease or pause collection to create a GPU window without
an explicit owner request. Keep collecting; the later shared-GPU authorization
below permits concurrent evaluation. The previous 19 targeted
CPU tests included scheduling recovery checks; those checks do not establish
authorization to change another job's schedule. This correction ran no new
training or simulation.

## Shared GPU continuation

The owner subsequently pointed out approximately 60 GB of available GPU memory
and requested concurrent execution. This authorizes sharing and supersedes the
earlier no-overlap preference, while keeping the collector running.
`eval_static_flow.py --allow-shared-gpu` bypasses the existing idle-device wait
and records the scheduling mode in `status.json`; the default remains the
idle-device wait. Only one evaluation case runs at a time. No collection process
or source configuration is modified. The two existing reproduction-guard tests
and Python compilation pass after the CLI change.

The remaining three cases were launched in a fresh output directory, preserving
the original completed and interrupted outputs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUDA_MPS_PIPE_DIRECTORY=/home/ge47gax/.mps_pipe \
CUDA_MPS_LOG_DIRECTORY=/home/ge47gax/.mps_log \
  /home/ge47gax/kun/genesis-world/.venv/bin/python -u \
  scripts/wang_transfer/eval_static_flow.py \
  --data output/anticipatory_dressing/flow_bc_v45_data \
  --flow-checkpoint output/anticipatory_dressing/flow_bc_v45_train/best.pt \
  --out output/anticipatory_dressing/flow_bc_v45_stage/ipc_shared_20260927 \
  --garments tshirt_68 tshirt_392 hospital_gown \
  --allow-shared-gpu --wall-budget 1800
```

Before launch, GPU 0 had 59,274 MiB free with 100% utilization. An early sample
with the evaluation client allocated reported 55,313 MiB free; all six collection
workers remained active and the v5 ledger contained 1,467 accepted entries.
The free memory accommodates this additional job, but compute contention may
increase wall times. Live samples are retained in
`flow_bc_v45_stage/shared_gpu_monitor.jsonl`. These are operational observations,
not a controlled throughput benchmark. This first continuation subsequently
exhausted its wall budget as recorded below.

### Results and remaining-case retry

The shared run completed `tshirt_68` / body 10040 in 433.01 s wall time.
FMVP reached the completion condition at decision 360 and flow at decision
295; both completed the 20-decision hold with valid grasps and no simulator
error. Maximum initial cloth difference was 3.34 mm, with identical initial
TCP positions. Along with the two original cases, this gives three selected
static starts completed by both methods, not a general success rate or a
controlled improvement claim.

The 1,800 s total wall budget expired during `tshirt_392`. Its last logged
decision was 621; flow had not reported success and the baseline slot had
already stopped without reported sleeve progress. There is no completed
metrics archive for this case. Wall-time interruption must not be equated
with either success or a fully evaluated policy failure. The gown case did
not start. The first shared status is now `error` / `TimeoutExpired`.

A fresh run at `flow_bc_v45_stage/ipc_shared_remaining_20260927` uses the same
checkpoint and dataset contract, `--garments hospital_gown tshirt_392`,
`--allow-shared-gpu`, and `--wall-budget 7200`. Putting the gown first prevents
the previously slow case from again consuming its opportunity to run. The
remaining command and MPS/thread settings are unchanged. All previous output
is retained. Six collection workers were verified active at this launch.

The accompanying raw snapshot is 810 v4 + 1,523 v5 accepted entries, totalling
2,333 entries / 732,617 commands. The model still uses the frozen audited
1,275-episode cache (1,079 training, 196 validation), not every collected entry.
No dynamic student or new-garment result follows from these static tests.

### Gown result from the remaining-case run

`hospital_gown` / body 10040 completed in 870.06 s wall time while sharing the
GPU. Flow reached the endpoint at decision 303 and completed the hold at 323,
with valid grasp and no simulator error. FMVP ended at decision 288 when its
grasp exceeded the tracking-validity bound; it had not reached the endpoint.
Maximum initial cloth mismatch is 7.91 mm and the initial TCP difference is
zero. This is a preliminary flow completion, not controlled evidence of
superiority. The four completed selected starts all succeed for flow; FMVP
succeeds on three. `tshirt_392` is still running, so the five-case result is
incomplete. These initial-state mismatches must be resolved before comparative
performance claims or a larger matched evaluation.
