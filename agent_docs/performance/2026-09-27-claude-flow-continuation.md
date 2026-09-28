# Continuation of Claude's static flow policy

## 2026-09-28 17:03 CEST: completed static evaluation and live garment transfer

This update supersedes the running-status statements in the dated entries
below. The owner excluded MaxRL and retained the garment-transfer and
dynamic-arm plan. The evidence snapshot is
`output/anticipatory_dressing/status_review_20260928_150255/report.json`, with
hashed copies of the original ledgers, training index, logs and motion status.
It is a ledger/log review, not a new simulation or geometry validation.

| Evaluation population | Started units completed by all three policies | FMVP | r1 | Flow |
| --- | ---: | ---: | ---: | ---: |
| Five existing garments, completed | 201 | 100 | 130 | 138 |
| ClothesNet, interim | 116 | 35 | 42 | 40 |

The static grid has 205 planned units and four without a legal start. The
ClothesNet snapshot has 378 policy rows, 119 complete triples, three of those
without a legal start, and 16 garments represented among the 116 started
triples. Lost grasps remain failures in the denominator. There are no duplicate
result keys. Pending units are not failures or completed trials. This is not
the final 20-garment, 14-body result, and its aggregate cannot establish a
policy ranking. Previously measured nonidentical settled cloth and the static
body selection limitations remain. A garment that all policies fail is not
automatically an invalid asset.

The v5 manifest has 2,477 accepted records: tshirt_26 428, tshirt_68 489,
tshirt_4 660, tshirt_392 570, and hospital_gown 330. Three collection workers
remain active under dispatcher 550064. ClothesNet coordinators 3485169 and
1056477 have three and two active workers respectively. Their output logs
continue to advance. The frozen flow training set remains 1,807 episodes;
these additional records are not automatically consumed by that checkpoint.

The training-budget control completed:

| Training body fraction | Updates | Best logged validation first-action MSE sum |
| --- | ---: | ---: |
| 25 percent | 30,000 | 0.0185 |
| 50 percent | 30,000 | 0.0185 |
| 100 percent | 30,000 | 0.0191 |
| 100 percent | 60,000 | 0.0173 |

The 60k result comes from `claude_flow_h16_60k_20260927/train.log`; evaluation
continues to use the frozen 30k checkpoint. These are sampled offline errors,
with one seed and non-nested fractional subsets. They do not establish a data
ceiling or an improvement in dressing success. Evaluate selected models on
validation before a separate final test; retain the existing test baseline.

Next retain the 20 ClothesNet test garments, prepare independent training and
validation garment manifests, and compare the same flow model trained with
and without garment diversity. Dynamic experiments can proceed on a statically
successful garment while that work continues; they need not wait for success
on every garment. The motion pilot record specifies the next comparison.

## Update at 19:05 CEST: five paired garment batches and a training-budget control

The first seven body/pose IDs have completed both policies on all five garments:

| Garment | r1 accepted | Flow accepted |
| --- | ---: | ---: |
| tshirt_26 | 4/7 | 5/7 |
| tshirt_68 | 6/7 | 6/7 |
| tshirt_4 | 6/7 | 6/7 |
| tshirt_392 | 7/7 | 7/7 |
| hospital_gown | 6/7 | 6/7 |
| Paired total | 29/35 | 30/35 |

The next tshirt_26 r1 batch also finished (2/7); its flow batch is not complete
at this snapshot, so it is excluded from the paired total. There are 77/410
reported trial outcomes. The initial geometry audit passes saved acceptance
evidence and body-split checks, but all 35 completed pairs exceed its strict
10-micrometre identical-start tolerance. Maximum cloth differences are
0.035--0.139 mm for tshirt_26, 2.564--4.355 mm for tshirt_68,
1.463--2.629 mm for tshirt_4, 0.948--3.037 mm for tshirt_392, and
1.616--4.795 mm for hospital_gown.
Human vertices and TCP starts match. One additional flow success on this
small subset is not evidence of superiority. The long-sleeve successes use
different starts, body IDs and a different flow model from the older failed
five-case diagnostic; do not treat them as a controlled fix of that failure.

To test the owner's training-duration question, a new 60,000-update run was
started at `output/anticipatory_dressing/claude_flow_h16_60k_20260927`.
It uses a byte-identical copied index and copies of all 16 feature shards:
the same 1,807 training episodes / 570,717 training states, 255 validation
episodes, architecture, force input, batch 1,024, seed zero and AdamW settings
as the 30,000-update run. No newly collected trajectories are added to this
control. The trainer source is copied; `run.json` records commands and hashes
for the index, shards, trainer, policy definition and r1 encoder checkpoint.
The existing trainer is loaded with the original external package path.

This starts from initialization, because the old checkpoint contains weights
without optimizer state. Cosine learning-rate decay spans the new 60,000
updates. It tests an increased training budget with its corresponding
schedule, not an exact continuation of the old optimization trajectory.
The 410-case evaluator retains its frozen 30,000-update checkpoint. Validate
the new model on the fixed validation split before using test outcomes for
claims; no improvement from additional training has been measured yet.
The live log reached 4,000 updates without an execution error.

Collection continues. The latest v5 ledger at 18:55 CEST had 1,728 accepted
records; new collection data is separate from both frozen training runs.

## Scope and provenance

The owner requested inspecting and continuing Claude's work on 2026-09-27.
The original workspace is on `research/expo-ft-dressing` at `49fcfd56`;
the policy implementation is in its `residual-rl` worktree on `sac-stability`
at `e124657c`. Both were inspected, along with the relevant recent Claude
session, live processes, training log and checkpoint. Preserve its active
collection and evaluation. The continuation adds a CPU result audit on
`research/anticipatory-dressing`; it does not launch another training or
evaluation run over the same cases.

Relevant existing changes:

- `a9a67ccd`: `build_dataset_index.py`, one index across accepted v4/v5 runs.
- `49fcfd56`: `train_flow_policy.py`, plus native flow action execution in
  `collect_garment.py`.
- `e124657c` in the policy package: `flow_policy.py` and bridge dispatch.
- The original workspace also has an untracked `eval_policies_batched.py`;
  it is actively running and is preserved as found.

Output evidence is in
`output/anticipatory_dressing/claude_flow_continuation_20260927/`:
`provenance.json`, `source_snapshot/`, `evaluation_audit.json`,
`audit_monitor.json` and `audit_watch.log`. Source files, dataset index and
completed training log were copied and hashed after training completed.
The checkpoint records the index path but no training-time index hash; this
later snapshot must not be represented as proving immutable training inputs.

## Dataset and completed training

The original dataset directory is
`/home/ge47gax/kun/libuipc/output/uipc_manip/dressing_dataset_20260927`.
Its frozen index contains 2,361 accepted episodes (810 v4 and 1,551 v5),
741,201 commands, and the five existing Cloth3D garments.

| Split | Episodes | Commands | Body/pose IDs |
| --- | ---: | ---: | ---: |
| Train | 1,807 | 570,717 | 301 |
| Validation | 255 | 78,917 | 41 |
| Test | 299 | 91,567 | 41 |

The new flow head uses the frozen r1 PointNet++ encoder: 250 pooled features
and three normalized gripper-force components. It predicts a 16-action
chunk with three translation components and one vertical rotation component.
Serving integrates ten Euler steps, executes the first action and replans.
It receives the current observation, without temporal observation history
or human-motion prediction. Training uses 30,000 updates, batch 1,024,
width 512, depth four, and seed zero. Validation uses sampled first-action
MSE, with a changing subset/noise, for checkpoint selection.

Training completed. Selected update 30,000 has first-action MSE
`[0.0000649620, 0.0016926380, 0.0094555914, 0.0078743827]` in the
normalized model-action coordinates. The same log's mean-action baseline
is approximately `[0.0377, 0.0152, 0.0491, 0.0618]`. These are offline
imitation errors, not dressing success rates or realized tracking errors.
The copied evaluation checkpoint is `flow_r1_h16/flow_policy_eval.pt`, SHA256
`9b0c3feca5ad83d7c180655e0416cf265d226fcaa2738c8252e6b1eaad302b97`.

This is different from the earlier random-encoder, geometry/history-only
flow BC on 1,275 episodes documented in `2026-09-27-static-flow-v45.md`.
Do not attribute that model's four accepted diagnostic cases to this model.

## Active evaluation and additional audit

Claude's evaluator started after training completed, at
`/home/ge47gax/kun/libuipc/output/uipc_manip/policy_eval_test_20260927`.
It schedules 41 test body/pose IDs x five garments x two policies: 410
planned results, initially 60 batched jobs, four workers, alongside the
six-worker v5 collector. It compares r1 with the frozen flow checkpoint.
The flow receives gripper force and native actions; r1 retains collection
inputs and action processing. Both use armhole-aligned starts and the
interior-sleeve/armhole endpoint followed by the external 20-decision hold.
There were no completed results at this handoff; early rollout logs advance.

`scripts/wang_transfer/audit_batched_policy_eval.py` reads the declared grid
and saved results without controlling any workers. It:

- Keeps planned, pending, attempted, failed and no-legal-start counts separate.
- Flags duplicate rows, unexpected units, body-split overlap, missing archives,
  and inconsistent checkpoint/ledger/archive fingerprints.
- Checks finite initial states, aligned command/endpoint arrays, and the
  recorded success state's complete, valid, zero-command hold. This checks
  saved evidence, not an independent recomputation of contact/sleeve geometry.
- Compares paired initial cloth, human vertices, TCP and observations;
  checks common simulation settings and records differences explicitly.
  The existing 10-micrometre diagnostic tolerance labels geometric matches;
  equal seeds alone do not establish identical initial states.

Seven CPU regressions pass: pending denominator, displaced starts, a falsely
accepted hold, asymmetric legal starts, duplicate/leaked bodies, partial
JSONL writes, changed seed, and non-finite states are covered (some tests
cover multiple conditions). The real run initially reports 0/410 results,
no errors. A CPU watcher refreshes the report every 30 seconds for up to
four hours; its PID and exact command are in `audit_monitor.json`.

To refresh once:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/ge47gax/kun/genesis-world/.venv/bin/python \
  scripts/wang_transfer/audit_batched_policy_eval.py \
  --run /home/ge47gax/kun/libuipc/output/uipc_manip/policy_eval_test_20260927 \
  --out output/anticipatory_dressing/claude_flow_continuation_20260927/evaluation_audit.json
```

## Interpretation and next work

The test body set was selected from bodies with at least one accepted source
recording. Crossing these bodies with all five garments avoids selecting
only successful garment/body pairs, but does not create an unselected
population benchmark. All five garments occur in training. The r1 encoder
may have seen these bodies in its earlier training; this is a split for the
new flow head. There is no ClothesNet holdout result.

First finish and audit the current evaluation, retaining failures and
initial-state mismatches. Compare per-garment outcomes and matched eligible
pairs with the declared denominator. Do not infer an algorithmic benefit
from changing architecture, force input and training data simultaneously.

The simulated-force input differs from the proposed geometry-only research
student and needs a matched no-force ablation before adopting this baseline
for that claim. Neither this checkpoint nor GRAB retargeting implements
anticipatory dressing. The existing pass-motion pilot remains a separate
running diagnostic, with static competence as its gate. Dynamic data,
causal prediction, recovery supervision and garment holdouts remain open.
