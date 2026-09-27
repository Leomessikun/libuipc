# Continuation of Claude's static flow policy

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

