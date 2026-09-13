# Offline representation pretraining from sequence replay

Date: 2026-09-13. Status: implemented and CPU-tested on `pretrain/recurrent-force-memory`; run on
a toy corpus only. Continues the [infrastructure review](2026-09-13-pretraining-infrastructure-review.md),
whose item 2 — "an offline pretraining command, to verify first whether history improves state
estimation and action-consequence prediction" — this is. The receiving half,
`SACAgent.initialize_representation`, was left uncommitted by that review's author and is committed
here with the sending half.

## What exists now

`python -m uipc_manip.pretrain_offline --replay SNAPSHOT [...] --out RUN [network flags]` trains the
actor's representation on recorded episodes and writes a checkpoint a fresh online run adopts with
`--init-representation RUN/checkpoints/pretrain_best.pt` (both trainers).

| Piece | Where | What it fixes |
|---|---|---|
| Snapshot reader | `pretrain_offline.read_snapshot` | A `replay_latest` directory (one buffer) or a `replay_set.json` directory (one buffer per garment); flat snapshots are refused, they carry no episodes |
| Episode split | `split_episodes`, `fill_buffers` | Whole episodes go to training or validation by a seeded shuffle; rows are re-added chronologically into two sequence buffers; the same stream id from two runs becomes two streams, so episodes of different runs never link |
| Targets | `target_statistics`, `Representation.losses` | Mean and standard deviation of the successor privileged state and the reward, fitted on the training rows only, stored in the checkpoint; the head predicts normalised targets |
| State function | `Representation.states` | `--history-length 1`: the frame latent alone; `--history-kind rlt`: the recurrent readout at every recorded position; the parameter-free frame concatenation has no per-position state and is refused |
| Objective | `rlt.TrajectoryPretrainingHead` (review's action-conditioned form) | From `[state_t ; a_t]` predict the next privileged state and the reward; command regression off by default |
| Sampling | `sample_sequences(pad=True, strict_context=True)` | The review's contract: a full window, or a genuine episode opening; never an evicted prefix |
| Baseline | `val_priv_baseline`, `val_reward_baseline` | The constant predictor's loss on the same validation windows, in the same normalised units — the number "does the representation predict anything" is read against |
| Learning rates | `--lr`, `--history-lr` | The recurrent history gets its own rate (the review's RESeL note) |
| Checkpoint | `checkpoints/pretrain_best.pt`, `pretrain_final.pt` | `actor` and `head` weights, `sac_config`, `protocol`, and `metadata.pretraining`: sources with sha256, split, normalisation, weights, steps, validation losses |
| Transfer | `SACAgent.initialize_representation` | Fresh agents only (no update, no optimiser state); the protocol must match apart from `rlt_learning_mode`, which is an online setting; copies `actor.encoder` and, when present, `actor.history`; the policy trunk, critic, temperature and optimisers stay the new run's; returns provenance that lands in the checkpoint metadata under `representation_init` |
| Trainer flag | `--init-representation` in `train_sac`/`pretrain_wang` | Refused with `--resume` before any file is read; a resumed Wang run replays its saved command line and skips the initialisation, its weights come from its checkpoint |

Log: `pretrain_log.csv` with per-step training losses and, at every `--eval-every`, the validation
losses and baselines over the same `--eval-batches` windows.

## The diagnostic it is for

Two runs on the same snapshots, the same split seed and the same steps:

```bash
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.pretrain_offline \
  --replay RUN/checkpoints/replay_latest --out output/uipc_manip/pre_single --history-length 1 \
  --obs-mode wang_static_arm --steps 20000 --seed 1
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.pretrain_offline \
  --replay RUN/checkpoints/replay_latest --out output/uipc_manip/pre_rlt8 --history-length 8 --history-kind rlt \
  --obs-mode wang_static_arm --steps 20000 --seed 1
```

Read `val_next_priv` against `val_priv_baseline` in both: if the single-frame predictor already
sits far below the baseline and the recurrent one is not lower, history does not resolve hidden
state on these episodes and the elbow question is elsewhere (reaching it, or recovering there); if
the recurrent one is lower, the online arm with `--init-representation` has a reason to exist. A
low loss is a prediction result, not a dressing result, and the head is a small linear predictor
on normalised targets, not a cloth model.

## What does not exist

- **A corpus.** No finished run recorded sequence identities, and none recorded `priv` outside the
  privileged critic; the trainer was exercised on a toy snapshot only. The first usable corpus is
  the first run launched with `--sequence-replay --record-privileged`. Which run that is — an
  ablation arm carrying the flags, or a dedicated collection — is a launch decision, not made here.
- **Corpus roles.** The review asks for successful prefixes, failed attempts and recoveries with
  distinct roles, filter labels and controller feedback; the snapshot carries none of these labels.
  The reader takes what the replay has.
- **An online result.** No run has started from a pretrained representation.

## Validation

`tests/test_pretrain_offline.py` (5): both the single-frame and the RLT trainer run end to end on a
toy corpus, split episodes (not rows) and keep validation out of the statistics, log and save; a
fresh agent adopts the encoder (and history) exactly with its trunk untouched, in either online
learning mode; the transfer refuses a different width, an online checkpoint and an agent that has
updated; the trainer refuses flat snapshots, the frame concatenation and a missing privileged
target unless its weight is zero; two runs' episodes stay apart. Stub tests in
`tests/test_train_sac.py` and `tests/test_pretrain_wang.py` cover the flag on fresh runs, its
refusal with `--resume`, and the skip on a Wang resume. Full CPU suite: see the handoff entry.
