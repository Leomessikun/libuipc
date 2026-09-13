# Episode-aware replay for dressing pretraining

The first implementation stage of the [recurrent pretraining proposal](2026-09-13-recurrent-pretraining-proposal.md)
records continuous interaction histories without changing the current actor, critic or SAC loss.
`train_sac` and the regional teacher/student launcher accept `--sequence-replay` (default: disabled).
Enabling it records additional metadata; the existing learner still samples independent transitions.
H4/H8 policies, GRU/RLT models and sequence losses are subsequent work, not features of this change.

## Collection and boundaries

`ReplayStreams` allocates episode IDs above the replay snapshot's persisted high-water mark and
tracks each vector slot's physical decision index. It advances excluded curriculum/held-out slots
too, so a later admitted row cannot hide a missing step. Episode IDs change on auto-reset,
evaluation reset and world rotation. Reusing a slot for another garment does not reuse its history.

Normal horizon boundaries set `episode_end=True` while retaining `not_done=1` and the true
pre-reset successor observation. Simulator-error transitions remain excluded. Before rebuilding
a failed world, collection closes the last valid row without changing its reward or Bellman mask.
The generic trainer also closes retained history before evaluation interrupts the training world.
Resume creates new identities because the previous physics is not restored by loading replay.

Commands are stored in the existing action space. This stage does not replace them with measured
cloth motion, add force labels, or change observations. An ended trajectory and an absorbing
terminal state remain separate concepts.

## Replay interface

Both `FlatReplayBuffer` and `ReplaySet` accept `sequence=True`. In that mode every insertion requires
keyword-only `stream_id`, `episode_id`, `episode_step` and boolean `episode_end`. IDs and steps are
nonnegative integers; steps must increase within a retained stream/episode. Bellman `done=True`
requires episode end, while episode end alone does not require `done=True`.

`sample_sequences(length, batch_size=None)` returns an immutable named `SequenceBatch`:

| Field | Shape and meaning |
|---|---|
| `obs` | `[B,L+1,D]`, L current observations plus the final saved successor |
| `actions` | `[B,L,A]`, recorded policy commands |
| `rewards`, `not_dones` | `[B,L,1]`, existing scaled rewards and bootstrap masks |
| `episode_ends` | `[B,L]`, independent boundary flags |
| `stream_ids`, `episode_ids`, `episode_steps` | `[B,L]`, validated temporal identity |
| `priv` | Optional `[B,L+1,P]`, including the matching final privileged successor |
| `labels` | Optional `[B,L]`, retaining regional teacher labels |
| `buffer_index` | Optional integer, the selected ReplaySet buffer/temperature index |

Only complete windows are returned; there is no padding or burn-in policy yet. Windows cannot
cross streams, episodes, step gaps or episode ends. A window may end at a boundary and include
that transition's pre-reset successor. Sampling uses replacement and fails explicitly when no
complete window exists. ReplaySet first selects uniformly among buffers with valid windows, then
samples the entire batch from that buffer.

Linked rows survive circular storage order changes. Candidate endpoints are built once per
requested length and updated as rows arrive or are overwritten; repeated sampling does not scan
the entire capacity. This is an implementation property, not a measured throughput claim. No
stacked H-frame observations are persisted, so adding history does not multiply point-cloud storage.

## Persistence and compatibility

Sequence snapshots carry schema version 1, identity arrays and a monotonic next-episode-ID mark.
Loading reconstructs links from retained chronological rows, including when the new capacity is
smaller. Overwritten history is unavailable; it is never filled from unrelated rows. Compressed
metadata arrays are materialized once before link reconstruction.

Legacy flat sampling and snapshots remain supported with recording disabled. Flat and sequence
snapshots are deliberately not interchangeable: missing episode IDs cannot be inferred reliably.
The generic trainer persists the flag in its training settings; the Wang launcher preserves it
through its existing saved command line. A fresh run may select either mode; resuming a replay
requires matching its mode.

To record sequence-capable data during a new existing-policy teacher run, add the flag to the
normal command, for example:

```bash
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.pretrain_wang teacher \
  --region 13 --sequence-replay --run-name r13_sequence_recording
```

This command launches training when run; it was not launched for this change. The already running
dense/plain ablation continues separately and does not acquire sequence metadata retroactively.

## Validation

CPU coverage exercises interleaved slots, gaps, episode ends versus bootstrap, stale circular
indices, candidate-cache maintenance, optional privileged states and labels, ReplaySet selection,
schema rejection, save/load/shrinking, collector exclusion/reset and fresh IDs on resume. A stub
simulation drives the real Wang training loop through a failed episode, world rotations and
checkpoint resume. These checks validate data semantics; they do not establish GPU simulation
performance or improved dressing success.
