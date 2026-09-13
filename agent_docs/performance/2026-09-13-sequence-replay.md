# Episode-aware replay for dressing pretraining

The first implementation stage of the [recurrent pretraining proposal](2026-09-13-recurrent-pretraining-proposal.md)
records continuous interaction histories without changing the current actor, critic or SAC loss.
`train_sac` and the regional teacher/student launcher accept `--sequence-replay` (default: disabled).
Enabling it records additional metadata; the existing learner still samples independent transitions.
Padded windows, the matching streaming rollout state and the stage 2 ordered frame history were
added afterwards; `--history-length H` above 1 switches both trainers to it and requires the flag.
GRU/RLT models and burn-in sequence losses are subsequent work.

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

`sample_sequences(length, batch_size=None, *, pad=False)` returns an immutable named `SequenceBatch`:

| Field | Shape and meaning |
|---|---|
| `obs` | `[B,L+1,D]`, L current observations plus the final saved successor |
| `actions` | `[B,L,A]`, recorded policy commands |
| `rewards`, `not_dones` | `[B,L,1]`, existing scaled rewards and bootstrap masks |
| `episode_ends` | `[B,L]`, independent boundary flags |
| `stream_ids`, `episode_ids`, `episode_steps` | `[B,L]`, validated temporal identity, `-1` where padded |
| `valid` | `[B,L]`, the recorded transitions; padded positions are zero and carry no identity |
| `priv` | Optional `[B,L+1,P]`, including the matching final privileged successor |
| `labels` | Optional `[B,L]`, retaining regional teacher labels |
| `buffer_index` | Optional integer, the selected ReplaySet buffer/temperature index |

Windows cannot cross streams, episodes, step gaps or episode ends. A window may end at a boundary
and include that transition's pre-reset successor. Sampling draws endpoints uniformly with
replacement; ReplaySet first selects uniformly among buffers that can supply a window, then samples
the entire batch from that buffer.

`pad=False` returns only complete windows, and fails explicitly when none exists. `pad=True` also
draws windows whose episode began fewer than `L` steps earlier and left-pads them, so every
recorded transition is reachable as a learning step. Without it, the first `L-1` decisions of every
episode are never trained on, while a deployed policy has to act through exactly those steps with an
empty history. Padded positions carry zero observations, commands, rewards and bootstrap masks,
identity `-1`, and `valid=False`; a retained prefix is contiguous and right-aligned, so padding
occurs only at the left and never reaches into an earlier episode. Padding does not change the
windows that already existed. Consumers must normalise losses by the valid count and must not let a
padded position update a recurrent state. Endpoint-uniform sampling means a transition within `L-1`
steps of the end of its retained episode appears in fewer windows; this is the usual bias of
sequence replay, recorded rather than corrected.

There is no burn-in policy yet: burn-in is a property of the recurrent learner, not of the sampler.

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

## Streaming rollout state

`RolloutHistory` in `history.py` is the collection-side counterpart of a padded window. It keeps the
raw prefix — the last `L-1` observations and commands per stream — not encoded features. Raw frames
are the reason a weight update cannot leave the state stale: every decision re-encodes the retained
prefix under the current parameters, so streaming and window evaluation are comparable by
construction rather than by a version tag. The price is `L` encoder passes per decision, which the
GPU smoke of a later stage has to measure; nothing here prices it.

`window(obs)` returns `obs[N,L,D]`, `actions[N,L-1,A]` and `valid[N,L]`: the current observation
always occupies the last position, and a command is valid exactly where its observation is. `push`
records the decision taken. `reset` forgets a stream's prefix and must be called on every physical
discontinuity — episode end, world rebuild, slot reassignment, resume into fresh physics — because a
prefix from different physics is not a shorter history but a wrong one.

`length=1` is the single-frame policy: an all-valid window holding only the current observation and
no command. It is the parity setting against which the history-aware model must reproduce the
current feedforward baseline.

## Running the finite-history policy

`--history-length H` sets `SACConfig.history_length`. At 1, the default, nothing changes: no rollout
state is created and the agent is called exactly as before. Above 1 the actor and the dense critic
encode each of the last `H` frames on its own and read the ordered frame vectors, their validity
and the commands between them (`models.FrameHistory`); learning draws padded windows of length `H`
from sequence replay and trains one step per window, its last transition; and every collector —
the training world, each evaluation world, `expert_baseline`, `collect_rollouts` — owns a
`RolloutHistory` that it empties on episode end, rotation, evaluation, simulator error and resume.
Warm-up commands enter the history like policy commands. Each decision costs `H` encoder passes;
that price is unmeasured here. The successor window of a Bellman target advances the history with
the command actually recorded, never with a fresh candidate.

The H4 arm of the proposal's comparison against the running dense/plain ablation is:

```bash
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.pretrain_wang teacher \
  --region 13 --transitions 125000 --seed 1 --checkpoint-every 25000 \
  --obs-mode wang_static_arm --no-obs-augment --critic-action-mode dense --trunk-style plain \
  --sequence-replay --history-length 4 --run-name abl_h4_s1
```

It was not launched: the GPU is held by the critic ablation, and the proposal gates this arm on
that ablation's valid held-out result. Refused before a world is allocated: flashsac, the privileged
and latent critics, teacher distillation onto a history-aware student, stochastic augmentation, a
history checkpoint given to `distill`, and `--history-length` without `--sequence-replay`. Earlier
checkpoints carry no `history_length` protocol key and keep loading as single-frame policies.

## Validation

CPU coverage exercises interleaved slots, gaps, episode ends versus bootstrap, stale circular
indices, candidate-cache maintenance, optional privileged states and labels, ReplaySet selection,
schema rejection, save/load/shrinking, collector exclusion/reset and fresh IDs on resume. It also
covers padded coverage of every transition, zeroed and identity-free padding, the refusal to pad
across an episode boundary, and equality between padded and complete windows. A stub simulation
drives the real Wang training loop through a failed episode, world rotations and checkpoint resume.

The parity gates are explicit: for `L` in 1, 2, 4 and 7, stepping `RolloutHistory` through a
recorded episode reproduces the sampler's padded window at every decision, mask included; at
`history_length=1` the heads keep their parameter names and outputs, the default agent acts, updates,
saves and loads exactly as before, and the trainers never create a state. Model tests cover masked
padding, that the oldest frame and command are read, and that the candidate action touches only
the current frame while keeping its gradient. Stub runs of both trainers drive an H2 policy through
warm-up, episode ends, rotation, evaluation and resume, checking the state empties at each. These
checks validate semantics; they do not establish GPU cost, throughput or improved dressing success.
