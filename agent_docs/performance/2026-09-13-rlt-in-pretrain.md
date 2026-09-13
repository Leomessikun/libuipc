# The recurrent looped transformer in the pretraining infrastructure

> **Superseded assessment:** The [independent review](2026-09-13-pretraining-infrastructure-review.md) found collection/learning context mismatch, biased position weighting, incomplete optimizer-cost measurement, and action-blind prediction targets. Endpoint learning is now the new-run RLT default; prefix remains explicit. The old timing table and proposed single feature-cache equivalence below must not be used as current conclusions.

Date: 2026-09-13. Status: implemented and CPU-tested on `pretrain/recurrent-force-memory`; cost
measured on CPU only; no GPU run. This answers, concretely, how Zhang's *Recurrent Looped
Transformer* (RLT; upstream `1bee93a9`, report only) is used in our pretraining, after the
[proposal](2026-09-13-recurrent-pretraining-proposal.md) had deferred it to a stage 4. It is an
arm now; the H4 frame history stays a control, not a gate.

Tags: [MI] measured here, [RA] read from the report, [E] inferred.

## In one sentence

**A token is one decision; the RLT sits in the history slot of the actor and the dense critic,
learns at every recorded position of a replay window for about the spatial-encoding price of one
single-frame transition per position [MI, CPU], and can be pretrained on every recorded trajectory
— failed ones included — by predicting the next command, the next privileged state and the reward
[RA 5.1, built]; its exact-replay contract [RA 5.3–5.4] is what the windowed update implements,
and extending it to whole 300-decision episodes needs a cached spatial feature column that does
not exist yet.**

## The mapping, equation by equation

| RLT (report) | Here (`python/uipc_manip/rlt.py`, `models.py`, `sac.py`) |
|---|---|
| token `x_t`, BOS | one decision: the frame's spatial encoding (PointNet++ global vector, or the Wang tool-point row, ⊕ `extra`) and, for the actor, the command `a_{t-1}` that led to it; reset is the empty history |
| causal encoder `E_θ` (2.1), memory `M_≤t` (2.2) | `EncoderLayer` × `layers` under a causal mask; `Memory` projects the encoder output into `groups` key/value groups with a rotary positional transformation |
| `H_0 = (s*, ∅)` (2.3) | learned `s_star`; a padded window starts from it at its first recorded frame |
| gated merge (2.9–2.11) | `Merge`: `r = RMSNorm(s_{t-1})`, `g = σ(W_g[e;r]+b)`, `u = e + α g ⊙ W_s r` |
| decoder block: SWA over decoder activations within `W` (2.12–2.14) → cross-attention to `M_≤t` (2.15) → FFN (2.16); `s_t = z_t^{L_D}` | `DecoderLayer`, in that order; current KV formed from the layer input before SWA; the last `W−1` positions retained |
| tied configuration (2.6) | `RLTConfig.tied`: decoder SWA and FFN reuse the encoder layer's modules; norms, cross-attention and memory projections stay separate |
| readout softmax (2.6) | actor: the existing squashed-Gaussian trunk on `RMSNorm_o(s_t)`; critic: the twin-Q trunks on it, with the candidate action still a per-point feature of the *current* frame only |
| prefill / incremental decoding must agree (2.4) | three schedules of one computation — `run` (window), `branch` (alternative token at every position against the recorded prefix), `step` (streaming with `RLTState`) — held equal at every position by tests, through left padding and per-stream resets |
| batch across sequences, preserve the recurrence within each (4.1) | `branch`: K candidates at position `t` are K parallel final transitions from the one recorded `H_{t-1}` |
| autoregressive pretraining (5.1), SFT masks without detaching (5.2) | `TrajectoryPretrainingHead` + `pretraining_step`: next command (squared error; the expert is bang-bang), next privileged state, reward, averaged over recorded positions, state never detached |
| RL replay rebuilds encoder KV, `s_t`, decoder SWA KV under current Θ (5.3); states stale after any optimizer step (5.4) | `SACAgent._update_rlt`: every update re-runs the recorded window from its first frame under the current parameters; the actor loss re-runs the critic's recorded pass after the critic step rather than reusing pre-step states |
| truncated BPTT must declare its detached paths (5.4) | the window start is the declared truncation; nothing inside the window is detached |

What is *not* transplanted: the language-model importance ratios of 5.3. SAC is off-policy and
the scripted expert has no behaviour density; the proposal already declined this.

## What one update does

`--history-kind rlt --history-length H` (both trainers; needs `--sequence-replay`) draws
`batch_size` padded windows of `H` transitions plus the saved successor. For every window:

1. the actor encodes the `H+1` frames, runs the RLT over `[frame_t ; a_{t-1}]`, and samples a
   candidate `π_t` at every position;
2. the target critic runs the recorded pass (frame `t` encoded with its recorded command `a_t`) and
   *branches* `π_{t+1}` at every position: `Q_targ(h_{t+1}, π_{t+1})` for all `t`, from the recorded
   `s_t`, with the successor advanced by the recorded command, never a fresh one;
3. the online critic's recorded pass gives `Q(h_t, a_t)` at every position; the squared Bellman
   error is averaged over the `H` recorded positions (padding neither learns nor dilutes);
4. after the critic step, the actor loss branches `π_t` against the critic's fresh recorded pass;
   `dQ/dπ_t` flows through frame `t`'s spatial encoding exactly as in the single-frame dense critic;
5. the target critic's temporal parameters are soft-updated with the encoder's `τ`
   (the frame history had none to track).

Compared with `--history-kind frames`, which spends `4H` spatial passes on one learning step per
window, this spends about `6(H+1)` on `H` learning steps.

## Cost, measured on CPU

`python -m uipc_manip.rlt_cost_probe --device cpu --points 200`, 8 threads, torch 2.12, production
widths (`d = 64`, 2+2 layers, `W = 8`, 240,768 temporal parameters; PointNet++ at the reference
widths on 200-point clouds) [MI]. Indicative ratios; the GPU numbers must be measured when it frees.

| | Learning positions | Clouds encoded per update | Update | Per learning position |
|---|---|---|---|---|
| single-frame | 64 | 64 | 1.25 s | 19.6 ms |
| `frames`, H4 | 64 | 256 | 5.60 s | 87.5 ms |
| `rlt`, H4 | 64 (16 windows) | 80 | 2.01 s | 31.3 ms |
| `rlt`, H8 | 64 (8 windows) | 72 | 1.99 s | 31.2 ms |

The temporal stack alone (no clouds), forward under `no_grad`:

| Window | B = 1 `run` | B = 64 `run` | B = 64 `branch` | B = 64 `run` + backward |
|---|---|---|---|---|
| 8 | 3.7 ms | 6.5 ms | 3.0 ms | 22 ms |
| 32 | 12 ms | 24 ms | 6.6 ms | 87 ms |
| 300 | 119 ms | 275 ms | 128 ms | 985 ms |

One decision for 25 streams through the acting path, which re-encodes the raw window each decision:

| Arm | Clouds encoded per decision | Per decision |
|---|---|---|
| single-frame | 25 | 97 ms |
| `frames`, H4 / `rlt`, H4 | 100 | 479 / 483 ms |
| `rlt`, H8 | 200 | 1,030 ms |

Streaming `step` of the temporal stack alone at B = 1: 0.70 ms per decision. What these say: at `H ≤ 16` the temporal stack is
a rounding error next to the point encoders, and the per-position cost sits between single-frame
and the frame history's; at `L = 300` the decoder recurrence is `300 × layers` small sequential
steps — latency-bound on a GPU (about 9,000 kernel launches per `run`) unless captured in a CUDA
graph — and the *spatial* side, 300 clouds per window, is what the exact-replay path cannot pay
without a feature cache. Collection scales the same way — H8 acting encodes eight clouds per
stream per decision, which the cached `RLTState` would reduce to one plus a 0.7 ms temporal step —
so the GPU probe has to price both sides before the arm. On the measured GPU update (36 ms per single-frame transition, batch 64
[`2026-09-12-wall-clock-budget.md`]), the H8 arm is expected at roughly 1.6× per learning
position [E from the CPU ratio], which the probe on `--device cuda` has to confirm.

## Three decisions, recorded as decisions

1. **Frozen spatial encoder for the exact-replay path.** Whole-episode replay is affordable only
   if the 300 clouds of a window are not re-encoded each update: the spatial encoder trains in the
   pretraining stage and is frozen during RL, so a stored feature is an exact prefix checkpoint
   under 5.4. This is the same bet as DICE-RL's frozen visual features that
   [the residual-RL record](2026-09-13-prior-and-residual-rl.md) flagged, and the latent
   arrangement the reference measured 0.11 worse for the *critic*. It is therefore an arm — frozen
   versus slowly unfrozen encoder, dense critic kept in both — not a default. Not built.
2. **Learning positions.** Within `H ≤ 16` every recorded position learns (built). A 300-position
   window with the dense critic would need one spatial pass per learning position for the
   candidate, so the exact-replay path learns at a sampled subset of 8–16 positions per window; the
   recorded pass stays complete. Not built; a mask on `learn` in `_update_rlt` when the cache exists.
3. **Where cached spatial features live.** A `features [capacity, d]` column in
   `FlatReplayBuffer` in sequence mode, written at collection (the collector already encodes the
   frame to act), gathered by the same linked-window indices, stamped with an encoder version so a
   thawed encoder invalidates it; `act` then carries an `RLTState` instead of re-encoding an
   `H`-frame prefix. Not built; it is the next replay change.

## A declared mismatch of the windowed arm

At collection the policy acts from a sliding window of the last `H` frames; at learning, position
`t` of a window is conditioned on the window's prefix, `t` frames long. The two coincide only at
episode openings and for the window's last position. The frame history does not have this
mismatch (it learns only the last position), the exact-replay path removes it (the window is the
episode). If the H8 arm loses to H4, this is the first suspect, before the temporal model itself.

## Data for the pretraining stage, and the missing trainer

The objective is a tested function (`TrajectoryPretrainingHead`, `pretraining_step`); there is no
command yet that iterates a replay snapshot, optimizes the actor's spatial encoder and RLT with it
and saves weights `SACAgent.load` accepts. That trainer is the next code change of this stage.

Data: today there is none: no finished run recorded sequence identities and the expert episodes hold no
observations. The replay snapshot of any `--sequence-replay` run carries observations, commands,
rewards and boundaries, so the H4/RLT arms below double as the recording. The privileged-state
target needs one more change: the sequence replay stores `priv` only under the privileged critic,
which the history refuses; recording it unconditionally in sequence mode is a small `pretrain_wang`
change, not made here.

## Commands

The RLT-H8 arm, matched to `abl_dense_s1` (region 13, seed 1, dense/plain, no augmentation) and to
the H4 command in the [replay record](2026-09-13-sequence-replay.md); the batch is 8 windows so
each update learns 64 positions like the others:

```bash
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.pretrain_wang teacher \
  --region 13 --transitions 125000 --seed 1 --checkpoint-every 25000 \
  --obs-mode wang_static_arm --no-obs-augment --critic-action-mode dense --trunk-style plain \
  --sequence-replay --history-length 8 --history-kind rlt --batch-size 8 --run-name abl_rlt8_s1
```

The GPU cost probe, before the arm:

```bash
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.rlt_cost_probe --device cuda --points 400
```

Neither was launched: the GPU holds `abl_dense_s1` (115k of 125k transitions at the time of
writing, held-out upper-arm ratio 0.10). Read the H8 arm's batch with care: 8 windows × 8
positions are cost-matched to the 64 transitions of the other arms, not statistically equivalent
to them — positions within a window are correlated and 8 trajectories per gradient step is few.
If the arm loses, the batch is the second suspect after the window mismatch above, before the
model.

## Validation

`tests/test_rlt.py` (20): streaming `step` equals the window `run` at every position for `W` in
{1, 3, 8}, shared and per-layer memory groups, tied and untied; a padded row equals the shorter
sequence and its first recorded position equals a fresh stream's first state; padding content is
unreadable; a branch with the recorded token reproduces the recorded state, and an alternative
token at `t` reproduces the spliced sequence's state at `t` without touching the run; gradients
reach the first token and `s*`; streams of different ages advance together and reset individually;
the pretraining objective ignores padding and fits a predictable sequence; a pretraining step runs
a real actor on a sequence batch. `tests/test_history_models.py` (+3): the acting path reads the
last state of the same run learning uses; `Q` of the acting path, the recorded pass and a branch
with the recorded command agree, a different candidate at `t` changes `Q_t` only, and `dQ/da`
exists at every recorded position; the frame history refuses per-position learning.
`tests/test_sac_agent.py`: the H4 end-to-end test now runs for both kinds through both encoders,
checks that every recorded position learned and that the target's temporal parameters moved; the
`rlt` kind at `history_length 1` and an unknown kind are refused at construction.
`tests/test_train_sac.py`: the `--rlt-*` knobs reach the config, and checkpoints without the keys
default to the frame history. 323 CPU tests pass. None of this measures dressing success.

## What this does not claim

No measurement shows the RLT policy dresses the elbow, and the CPU ratios are not GPU throughput.
The claim is narrower: the report's method is now in the pretraining infrastructure end to end —
model, per-position SAC update under its replay contract, pretraining objective, trainer flags —
with what is missing named precisely — the pretraining trainer command, recorded sequence data,
the spatial feature cache and frozen-encoder arm for whole-episode replay, `priv` in sequence
replay — rather than deferred.
