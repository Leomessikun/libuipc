# Review of recurrent dressing pretraining

The sequence infrastructure and the RLT implementation are useful, but the original H8 experiment
does not isolate the value of RLT. The primary risks are mismatched conditioning histories,
incomplete cost accounting and a pretraining objective that omitted the action causing its targets.
Fix those before a sustained learning comparison. No evidence yet shows that this architecture
improves dressing, and a larger memory alone does not solve insufficient successful exploration.

Audited baseline: `dc953a34`. Supporting records: [model audit](2026-09-13-rlt-model-audit.md),
[learning and cost audit](2026-09-13-rlt-training-audit.md), and
[primary-source research](2026-09-13-recurrent-rl-literature.md). The findings below separate the
baseline from the repairs in this change. No GPU policy training was launched.

## What is sound

The geometry encoder remains spatial; one decision supplies one temporal token. Recorded commands
advance critic histories, candidate actions retain their current-frame dense `dQ/da` path, and
online/target temporal parameters are separate. Replay stores episode identities and true terminal
successors. Rollout histories reset across physical discontinuities. Independent prefix-splice
checks support the implemented RLT branch computation, including gradients, on its supported masks.

These are meaningful correctness properties. They justify keeping the implementation. They do
not establish data efficiency, task success, or equivalence of the different histories actually
passed by collection and learning. The upstream report itself has no measured robotics results.[^1]

## Findings and repairs

| Priority | Baseline issue | Consequence | Current status |
|---|---|---|---|
| High | H-frame collection versus 1…H prefix losses; final target uses H+1 | Trains and bootstraps on histories different from those used to act | New RLT `endpoint` default; old `prefix` objective explicitly retained |
| High | Uniform endpoints plus all-prefix losses | Interior transitions participate up to H times, the final transition once | Endpoint mode learns one uniformly selected eligible transition |
| High | Next-state/reward heads omit current action | Mixed behavior produces action-marginal averages | Heads now condition on recorded current action |
| High | Cost measurement stops before first actor update | Published costs miss periodic work | Probe covers complete actor/target cadence cycles |
| Medium | Cloud count reports distinct frame slots | Hides repeated actor/critic/target encodings | Forward hooks count every spatial encoder input |
| Medium | Padded openings reduce actual loss count | Nominal B×H is not the measured denominator | Probe counts valid learning positions |
| Medium | Lost replay prefix is padded as a fresh episode | Fabricates a reset in mid-episode | Strict-context sampling rejects incomplete non-opening histories |
| Medium | Command regression on failures is called non-imitation | Random behavior may train an unhelpful action mean | Explicit optional command weight, zero by default |
| Medium | Loss sums feature dimensions and masks after arithmetic | Implicit task weighting; padded NaN labels contaminate loss | Per-target element means, named weights, selection before arithmetic |
| Medium | Streaming KV allocation assumes float32 | Other model dtypes fail on first step | Caches follow model dtype; float64/bfloat16 parity checked |
| Medium | Privileged recording is tied to privileged critic | Point/RLT runs cannot supply prediction targets | Opt-in `--record-privileged` records targets separately |

The trajectory heads remain small auxiliary predictors. Elementwise means remove dimension-count
bias, but do not normalize physical units. Dataset-derived target normalization and state-dependent
nonlinear dynamics heads remain appropriate future extensions; this is not a validated cloth world
model. Disabled command loss supplies diagnostics without behavior-cloning gradients.

## The exact finite-window reference

Let H denote the number of observed frames. At time t the declared policy reads observations
from t−H+1 through t and the H−1 intervening commands, with real episode openings left-padded.
The learner now selects a transition as an endpoint and supplies that same window. Its Bellman
successor drops the oldest frame and appends the saved successor and recorded command a_t.
Both current and successor windows have H frames. No H+1 continuation substitutes for the latter.

This matches the **declared finite-context policy**, not a proof that H observations make the
physical process Markov or that finite-context SAC is unbiased for every POMDP. Insufficient
history remains a modeling limitation.

`--history-kind rlt --rlt-learning-mode endpoint` is the new-run default. Use batch size 64 to
compare 64 endpoint losses with the B64 single-frame baseline. `prefix` preserves the previous
all-position objective and its different sampling/context semantics. Old RLT checkpoints lacking
the new field deserialize as `prefix`; protocol checks prevent silently resuming them as endpoint.
Switching objectives requires an explicit new experiment rather than an unnoticed resume change.

Strict-context sampling accepts a full H-frame prefix or a complete shorter prefix beginning at
episode step zero. Ring eviction, dropped rows and curriculum exclusion are not episode openings.
It fails explicitly if no eligible window exists. Legacy approximate padded sampling remains
available for the explicitly retained prefix objective and existing callers. Candidate pools are
maintained incrementally, rather than rescanning the replay capacity every update.

All-position learning is valuable, but not specific to RLT. Recovering its efficiency responsibly
requires a separate contract: full-episode memory with prefix replay, measured burn-in approximation,
or overlapping rolling windows with shared deterministic spatial work and deliberate sampling
weights. Simply supervising more prefixes cannot be described as the same rolling-H policy.

## Corrected cost evidence

The old probe warmed up once and measured two updates. With actor frequency four, neither measured
update trained the actor. Its 64/72 cloud figures described input slots, while the encoders run
multiple times. These figures cannot price complete SAC optimization.

The revised probe warms up and measures complete least-common-multiple actor/target cycles. A
bounded CPU rerun used production head widths, 40 points, four CPU threads, one warm-up cycle and
one measured four-update cycle. These are diagnostic timings under shared-machine load, not GPU
estimates or repeated benchmark confidence intervals. [Raw output](2026-09-13-rlt-cost-cpu.jsonl):

| Arm | Windows/update | Valid loss positions/update | Actual clouds/update | Mean update seconds |
|---|---:|---:|---:|---:|
| Single frame | 64 | 64 | 224 | 0.370 |
| Frames H4 | 64 | 64 | 896 | 1.745 |
| RLT H4 endpoint | 64 | 64 | 896 | 1.785 |
| RLT H8 endpoint | 64 | 64 | 1792 | 3.757 |
| RLT H8 prefix | 8 | 60.25 | 324 | 0.525 |

Endpoint H8 is a costly correctness reference in this implementation. Prefix H8 amortizes spatial
work, but that saving is coupled to a different objective. Do not present their ratio as the
architecture's isolated advantage. Eight windows also provide fewer independent histories than
64 transition samples, even if their nominal position counts match.

A fair training log should track environment transitions, optimizer steps, valid loss positions,
actor updates, independent episodes, and simulation/acting/update/evaluation time separately.
The historical 36 ms/update predates the dense critic and is not a price for these new models.

## The proposed streaming cache changes the policy

An episode-persistent `RLTState.step` retains information from before the most recent H frames.
Freezing weights does not remove that information from s or contextualized KV. Therefore replacing
rolling-H recomputation with persistent streaming is a model change. Dropping old KV entries
alone does not restore rolling-H semantics.

For exact rolling-H inference, cache only deterministic independent-frame actor features under
an unchanged encoder/preprocessing/precision contract, and rerun temporal computation over H.
Invalidate features when any producing dependency changes. This does not accelerate every critic
path: dense Q encodes `E_Q(o,a)`, with its own parameters and candidate action. Recorded-action,
candidate-action and target-critic features are different objects. A single actor feature column
cannot replace them while claiming the same dense critic.

The current streaming reference also allocates decoder caches to `max_len` and masks attention to
W; it does not physically retain only W−1 entries. Its run/branch API supports contiguous left
padding, not arbitrary packed-episode masks. These limitations should remain visible until a
separately tested cache implementation changes them.

## Which architecture is worth testing

Empirical recurrent RL gives stronger comparators than frame concatenation alone. Ni et al. show
that carefully configured recurrent model-free methods can be strong POMDP baselines; architecture
and context hyperparameters matter.[^2] Their later work distinguishes remembering an observation
from assigning credit to an action whose benefit arrives much later.[^3] AMAGO demonstrates a
parallel causal sequence approach, with deliberate gradient routing; its full algorithm cannot be
reduced to swapping an attention block.[^4] RESeL motivates testing smaller learning rates for the
temporal encoder rather than assuming all modules should share one rate.[^5]

Recommended comparison: dense single-frame, GRU with separate actor/critic state, small parallel
causal Transformer, and RLT under the same history and objective. Match observation information,
loss positions, data provenance and measured compute. GRU/Transformer alternatives are proposed,
not implemented by this repair. A prospective efficient split is an action-independent history
encoder plus current-frame dense `E_Q(o_t,a_t)`; it retains local action-conditioned geometry but
is an architectural ablation, not an exact cache optimization of the present model.

H8 at 0.1 seconds covers 0.7 seconds between oldest and newest observations. That may help infer
recent cloth motion or blocked commands. It does not by itself provide evidence for episode-scale
planning or better delayed credit assignment around the elbow.

## Data and curriculum have higher priority than a larger backbone

Our judgment is to invest next in successful-state visitation and reusable trajectories. The
running dense/plain run's recorded final evaluation at 125016 transitions has mean upper-arm
ratio 0.1502, zero successes and zero simulator errors. That is a valid poor result at this budget;
it does not establish asymptotic failure or identify missing memory as its cause. The old expert's
11/25 final successes still had zero reference-filter passes, so treating every expert action as
a policy target would import its defective path.

First collect a training-only corpus with observations, commands, true successors, reward components,
privileged targets, controller feedback availability, filter labels and full provenance. Store
successful prefixes, failed attempts and recoveries with distinct roles. The new target-recording
flag helps create this corpus without adding privileged actor inputs. It does not add the missing
offline optimization/checkpoint CLI or recover absent history from old flat snapshots.
(Status, later the same day: the offline command exists, see
[offline representation pretraining](2026-09-13-offline-pretraining.md); the corpus still does not.)

Use offline probes to distinguish hypotheses: current privileged-state estimation tests whether
history resolves hidden state; action-conditioned successor prediction tests whether it captures
motion consequences; selected command imitation tests only the quality of the selected behavior.
Split complete episodes by training body/garment/run, fit normalization on the training split, and
reserve poses 45–49 for final evaluation. Low prediction loss is not a deployment success result.

For elbow practice, Jump-Start RL motivates using a guide policy for a prefix and gradually
transferring more decisions to the learner.[^6] RFCL instead uses demonstration state resets and
then broadens the initial-state distribution.[^7] Neither paper validates IPC dressing. A local
roll-in is the easier first engineering option because it naturally advances anchors, controller
state and observation history. Its simulation cost must still be counted. Snapshot acceleration
needs matching engine, Python controller, RNG and policy history; restoring cloth positions alone
does not restore the episode.

The prerequisite is a usable, filter-valid approach to the elbow on training configurations.
Correct or search that approach first; do not optimize its parameters on the reserved held-out
cells. A roll-in ablation can then distinguish inability to reach the elbow from inability to
solve it once reached. If the latter remains, inspect local action feasibility and recovery data
before increasing memory depth. Final evaluation must always include the original start state.

## Decision and next gates

Keep the sequence and RLT code, use the repaired endpoint arm as a semantic reference, and retain
prefix mode only as a named different objective. Do not spend the next long run solely comparing
RLT H8 against a stateless run while pretraining data and context semantics differ.

The next defensible investment is a versioned trajectory corpus and offline trainer, followed by
a training-only elbow roll-in comparison and an empirically grounded GRU control. Only after that
choose whether episode memory, frozen actor features, higher update ratios or a shared temporal
encoder justify their complexity. This order tests the information/exploration bottleneck before
committing to a particular recurrent architecture. No change here establishes higher dressing
success or real-world force safety.

## Sources

[^1]: Yifan Zhang, [Recurrent Looped Transformer](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/blob/1bee93a9b01c21bea0c7a50ce3f6619f24731e19/Recurrent_Looped_Transformer.pdf), September 2026, sections 2–5.
[^2]: Ni, Eysenbach, Salakhutdinov, [Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://proceedings.mlr.press/v162/ni22a.html), ICML 2022.
[^3]: Ni et al., [When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment](https://arxiv.org/html/2307.03864v3), NeurIPS 2023.
[^4]: Grigsby, Fan, Zhu, [AMAGO](https://arxiv.org/html/2310.09971v3), ICLR 2024.
[^5]: Luo et al., [Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific Learning Rate](https://proceedings.neurips.cc/paper_files/paper/2024/file/5706668422bd0d82588998ebe1067133-Paper-Conference.pdf), NeurIPS 2024.
[^6]: Uchendu et al., [Jump-Start Reinforcement Learning](https://proceedings.mlr.press/v202/uchendu23a.html), ICML 2023.
[^7]: Tao et al., [Reverse Forward Curriculum Learning](https://arxiv.org/html/2405.03379v1), 2024. Original state-reset method and reported manipulation benchmarks; proposed local transfer is our inference.

## Validation record

The final full CPU suite passed 344 tests, with 14 CUDA-marked tests deselected (26.39 seconds).
Command: `PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m pytest python/uipc_manip/tests -o addopts='' -q -m 'not cuda'`.
GPU-marked tests and sustained policy training were not run. The known invalid-evaluation fixture
still emits a mean-of-empty-slice warning; this is unrelated to the repaired learning paths.
