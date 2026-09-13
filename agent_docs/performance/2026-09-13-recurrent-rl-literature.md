# Recurrent pretraining infrastructure: evidence and architecture decisions

> Historical audit of `dc953a34`. See the [final review and implementation status](2026-09-13-pretraining-infrastructure-review.md) for fixes and remaining limitations.

Scope: review of the RLT-H8 design reported at dc953a34, not a claim that any new dressing policy has been trained. The local RLT report and `rlt.py` pretraining heads were read; the companion training audit covers the full SAC/replay implementation. Sources below are original papers and author implementations. Recommendations are engineering deductions unless a paper is explicitly cited.

## Assessment

Sequence replay, a causal history interface, all-position learning, and tests of run/branch/step equivalence are useful infrastructure. They do not establish that RLT improves elbow passage. At H8 and 0.1 seconds per decision, the nominal observation window covers only 0.7 seconds between oldest and newest frame. That may help infer recent motion or repeated blocked commands, but is not an episode-scale reasoning capability. Neither memory nor pretraining creates reachable successful transitions where the environment, controller, or exploration distribution prevents them.

The most urgent problem is the memory contract. The reported collector uses a rolling H-frame policy; the learner applies losses to prefix lengths 1 through H. These are different policies at most learning positions. Describing the discrepancy is helpful, but leaving it in the primary experiment makes negative and positive results difficult to interpret. A synthetic equivalence test on one input window does not test equivalence of the windows actually supplied by collection and learning.

## What empirical recurrent RL research supports

**Recurrent SAC is a serious baseline.** Ni et al. (ICML 2022) compared 21 environments from six specialized methods and obtained competitive or better performance on 18. Their ablations favor separate recurrent actor and critic weights in the examined implementation; a shared encoder failed two diagnostic tasks. GRU versus LSTM and context length were task dependent: longer history was not uniformly better. The paper explicitly treats input selection, RL algorithm, and context length as consequential design decisions. This supports implementing a small GRU/LSTM arm using exactly the same replay and losses as RLT, rather than comparing RLT only against frame concatenation. It does not predict a gain in cloth manipulation.[1]

**Long memory and long credit assignment are different.** Ni et al. (NeurIPS 2023) show strong Transformer recall at long horizons but little advantage on the hardest long-credit-assignment tasks. Medium credit horizons can improve; the correct statement is not that attention can never aid credit assignment. For dressing, “infer whether the cloth moved after recent commands” is a memory hypothesis, while “take a temporarily unproductive lateral action that enables later elbow passage” is also an exploration/credit-assignment hypothesis. An H8 actor changes the first mechanism more directly than the second.[2]

**Context-specific optimization matters.** RESeL (NeurIPS 2024) studies parameter changes amplified through recurrent trajectories and uses a smaller context-encoder learning rate with ordinary head learning rates. Its evaluated full-trajectory recurrent SAC models improve stability across POMDP and locomotion tasks; its official implementation includes GRU, Mamba, and causal Transformer alternatives. The published values include context LR 1e-5 versus head LR 3e-4, but these are comparison candidates, not validated dressing defaults. Separate optimizer groups and state-drift logging are better infrastructure than a single unquestioned LR for all new temporal parameters.[3]

**Off-policy Transformer RL already has stronger comparators than a report-only architecture.** AMAGO (ICLR 2024) trains a parallel causal trajectory Transformer with actor/critic losses at each position. It deliberately redesigns gradient routing to share one sequence representation and adds normalization/attention-stability measures. Its paper warns that train/test context mismatch creates out-of-distribution behavior; a naive shared-Transformer ablation collapses on several benchmark environments. We should borrow its sequence-format and optimization lessons, not transplant its entire actor/critic update into dense spatial SAC without an explicit experiment. AMAGO's improvements combine an algorithm and an architecture; they are not evidence that replacing a GRU with any Transformer suffices.[4]

**Gating alone is not validation.** GTrXL established that layer-normalization placement and gates can make Transformer RL much more stable than a canonical Transformer. Its principal benchmarks and optimization setting differ from this off-policy dressing code. RLT having a gated merge does not automatically inherit GTrXL's measured stability.[5]

## Three valid memory contracts

| Contract | Collector | Learner | Benefit and limitation |
|---|---|---|---|
| Exact rolling window | Recompute temporal model on the last H independent frame features | Each supervised position is supplied its own identical H-frame history, including true short histories at reset | Cleanest short-memory comparison; overlapping windows can share frame encoding, but recurrent temporal outputs cannot generally be shared exactly |
| Episode memory | Carry a recurrent state from the real episode start | Rebuild the prefix under current parameters; optionally backpropagate only through a suffix | Matches the intended recurrent policy; prefix compute and parameter changes must be budgeted |
| Approximate burn-in | Carry episode state at collection | Rebuild from a stored/zero approximate state through B burn-in positions, then learn U positions | Efficient approximation; measure discrepancy versus full-prefix reconstruction and state age |

R2D2 addresses representational drift and stale recurrent state in replay. The official DeepMind Acme learner reconstructs online and target states separately during burn-in and excludes burn-in positions from the loss. This is a useful implementation reference for boundaries and target state handling, but it is discrete Q-learning and does not directly supply the continuous SAC gradient contract.[6]

Adding a burn-in prefix does not make a rolling-H collector exact: a recurrent learner that carries state through B+U observations gives later supervised positions more than H history. Conversely, adding `RLTState.step` at collection does not preserve the rolling-H model even with frozen weights. Retained hidden states and KV entries already contain earlier information. Dropping old KV entries does not erase that influence. Choose and name the contract before optimizing caches.

For a quick reliable experiment, exact rolling H8 with last-position loss is an acceptable costlier reference. It checks whether context can help at all. To retain all-position efficiency, a proper episode/prefix or explicitly approximate burn-in arm is preferable. Benchmark the approximation against a small full-prefix reference rather than calling it exact replay.

## Spatial and temporal architecture

Maintain the existing dense critic as the reference. A useful alternate decomposition is:

`z_t = temporal(E_history(o_t), a_(t-1), observable_controller_state)`

`actor = policy(z_t, E_actor(o_t))`

`Q_j = head_j(z_t^Q, E_dense_Q(o_t, candidate_a_t))`.

The temporal state summarizes recorded history independently of the candidate action. The candidate still enters current-frame point features before spatial aggregation, retaining a direct dQ/da path. This avoids making the entire history representation depend on alternative current actions, and permits batching current candidates against a shared history. It is an architectural alternative, not a claim of mathematical equivalence to the current recurrent dense critic. Compare against the current branch implementation using identical data and context.

Start with separate actor and critic temporal modules. Sharing can reduce cost, but Ni's negative result and AMAGO's carefully modified positive result show that sharing requires its own gradient-routing decision. A later AMAGO-inspired shared temporal encoder should be a separate arm, with tests showing which actor, critic, and target paths are detached.

Do not jump from H8 to a large full-episode RLT solely because the source uses “infinite depth.” A small causal Transformer can train all positions in parallel; GRU offers a simpler recurrent-state contract. RLT's sequential decoder may be useful, but it needs to earn its additional latency relative to these comparators.

## Feature caching: what is actually exact

A cache `E(o)` is exact only while its complete producing function is unchanged: encoder parameters, running normalization statistics, observation preprocessing, point sampling, augmentation, precision contract, and any appended controller features. Use a fingerprint of these dependencies plus observation/schema identifiers. Retain raw observations so a cache can be regenerated. A version mismatch should invalidate or recompute, never silently consume stale features.

An actor feature cache cannot substitute for the dense critic's `E_Q(o,a)`: the critic has different parameters and candidate action enters before pooling. A recorded-action dense feature can at most accelerate a recorded prefix under that exact frozen critic. Candidate-action features must be recomputed; target-critic features require their own parameter identity. Freezing one spatial encoder does not freeze the recurrent states generated above it.

Freezing pretrained perception is an empirically valid arm: R3M uses frozen representations successfully for downstream manipulation, and MVP reports frozen visual pretraining across real-world tasks. Their RGB encoders, data, and policy settings differ substantially from a task-trained PointNet++ dense critic. They justify an experiment, not the assertion that a frozen low-dimensional cloth feature retains all contact-sensitive geometry.[7][8]

A safer early collection optimization is a deterministic independent-frame cache for the actor, while continuing to rerun the temporal H-window. This preserves rolling-window semantics. If actor weights change each update, rebuild the small cache when the encoder changes, or collect a short chunk under fixed actor weights and refresh between chunks; the chosen lag becomes a logged training parameter. Episode-state streaming must be treated as a policy change.

## Pretraining objectives need a causal split

The inspected `TrajectoryPretrainingHead` predicts command, next privileged state, and reward from the same pre-action history state. The command is unknown at that point. When identical history is followed by different exploratory commands, the next-state and reward targets differ. Squared error without action conditioning learns their behavior-policy average. It can still be a representation objective, but it is not an action-conditioned dynamics model and cannot distinguish the consequences of candidate commands.

Use separate roles:

1. A state-estimation head `g(h_t) -> normalized privileged state_t` to assess whether memory recovers currently hidden information.
2. An action-conditioned transition head `f(h_t, recorded_a_t) -> delta_priv, reward, terminal` trained on both successes and failures.
3. An optional behavior head `p(recorded_a_t | h_t)` for action modeling. With mixed failed/random behavior, it should not be described as a policy-improvement target. Do not automatically copy this head into the SAC actor or keep its loss active during RL.

The current losses sum errors across feature coordinates then average only over valid positions. A 35-dimensional privileged target can dominate a scalar reward through dimension count and physical units. Record target normalization on the training split, normalize per target dimension, and expose weights. On deterministic bang-bang demonstrations, bounded action regression is a reasonable baseline; on multimodal or random behavior, the conditional mean can represent an action never chosen in useful demonstrations. Action likelihood or a mixture is an experiment, not a requirement to build an entire diffusion policy.

Pretraining should answer measurable questions before a costly RL run: does memory reduce held-out privileged-state error versus a single frame? Does conditioning on action improve successor prediction versus an action-blind baseline? Can predicted progress distinguish advance from blocked movement? Compare trajectories by held-out garment/body/run, not randomly split overlapping windows. Keep unreliable per-step force labels out of the initial objective.

## Data and fair evaluation

The expensive resource is simulation. Collect reusable complete episodes containing observation, requested action, observable controller state, privileged targets, reward components, terminal versus truncation, simulation validity, and provenance. Include successful expert prefixes and recoveries, ordinary rollouts, and failures with explicit labels. Do not require every comparison arm to first generate an incompatible private dataset. A versioned offline corpus permits pretraining comparisons before online data distributions diverge.

Log at least four budgets: environment transitions; optimizer steps per transition; valid loss positions processed per transition; and end-to-end wall time split into simulation, collection, replay, update, and evaluation. Eight windows times eight positions equals 64 scalar loss positions, not 64 independent histories. Padding changes the actual count; overlapping sampled windows can repeat transitions. Record unique episodes and unique transitions per update. Keep actor/target update schedules explicit when changing U or batch size.

Suggested first matrix: dense single-frame; exact rolling H8 GRU; exact rolling H8 RLT; small parallel causal Transformer under the same history contract. Match 64 valid learning positions when possible, retain enough independently sampled episodes, and report wall-time as well as transition-matched results. Next compare from-scratch versus the same pretraining corpus for the best two temporal arms. Do not mix a new reward, a new critic, full-episode memory, freezing, and pretraining into the first comparison.

A successful infrastructure outcome is a trustworthy ability to falsify the memory hypothesis. If sequence models improve state estimation but fail online, investigate exploration, return propagation, controller blockage, and reachable elbow recovery data. If they do not improve estimation or action ranking, increasing model complexity has weak justification. CPU unit tests and a cost probe establish mechanics and cost; they cannot establish task-level benefit.

## Sources

1. Ni, Eysenbach, Salakhutdinov. Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs. ICML 2022. https://proceedings.mlr.press/v162/ni22a/ni22a.pdf ; author code https://github.com/twni2016/pomdp-baselines
2. Ni, Ma, Eysenbach, Bacon. When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment. NeurIPS 2023. https://arxiv.org/html/2307.03864v3 ; author code https://github.com/twni2016/Memory-RL
3. Luo, Tu, Huang, Yu. Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific Learning Rate. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/5706668422bd0d82588998ebe1067133-Paper-Conference.pdf ; author code https://github.com/FanmingL/Recurrent-Offpolicy-RL
4. Grigsby, Fan, Zhu. AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents. ICLR 2024. https://arxiv.org/html/2310.09971v3 ; author code https://github.com/UT-Austin-RPL/amago
5. Parisotto et al. Stabilizing Transformers for Reinforcement Learning. ICML 2020. https://proceedings.mlr.press/v119/parisotto20a.html
6. Kapturowski et al. Recurrent Experience Replay in Distributed Reinforcement Learning. ICLR 2019. https://openreview.net/forum?id=r1lyTjAqYX ; official Acme learner, especially lines 78–108: https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/learning.py . Paper endpoint was browser-blocked during this audit; detailed burn-in statements above are grounded in the official code, not an unverified paper excerpt.
7. Nair et al. R3M: A Universal Visual Representation for Robot Manipulation. 2022. https://arxiv.org/abs/2203.12601
8. Radosavovic et al. Real-World Robot Learning with Masked Visual Pre-training. 2022. https://arxiv.org/abs/2210.03109
