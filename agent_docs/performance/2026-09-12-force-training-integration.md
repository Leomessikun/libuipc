# Force-training integration and elbow bottleneck audit

> **Status update, 2026-09-13:** This is a historical audit/proposal against `4bfe88c2`, not the current implementation status. Subsequent commit `024c5716` closed the proposed per-decision force-training direction after the reliability gate failed; see the [closure](2026-09-12-research-direction.md) and [measurements](2026-09-12-contact-force-calibration.md). No force-training result is established here. The dense action-per-point critic was subsequently implemented in `ef1b3c81`, with optional residual trunks in `01bf913e`. Any future experiment must use those changes as an explicit baseline and revalidate its labels. Literature findings remain reference material; implementation recommendations below are conditional.

Read-only audit, 2026-09-12, repository HEAD `4bfe88c23e723f9394c04a6d54fc424b6834dd28`. Read `CLAUDE.md`, `agent_docs/rule.md`, architecture index and current Python sources. No repository edits, training, or GPU jobs. Paths below are relative to `/home/ge47gax/kun/libuipc`. Findings distinguish source-code facts, recorded experiments, and proposed tests; they do not establish clinical safety.

## What the existing results establish

The environment can get garments beyond the elbow, and a learned policy has done so on some held-out cells. The present evidence does not support either “IPC makes the task impossible” or “the policy has never learned anything.” It also does not establish a reliable deployable dressing skill.

Actual run `output/uipc_manip/wang_teacher_r13_s1/config.json` saves **discount 0.995**, reward scale **0.5**, horizon **300 decisions**, action repeat **6**, dt **1/60 s**, speed cap **0.15 m/s**, resolved friction **0.3**, cuff strength **10000**, and tether **0.06 m**. The trainer-argument dictionary contains friction 0.6, but the resolved environment contains 0.3: use the resolved environment for comparisons. There is no evidence here for discount 0.25. `sac.py:35` gives 0.995 at horizon 300; effective geometric horizon is 200 decisions, or 20 s at this decision duration.

Direct CSV inspection of `output/uipc_manip/wang_teacher_r13_s1/eval_log.csv` gives 27 evaluation rows, only 13 without simulator errors:

| Admitted transitions | Mean final upper-arm ratio | Success | Paper filter | Interpretation |
|---:|---:|---:|---:|---|
| 125016 | 0.282891 | 1/25 | 1/25 | Highest mean among error-free rounds |
| 174264 | 0.211065 | 3/25 | 0/25 | Highest success among error-free rounds |
| 241056 | 0.367047 | 0/25 | 0/25 | **25 simulator errors**, not a valid complete-episode comparison |
| 251568 | 0.090313 | 0/25 | 0/25 | Error-free later degradation |
| 265584 | 0.0 | 0/25 | 0/25 | **25 simulator errors**, not proof that the actor intrinsically cannot dress |

Documentation saying the best valid round was “at 265.6k” conflates total training reached with the earlier best evaluation. The three successes at 174.3k do not pass the early-turn filter: the raw CSV reports zero paper-filter rate. Preserve these distinctions in any user explanation.

The current expert artifact is `output/uipc_manip/expert_r13_heldout_s0/{manifest,records}.json`, manifest filesystem mtime 2026-09-12 15:27:23 local (mtime, not an embedded immutable experiment timestamp). Region 13, bodies 14045–14049, five garments, 25 full 300-decision episodes, no simulator errors:

| Garment | Final success | Mean final upper | Mean peak upper |
|---|---:|---:|---:|
| hospital_gown | 3/5 | 0.7563 | 0.8124 |
| tshirt_26 | 2/5 | 0.4111 | 0.8105 |
| tshirt_68 | 3/5 | 0.5893 | 0.8247 |
| tshirt_4 | 3/5 | 0.6611 | 0.8610 |
| tshirt_392 | 0/5 | 0.2914 | 0.2927 |

Overall: 11/25 success, mean final 0.541834, mean peak 0.720246; 17 episodes finish in `done`, 8 in `elbow_hook`; 24/25 trigger `early_turn`, including all successes, giving zero paper-filter yield. That classifier is a geometric trajectory proxy, not a direct measurement of tissue collision or injury. Likewise `final_threaded=0` alone is not a validated physical failure: its winding test uses the single finger-to-shoulder axis on a bent limb.

The expert manifest serializes `base_cfg` with `decision_watchdog=True`, while `expert_baseline.py:237` constructs actual worlds with `decision_watchdog=False`; `:262` saves the base configuration. This is a concrete provenance inconsistency, not evidence that this expert run tripped the watchdog.

## Why the elbow remains a bottleneck

**Established implementation mechanisms:**

- `dressing_env.py:604` scales/clips six action coordinates. X rotation is suppressed by default. `:622` drops translations whose requested anchor approaches arm points within 12 mm; `:627` can drop both rotation and translation under the 60 mm held-vertex tether. The learner currently receives the requested action in replay, but there is no explicit rejected-action diagnostic or actor input saying which mechanism blocked it. Requested actions in replay are correct for this environment's action semantics; pretending they were executed motions in a force predictor would be wrong.
- `dressing_heuristic.py:133` explicitly adds upward motion during `elbow_hook` because near-arm no-move rejection stalls the expert there. This is source evidence for an escape maneuver, not proof every learned failure has that cause. The script uses opening geometry and irreversible stages that the point actor does not receive.
- `dressing_reward.py:124` initializes task reward to negative fingertip-to-opening distance; it switches to forearm progress after a ray hit and to forearm length plus five times upper-arm distance after an upper-arm hit. Forearm chooses the first valid triangle, upper arm the minimum valid intersection. These discrete selections can change sharply as the cuff folds/rotates. The reward is absolute progress each decision, not progress improvement; remaining on the forearm earns positive reward. A temporary loss of intersection while navigating the elbow can therefore be unattractive. This is a plausible learning barrier requiring trace analysis, not a demonstrated numerical bug.
- `dressing_reward.py:167` adds only a 0.01 binary cuff proximity penalty and upper-arm center alignment. No force, early-turn, exposure, or tissue-deformation term enters the reward. `early_turn` is logged by `dressing_env.py:674`, accumulated for evaluation in `train_sac.py:533`, and filters collected BC episodes; it does not terminate training.
- `dressing_heuristic.py:144` now stops its final stage on progress/drop, addressing its former target past the shoulder. This cannot prevent subsequent cloth relaxation. The measured peak-to-final decline is large; do not call every final zero an elbow failure without seeing the trajectory.
- `dressing_env.py:688` packs a fresh segmented cloud relative to **commanded** anchor plus shoulder offset, absolute anchor and attached=True. There is no force, q/dq, previous action, velocity, grip lag or history in the current actor. `models.py:490` uses a feedforward tool-point feature. A geometry observation can alias different hidden snag/strain/friction states. The static-arm observation mode also assumes a prior arm scan, not continual unoccluded visibility.

**Hypotheses to discriminate:** (1) cloth/grasp/placement makes a specific outer-elbow path infeasible; (2) clipping/tether suppresses the escape action; (3) the reward dip suppresses discovering an available path; (4) partial observations cannot identify the needed correction; (5) actor/critic instability loses a previously learned skill; (6) watchdog censoring removes difficult contact-rich transitions. These can coexist. A force penalty alone does not solve missing progress demonstrations and may make forearm avoidance more attractive.

## Exact insertion points for force learning

### 1. Measurement before changing the objective

`contact_force.py:98` (`vertex_forces`) reads normal/friction gradient channels, negates them, divides by actual simulation dt squared, and sums repeated global-vertex indices. `geometry_vertex_block` supplies body-specific offsets. `contact_export_status` distinguishes missing channels from a zero measurement and reports `last_assembled_iterate`; its convergence fields are diagnostic, not a final-state-force guarantee.

Select each **body** block to avoid cloth self-contact contamination. This identifies forces on that body, not arbitrary pair identity: if other objects contact the same body, their loads also enter. A future cloth-body-only reporter would need contact-pair attribution or an explicit scene assumption. Missing friction channels must remain unavailable rather than a zero shear target. Static weight calibration establishes dimensional scaling, not dynamic accuracy of cached last-Newton-iterate gradients.

Read inside `dressing_env.py:617` after each successful `_sim_step`, before any subsequent step/reset. At repeat six, reading once per decision loses internal peaks. Retain per-decision peak, time-integrated cost, availability and sample count. Define whether a cost is mean dimensionless excess, integral of excess in seconds, impulse in N s, or normalized patch load; multiplying a mean by dt again silently changes the objective. Body net force, sum of nodal norms and largest nodal force are different N quantities. Pressure additionally requires a defensible area in m², yielding Pa; nodal peaks are mesh dependent. Do not apply a published wrist stop threshold directly to a node or summed-body quantity.

### 2. Minimal fixed force cost in the environment reward

For a first controlled experiment, preserve `wang_progress` as the task metric and combine it with a **fixed, versioned** normal-load cost after measurement in `dressing_env.step`. Add a nested force-cost config with default weight zero, explicit normalizer/aggregation and validity policy. Expose task reward, cost, combined reward and measurement status separately in `infos` and evaluation.

Both insertion paths already multiply raw reward exactly once: `train_sac.py:893` and `pretrain_wang.py:683`. Combine task and force cost **before** that shared multiplier. `sac.py:298` clips scalar Bellman targets using reward_abs_bound/(1-gamma), default 400 at gamma .995. Large added penalties can collapse targets to that floor; record saturation and choose a normalized objective. FlashSAC instead projects onto min/max value atoms (`sac.py:313`), so support must also cover the modified return distribution. Start with scalar SAC for a clean test.

`replay.py:33` stores only the final scalar reward; old buffers cannot be relabeled with a new force weight because force labels were never retained. Minimal safe protocol: load the working actor as an explicit warm start, use fresh/recollected replay with one fixed objective, and record the fork separately from exact resume. The current trainer resume path is not a general warm-start API.

### 3. Curriculum with replay consistency

Changing lambda at collection time while replay stores only scalar reward mixes different objectives. For an actual force curriculum, extend `FlatReplayBuffer`/`ReplaySet` save/load/sample with raw task reward, calibrated force cost and validity/version fields, then recompute `(task_reward - current_lambda * cost) * reward_scale` consistently at sampling time. Do not reconstruct forces from point observations. Integrate new batch fields explicitly: `sac.py:394` currently parses positional tuples ending in optional privileged pairs, region label and replay-buffer index.

Checkpoint/replay metadata currently validates dimensions and scale (`train_sac.py:230`; `pretrain_wang.py:489`) but has no force schema/calibration/aggregation fingerprint. Add these and schedule progress to the protocol. `pretrain_wang.py:367` curriculum counts admitted transitions; `train_sac.py:869` counts vector steps. They are not interchangeable force-schedule counters.

Preferred staged experiment: first acquire repeatable elbow progress, then fine-tune with a low calibrated force cost; admit harder garments separately. A hard safety terminal is a larger change than a soft cost (see below).

### 4. Privileged critic, without changing deployed observations

Already supported: `--critic-input privileged`; `dressing_privileged.py:22` defines a 35-float layout, env `:553` builds it, and both replay paths preserve current/next state. `sac.py:290` uses it only for Q; `:342` still obtains actor actions from point observations. This is the smallest architecture option: append a small fixed, normalized contact summary to the privileged state and update its layout/protocol. At a horizon reset, preserve **terminal** force state together with `terminal_privileged`, not the fresh snapshot's load.

A privileged **actor** is not implemented by this flag. The September11 proposal explicitly calls for one but the current actor remains point based. Extending actor inputs to oracle forces, then deploying with zeros or guessed values, would introduce a sensor mismatch. Increasing privileged_dim rejects old critic/replay dimensions; actor-only transfer needs an explicit loader and fresh compatible critic initialization.

### 5. Auxiliary predictor and the two student paths

Lowest-risk predictor experiment: collect a separate supervised dataset and train a detached head on actor encoder features plus action/history, with force targets as labels only. Current obs/actions NPZ collection (`collect_rollouts.py:88`) lacks force, next-state and substep labels; `distill.py:80` consumes only obs/actions. Add an explicit dataset schema rather than assuming legacy rollouts can train a force predictor. Split by episode and held-out body/garment; evaluate missed high-load events and uncertainty, not only mean MSE.

An action-conditional predictor must condition on the requested action and relevant controller state; labels belong to the following decision/substeps. A recurrent predictor needs sequence/episode boundaries; random independent replay rows cannot supply valid history. Joint auxiliary gradients into the actor encoder are a second ablation because they can harm control features. Shared modules must have clear optimizer ownership.

Wang online distillation (`sac.py:350`) calls each frozen regional teacher on the **same observation tuple** and matches distribution parameters. Privileged-actor teachers would require a separate teacher-input path. Existing force-aware critics do not change the teacher actor signature. FMVP-style BC (`distill.py:117`) matches stored actions only; adding safe-force episode filters does not itself teach a force model. The current expert's zero paper-filter yield makes “just distill expert successes” empty under the existing filter. Its irreversible stage machine should not be used as an arbitrary off-trajectory labeler without adaptation.

## Terminal, reset and failure requirements

`dressing_env.py:647` ends only on horizon and returns true terminal observation/state before auto-reset (`:681`). Both trainers deliberately store done=False to bootstrap time truncations; **that is correct for the current time-limit semantics**. If force threshold termination is introduced, propagate distinct terminated/truncated fields and set the bootstrap mask for genuine absorbing failure. `pretrain_wang.py:721` assumes all slots finish together; per-slot safety resets violate this and need an explicit whole-world or asynchronous design.

On simulator error the env resets and returns zero reward. Current `train_sac.py:865` aborts before insertion; `pretrain_wang.py:661` drops all transitions for that vector step and rebuilds. The earlier reset-to-new-episode bootstrap bug is therefore not present in these paths. Do not turn unavailable/invalid force into zero cost and retain the row. Force accumulators and predictor history must reset even when a world is rebuilt, a failed step exits early, or evaluation temporarily uses another world.

The current pipeline checkpoint (`pretrain_wang.py:574`) replaces replay_latest before publishing state.json and does not preserve full physics/RNG trajectory state. Do not describe a reward-objective fork or crash recovery as exact continuation. This read-only audit does not apply earlier detached-worktree patches.

## Diagnostic matrix before a broad training launch

| Priority | Bounded comparison | Evidence to retain | What it decides |
|---|---|---|---|
| 0 | Re-evaluate one previously successful checkpoint, same resolved physics/cells, valid complete episodes | errors, final/peak upper, paper filter, policy checksum | Whether apparent zero was evaluation failure or actual degradation |
| 1 | Expert vs actor on representative elbow-fail and elbow-success cells | every-substep requested/accepted anchor and rotation, no-move/tether reason, opening frame, held lag, stage | Controller rejection vs unavailable/undiscovered trajectory |
| 2 | Trace task reward around first elbow crossing and subsequent loss | forearm/upper ray hits and selected triangle, center penalty, ring geometry, reward dip, peak/final ratio | Reward discontinuity, relaxation or geometric snag |
| 3 | Body-block force readout replayed on the same trajectories | channel availability, solver state, calibrated patch loads, local velocity, substep peaks | Whether usable force labels explain snag/stall |
| 4 | Existing point critic vs expanded privileged critic, fixed reward and budget | held-out progress, action-rejection rate, Q saturation, alpha, force tails | Value-state information benefit without actor sensor gap |
| 5 | Successful actor warm start: zero force cost vs small fixed calibrated cost | final success plus force/exposure tails and elbow crossing time | Whether force optimization improves contact behavior without losing the skill |
| 6 | Only after 5: reweighted replay curriculum and history predictor | matching reward protocol, label validity, OOD error and missed events | Whether added complexity is supported |

Do not simultaneously change dt, grasp, body erosion, horizon, reward, camera and algorithm. Existing timestep studies already show elbow sensitivity, so matching the physics is essential. A no-FT deployed actor can learn lower **simulated** contact cost through reward and privileged critic; absolute real load or safety claims additionally require real calibration and independent observability/monitoring evidence.

## Evidence extraction and exact documentation corrections

Raw training evidence: `output/uipc_manip/wang_teacher_r13_s1/eval_log.csv`, parsed with Python standard-library `csv.DictReader`; valid means integer column `sim_errors == 0`. Compared columns `transitions`, `mean_final_upperarm_ratio`, `success_rate`, `paper_filter_rate`; equivalent `heldout_*` fields cover the same 25 cells. Do not maximize across failed rows. Expert aggregate independently read from `manifest.json["summary"]` and per-cell `records.json`: `success`, `final_stage`, `early_turn`, `paper_filter`, `final_upperarm_ratio`, `max_upperarm_ratio`, `sim_error`, `length`, `garment`, `human`. No numerical simulation was rerun.

In `agent_docs/performance/2026-09-11-training-infrastructure-proposal.md:181`, relabel the teacher column as “best valid mean round at 125016 transitions, out of a run reaching 265584.” At `:203`, preserve the distinction that the three successes at 174.3k did not pass the filter: CSV reports `paper_filter_rate=0`. At `:216`, do not say expert peak is 0.293 on all five tshirt_392 cells: **mean** peak is 0.292670, while body14045 final alone is 0.504349. A diagnostic classifier flag should be described as early-turn geometry, not asserted as independently measured human collision.
