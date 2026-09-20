# Which action-improvement experiments already exist?

The owner asked to inspect completed tests before proposing them again, and
authorized missing evaluations. This stage reuses the saved RAL branches and
adds a bounded comparison of two ways of extracting an action from the **same
frozen critic**. It does not train a new controller or modify production SAC.

Starting revision: `c22b6034`, branch `research/flow-latent-steering`. Tracked
files were clean; unrelated untracked build logs/worktrees were preserved.

## Completed work that should not be repeated

| Question | Existing evidence | What it does and does not answer |
|---|---|---|
| Does an IPC derivative or executed finite-difference correction improve SAC? | [Actor audit](2026-09-17-dressing-actor-audit.md): six proposal types, including learned-Q gradient and random, each pass 0/6 preset state gates; 1,794 decisions / 404.92 s | Already tested. No robust local benefit established; not evidence that all gradient estimators fail. |
| Can derivative-free sequence search do better? | [Parallel CEM](2026-09-17-parallel-dressing-trajopt.md): independent continuation coverage .57704 versus SAC .57259, all controls 0/4 sustained-valid successes; all stages 4,956 decisions / 1,025.53 s | Already tested. The small mean difference does not establish robust improvement. These were 72-decision continuations, not full episodes. |
| Do fixed recovery macros or state-dependent macro choices help? | [Recovery review](2026-09-18-recovery-decisions-review.md): held-out-repeat state selection adds .002266 / .000988 over fixed macros in the two late-state collections | Already tested at these states. Does not justify another selector learner. |
| Do these local recoveries improve full episodes? | [Intervention study](2026-09-19-intervention-and-agent-baseline.md): all six arms 0/8; detector rarely fires; observed failures include grasp loss and lateral sleeve escape | Already tested for this trigger/library. "Elbow stall" is not an adequate general diagnosis of these policy failures. |
| Can simple teacher overrides repair the task? | [Teacher supervision](2026-09-19-teacher-supervision.md): baseline 9/25, two supervisors 0/25 and 8/25 under that study's geometry | Already tested. Improving grasp validity alone can remove motion needed for dressing. These numbers use a different geometry from historical 6/25 and must not be pooled. |
| Does duration improve action distinguishability? | [RAL calibration](2026-09-20-upper-bound-and-ral-calibration.md): 24 states, 960 branches; four-step sigma-1 effects include 16 positive and two negative effects above three pooled SD | Already tested. Reanalysis below evaluates the actual held-out coverage gain, not only a signal/noise ratio. |
| Does tighter numerical solving reduce repeated-trajectory spread? | [Tolerance study](2026-09-19-tolerance-sets-the-noise-floor.md): large local repeatability improvement with about 2.5x measured wall cost | Already measured locally, not a completed learning comparison; tolerance settings were changed together and cross-setting approach states differ. |
| Have valid reverse resets solved start-to-finish dressing? | [Historical audit](2026-09-08-uipc-manip-pretraining.md): Newton reference reverse curriculum reported high intermediate-start coverage but `0/18 at the start` | Partial-state training was already attempted in the reference. The IPC garment curriculum gates replay slots; it is not the same reset mechanism. A matched IPC experience-distribution experiment remains unverified. |

The [algorithm research assessment](2026-09-20-rl-algorithm-research-assessment.md)
must be read with this coverage map. Existing local-gradient, CEM, macro and
duration tests are controls to reuse, not a fresh experimental program.

## Reanalysis of the existing action-duration data

Input: `output/uipc_manip/ral_calibration_20260920/t26_14046/result.json`.
Reproduction:

```bash
env PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python \
  scripts/reanalyse_action_resolution.py \
  --out output/uipc_manip/action_resolution_reanalysis_20260920/summary.json
```

For each of 24 known states, choose a candidate using four repeats and score it
on the excluded fifth. Repeat for all five exclusions. This avoids using the
same outcome to choose and evaluate a winner. It is **not** a learned policy
evaluated at new states, and overlapping folds are not independent samples.
The reference applies the current policy's first action once, then continues
the policy. Four-step actions hold the initial command, then resume the policy.

| Choice, selected by sustained coverage where applicable | Mean sustained coverage | Gain over one-step reference | States with >.01 gain in every held-out repeat |
|---|---:|---:|---:|
| Policy first action, one step | .550233 | 0 | 0/24 |
| Hold that same initial action for four steps | .549299 | -.000934 | 0/24 |
| Fixed four-step sigma-1 perturbation | .552595 | +.002362 | 2/24 |
| Select among one-step perturbations | .551467 | +.001234 | 0/24 |
| Select among four-step perturbations | .555312 | +.005079 | 2/24 |
| Select duration and perturbation jointly | .555726 | +.005493 | 2/24 |

The joint choice selects four steps in 89/120 held-out decisions. Choosing by
summed task reward instead gives +.005216 coverage and selects four steps in
102/120 decisions. The qualitative duration effect survives changing the
selection score, but the coverage benefit is small: **about half a percentage
point**, not a newly successful dressing controller. Merely holding the original
action longer is slightly worse; duration alone is not a solution.

All source states already have nonzero upper-arm coverage. The collector did
not record branch grasp validity, so these numbers cannot establish valid
dressing success. The source's 16,800 decisions / 1,579.37 s were paid earlier;
this reanalysis uses **zero new physical decisions** and records its input hash.

The same script also reopens the **existing full control episodes**, rather than
assuming where failure first occurs. All eight tshirt_26 controls first violate
the 2 cm whole-episode grasp criterion at decisions **32--50 (3.2--5.0 seconds)**,
before their first positive upper-arm coverage at decisions **61--65**. Each has
only 6--12 invalid decisions: a later valid instantaneous reading does not undo
the earlier violation. All eight tshirt_392 controls preserve grasp, so that
garment's failure has a different cause. A recovery beginning at decision 100
cannot rescue tshirt_26 under a rule already violated earlier. This localizes
the need for prevention; it does not establish which earlier command change
would preserve both the grasp and successful dressing.

## Missing comparison and predeclared native protocol

The old actor audit does not save its initial observations or same-state
candidate Q values, so it cannot retrospectively supply a sampled-Q selector.
The macro data save observations, but an eight-step macro is not the action
represented by the existing single-action `Q(s,a)`. Scoring its first command
would not be a valid macro-Q test.

`python -m uipc_manip.action_selector_audit` therefore adds this narrow comparison:

- One frozen warm-SAC actor and critic, checkpoint `checkpoint_00127416.pt` in
  `output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/`.
- Two copies each of tshirt_26/14046 and tshirt_392/14046, seeds 9201--9204;
  policy-visited decisions 60 and 140, eight states total. Inspect actual progress
  and prefix grasp validity; snapshot time alone does not identify a contact event.
- Candidate 0 is the policy action. Candidate 1 is the best of eight projected
  Q-gradient iterates, retaining the base if no iterate improves Q. Eight further
  candidates are random active-axis directions at normalized radius .5.
- Both selectors obey the same normalized L2 trust region and action box.
  The disabled X-rotation coordinate stays at the policy output for every
  candidate. Translation and rotation retain the checkpoint's physical scaling.
  The sampled-Q selector chooses among the base and eight random candidates;
  it cannot reuse the gradient proposal. It is an eight-sample diagnostic, not
  a claim about globally optimal derivative-free search or matched optimizer FLOPs.
- Execute **all ten candidates**, four times each, with randomized execution order.
  Each intervention modifies one action; the other 23 decisions follow the same
  deterministic actor. Report min coverage over the final 12 decisions, whole
  prefix-plus-branch grasp validity, final coverage and discounted environment
  reward separately. No policy fitting occurs.
- An additional forward selector chooses among this same bank using the other
  three repeats and is evaluated on the excluded repeat. Its physical-query cost
  is explicit; this is not a deployable observation-conditioned policy.
- Record position restore error, action tapes, initial and terminal observations,
  actual environment configuration, checkpoint/source hashes and total cost.
  Existing snapshot restoration is reused; zero position error does not certify
  every solver cache. Repeated closed-loop continuations are not identical tapes.

Budget: 320 branches, 7,680 branch decisions plus 560 approach decisions =
**8,240 physical decisions**. A 3,000-second internal wall cap and 3,300-second
external timeout bound the run. Construction, action queries, restores, saving
and evaluation are charged. Native build is the existing `build_raw` Release
backend; GPU is the RTX PRO 6000 Blackwell Workstation Edition. No competing
training worker was observed at launch. Driver 595.84; CUDA 12.8.93; CMake
Release, CUDA architecture 120, `-O3 -DNDEBUG`. GPU activity sampled during the
run was 96--99%, about 7.5 GiB allocated and 220 W; these counters are not an
occupancy or throughput-optimality claim.

```bash
env PYTHONPATH=build_raw/python/src:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib \
  timeout --signal=TERM --kill-after=20s 3300s \
  /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.action_selector_audit \
  --checkpoint output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt \
  --out output/uipc_manip/action_selector_audit_20260920
```

## Completed native comparison

Implementation-stage commit `7796cfba` was pushed before the run finished; the
recorded source hash matches that collector. All **320 branches / 8,240
decisions completed in 1,215.83 s (20.26 min)**, including construction,
approaches, restores, queries and saving. The timer starts after initial
argument/config/checkpoint reading. Maximum recorded position restore error
was zero; no simulator failure occurred. The process and its CUDA context exited.

| Action selector | Mean predicted Q increase | Sustained coverage change, percentage points | Coverage change after zeroing grasp-invalid outcomes, percentage points | States improving valid coverage >1 point in every repeat |
|---|---:|---:|---:|---:|
| Projected Q gradient | +.23416 | +.0812 | +.2207 | 1/8 |
| Best Q among eight random candidates and the base | +.11554 | -1.2696 | +.0302 | 0/8 |
| One fixed random candidate | +.00005 | -1.3773 | -.0082 | 0/8 |
| Select by independently repeated physical outcomes | Not the selection score | +.4940 | +.4940 | 1/8 |

The physical selector uses valid sustained coverage and excludes its scoring
repeat. All tshirt_26 prefixes are already invalid; ties therefore retain the
base there. The raw sampled-Q decrease is strongly influenced by one execution
whose final three coverage readings are zero. This small development set does
not establish that sampled action optimization is generally worse.

There is one useful **positive gradient result**: tshirt_392, seed 9203,
decision 140 gains **1.94, 2.11, 2.19 and 1.56 percentage points** of sustained
coverage over its four baseline repeats, with valid prefix and branch grasp.
The second tshirt_392 state at 140 also has four positive differences, but two
are below one percentage point. At decision 60, all four states' mean gradient
effects on raw sustained coverage are negative. Thus the sign depends on the
state; the local gradient is not universally useless, and its Q increase is
not a dependable certificate of physical improvement.

**No candidate completes dressing in the tested window: 0/320.** That is a count
of repeated short branches, not 320 independent episodes. There are two
garment/body cells, four seeded prefixes and two snapshots per prefix. These
snapshots have upper-arm coverage .041--.058 at decision 60 and .563--.595 at
decision 140; they do not diagnose the separate state-actor run that never
reached the upper arm. The checkpoint's warm stage trains tshirt_26, resumes
`abl_dense_s1` weights/replay, and is the same reference used by the earlier
actor, macro and duration studies. This is not a comparison over all trained
policies or a separation of training-distribution shift from value error.

Artifacts: `result.json`, `summary.json`, `checked_results.json`,
`observations.npz` and `runtime_metadata.json` under
`output/uipc_manip/action_selector_audit_20260920/`. Checks confirm 8 unique
states, 320 unique state/candidate/repeat records, 24 decisions per trace,
finite returns, correct action bounds/trust regions/disabled axis, and no
decrease in Q for either selected proposal relative to its base.

Do not interpret a predicted Q increase as an observed task improvement. The learned Q is a soft,
long-horizon value, whereas this diagnostic measures deterministic 24-decision
consequences; disagreement is not by itself proof of Bellman inconsistency.
Both action searches maximize Q without an entropy term; neither is a complete
SAC actor update or a test of every policy-gradient estimator.
Window success is not full-episode success. Production weights, replay, reward,
control period and solver tolerances are unchanged.

## Independent continuation check of the positive gradient result

The local result justifies a smaller final check, not policy training. Freeze
the rule: at decision 140 apply one projected-Q-gradient action, then return
to SAC through decision **300**, compared with SAC throughout. Recreate the
same four-slot cell layout in a fresh world with **new seeds 9301--9304**; no
validation outcome chooses the action radius, time, optimizer or weights.
Two repeats per arm give 16 continuations from four common prefixes. The new
seeds are not new garments/bodies or independent task draws.

The collector now accepts `--samples 0` for this two-arm test and handles the
expected time-limit termination without resetting away the final state.
The conditional validation budget is 3,120 decisions and 1,700 internal seconds;
combined native wall caps remain below the earlier one-hour initial budget.

```bash
# Same interpreter/native environment as above; a new output directory.
python -m uipc_manip.action_selector_audit \
  --checkpoint output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt \
  --out output/uipc_manip/action_selector_validation_20260920 \
  --steps 140 --window 160 --samples 0 --repeats 2 --seed 9301 --max-seconds 1700
```

Validation completed: **3,120 decisions / 250.13 s**, with zero position restore
error and no simulation failure. The expected time-limit flag was checked at
decision 300 without automatic reset. Four prefixes times two repeats give
eight executions per arm:

| Full-prefix-plus-continuation arm | Valid dressing successes | Whole-episode grasp valid | Mean sustained coverage, without grasp filtering |
|---|---:|---:|---:|
| SAC throughout | 0/8 | 4/8 | .262480 |
| One Q-gradient action at 140, then SAC | 0/8 | 4/8 | .262730 |

Both tshirt_26 prefixes already violate grasp; their final coverage is about
.52--.54. Both tshirt_392 prefixes keep grasp, but **every continuation ends
with zero coverage**, in both arms. In the four tshirt_392 comparisons, the
first 24-decision sustained-coverage differences are +.00599, -.00334, +.01413
and +.00040. The local effect is smaller/mixed on these new executions; the
positive cases also fail to secure terminal coverage under the old policy's
continuation. This distinguishes failure to retain useful progress from a
claim that no differentiable action direction can help locally.

Artifacts: `output/uipc_manip/action_selector_validation_20260920/`, including
source/checkpoint hashes and the complete 160-decision continuation traces.
Both native jobs finished, totaling **11,360 physical decisions / 1,465.96 s
(24.43 min)** on the workstation. No training worker remains from this stage.

Focused checks cover trust-region/action-box/disabled-axis constraints,
exclusion of the scoring repeat from selection and the two-arm bank: **3 passed**.
Integrity checks additionally verify unique/count-complete records, trace
lengths, finite outcomes, source identity and the expected episode endpoint.

## Research decision

Do not make "remove action gradients" the selected contribution. In this
limited bank, sampled-Q selection did not beat gradient extraction; a gradient
proposal had a repeatable local positive result. Neither intervention produced
successful full dressing. This does not compare all gradient-free optimizers,
all training algorithms, or an iteratively improved policy.

Two more concrete limitations are visible: tshirt_26 violates a whole-episode
constraint before the late corrections begin, and a short local gain on
tshirt_392 does not secure terminal success when the old actor resumes. The
next algorithmic experiment should improve a **closed-loop continuation** from
before the relevant failure, using complete grasp-valid completion as its
outcome. Reuse existing successful trajectories and early-state branches;
another late single-action correction or unchanged CEM pilot is not new work.
Changing a whole continuation is a new comparison, not an established novel
algorithm: PI2-GPS, MPO and trajectory policy-iteration controls from the
[research assessment](2026-09-20-rl-algorithm-research-assessment.md) still apply.

The current tests intervene once and then restore the old policy. They cannot
rule out benefits from jointly changing subsequent decisions. Conversely,
assuming small local gains will compound would also go beyond these results.
No policy-training run or new-algorithm performance claim follows from this stage.
