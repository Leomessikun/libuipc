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

The native result is pending at this implementation stage. Do not interpret a
predicted Q increase as an observed task improvement. The learned Q is a soft,
long-horizon value, whereas this diagnostic measures deterministic 24-decision
consequences; disagreement is not by itself proof of Bellman inconsistency.
Both action searches maximize Q without an entropy term; neither is a complete
SAC actor update or a test of every policy-gradient estimator.
Window success is not full-episode success. Production weights, replay, reward,
control period and solver tolerances are unchanged.

Focused checks cover trust-region/action-box/disabled-axis constraints and
preventing a held-out outcome from selecting its own winner: **2 passed**.
