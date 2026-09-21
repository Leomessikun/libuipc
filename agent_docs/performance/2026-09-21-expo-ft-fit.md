# Would Real-Time EXPO-FT finish this project? The one number that decides it

Date: 2026-09-21. Branch `research/expo-ft-dressing`, opened for this question.
Read-only measurement, no simulation, no training. Script
`scripts/measure_expo_edit_budget.py`, artifact
`output/uipc_manip/expo_edit_budget_20260921/summary.json`.

## 1. What the method is

[Real-Time EXPO-FT](https://pd-perry.github.io/real-time-expo-ft/) (Perry Dong,
Kuo-Han Hung, Dorsa Sadigh, Chelsea Finn, Stanford) runs three parts: a large
vision-language-action base policy proposes action chunks slowly, a learned **edit
policy** transforms those actions using the observation available at execution
time, and a learned **Q-function picks** among the base and edited candidates on
the fly. The base is frozen; only the edit policy and the critic are trained. Its
stated problem is latency — while the robot executes, the world moves, so the
observation the base saw is stale.

Its parent is [EXPO](https://arxiv.org/abs/2507.07986) (Dong, Li, Sadigh, Finn,
July 2025): an expressive base trained by imitation, a lightweight Gaussian edit
policy, and value-maximising selection over the base and edited actions for both
acting and the TD backup, claiming 2-3x sample efficiency over prior methods. The
project page for the real-time variant publishes no budget, task count, baseline
table or limitation section, so nothing quantitative from it is used below.

## 2. Why it is structurally attractive here

- It never backpropagates through the expressive base, which is what made the
  flow/diffusion fine-tuning work in this project unstable.
- Its budget regime is the one this project measured itself into: useful policy
  improvement in 10^3-10^4 decisions, which at .0610 s per transition is minutes.
- We already have both halves of its architecture: a deployable base policy, and
  a controller that demonstrably succeeds on the states where the base fails
  (teacher 4/4 with valid grasp and .979/.988 sustained coverage, fitted student
  0/4 from the same restored states).

## 3. The measurement that decides it

EXPO is sample-efficient **because the edit is small**. The edit policy has a
bounded output, so whatever lies outside that bound is unreachable however good the
critic is. This project can measure the required bound directly: take the 720
decisions of a controller that succeeds, ask what the deployable base would have
commanded on the same observations, and measure the gap in the units an edit bound
is expressed in — each action coordinate lives in [-1, 1], so the largest possible
edit norm over the five commanded coordinates is 2.236.

| prefix | base | median | p90 | max | largest single coordinate |
|---|---|---:|---:|---:|---:|
| original | original_bc | .384 | 1.484 | 1.867 | 1.680 |
| original | recovery_bc | .221 | 1.194 | 1.875 | 1.708 |
| student | original_bc | .308 | 1.434 | 1.838 | 1.636 |
| student | recovery_bc | .247 | 1.143 | 1.901 | 1.714 |

Fraction of the succeeding controller's moving decisions that an edit of at most
`e` could reproduce:

| edit bound `e` | .05 | .1 | .25 | .5 | 1.0 | 2.0 |
|---|---:|---:|---:|---:|---:|---:|
| original_bc base | .00 | .06 | .36 | .55 | .79 | 1.00 |
| recovery_bc base | .00 | .11 | .56 | .69 | .87 | 1.00 |

**A small edit cannot express the behaviour that works.** At the bound EXPO's
lightweight editor is usually run at, .05 to .1, the composite policy can reproduce
between 0 % and 11 % of the successful controller's decisions. Covering 90 % needs a
bound near 1.0 — half of each coordinate's full range — at which point the edit is
close to a replacement and the frozen base is contributing little of what matters.
The distribution is also heavy-tailed: the median gap is only .22-.38, so most
decisions need a moderate edit and a minority need almost everything. On this task
a minority is decisive, because the constraint that decides success is absorbing
and is first broken at decisions 32-50.

Two things this does **not** say. The succeeding controller is privileged — it
reads named opening and alignment vertices and its own stage history — so part of
that gap is information the base cannot see, not action the base could not have
chosen; an edit policy reading the deployment observation inherits that ceiling.
And a large required edit is not proof that EXPO fails, only that its efficiency
argument does not transfer unchanged.

## 4. What else this project has already measured against the mechanism

- **Value-maximising selection over sampled candidates was tested here and did not
  win.** With one frozen critic over 320 branches, choosing the best of eight random
  candidates plus the base changed sustained coverage by **-1.27** points, while a
  projected-Q gradient proposal changed it by **+.08**, and neither completed the
  task ([action selection](2026-09-20-action-selection-evidence.md)). EXPO's
  selection rule is that same operator family, so the critic, not the editor, is
  the first thing to fix.
- **Single-action edits never moved the binding constraint.** None of ten
  single-action candidates changed the feasibility label at any of eight audited
  states, while eight-decision segments flip it deterministically. An editor that
  edits one action at a time cannot reach the quantity that decides success here;
  it has to edit the chunk. EXPO-FT already operates on chunks, which is the part
  of its design that transfers best.
- **The behaviour prior does not contain the helpful actions.** The recovery macros
  that beat the policy sit at flow-prior percentile 1.000 with |z| 5.2-5.5
  ([flow prior support](2026-09-19-flow-prior-support.md)), which is the same
  conclusion as the edit table above, reached from the other side.
- **Latency is not our problem.** The real-time contribution addresses a slow VLA
  against a moving world; our actor is a small point encoder and inference is
  milliseconds against a .1 s control period. What transfers is EXPO, not the
  real-time part — and saying so keeps us from claiming a contribution that is
  theirs.

## 5. What this branch will try

The adaptation that the measurements above actually support, rather than a port:

1. **Base**: the deployable relation-goal policy of Stage 1, not a raw action
   policy, so that an edit of moderate norm corresponds to a meaningful change of
   where the sleeve opening should go rather than a jitter of one command.
2. **Edit**: bounded but **large** (the table says ~1.0 to cover 90 %), applied to
   a chunk rather than a single action, with the bound itself reported as the
   variable it is instead of a tuned constant.
3. **Selection**: value-maximising choice over base and edited chunks, but gated by
   the Stage 0 constraint cost, because the measured failure of pure Q-selection
   here is that it ranks an infeasible candidate first.
4. **Critic**: trained on the constrained objective from Stage 0, since a critic
   that does not represent the absorbing constraint cannot rank candidates by the
   criterion that decides success.

**What would kill it.** If a large edit bound reproduces the successful
controller's actions but the composite policy still fails from reset, the gap is
information rather than action, and the work belongs in the observation, not the
editor. If the constrained critic still ranks infeasible chunks first, selection is
not the mechanism. If the whole thing matches plain DAgger on the same budget, then
what helped was corrective labels, not the editor, and we report that.

Nothing here is a novelty claim: EXPO's editor, its selection rule and chunked
real-time execution are all published. Any contribution would have to be in what
the measurements above force — the bound, the chunk, and the constraint gate — and
would have to beat DAgger and a constrained baseline at equal workstation seconds.
