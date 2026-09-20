# The privileged upper bound, and what resolves — 2026-09-20

Correction: the [subsequent evidence review](2026-09-20-training-issue-review.md)
confirms the zero upper-arm evaluations and the local four-step separation, but
rejects the flat-junction argument and the matched-upper-bound interpretation.
The runs differ in control period, physical episode length, angular speed limit
and discount time. The calibration's 18/24 absolute separations include 16
improvements and two losses; its observed throughput implies about 45 seconds
per 480 decisions, not 12. Its states are already on the upper arm. The historical
causal conclusions and RAL feasibility claims below are not established.

Status: both runs complete. `output/uipc_manip/state_ub_clean_s1/` (270,000 transitions,
22 evaluations) and `output/uipc_manip/ral_calibration_20260920/t26_14046/` (24 states,
960 branches, 16,800 decisions, 26.3 min).

## 1. The feasibility gate: 0 of 25, with perception free

The pre-registered run of the 2026-09-19 handoff: a state actor reading the 35-float
privileged vector, a privileged critic, trained and evaluated on the same 25 cells with
nothing held out, for the budget of the longest SAC run on record.

| transitions | return | threaded, peak | forearm ratio, peak | **upper-arm ratio, peak** | success |
|---:|---:|---:|---:|---:|---:|
| 12,500 | -272.6 | 0.52 | 0.019 | **0.0000** | 0/25 |
| 75,000 | -59.3 | 0.32 | 0.029 | **0.0000** | 0/25 |
| 175,000 | -2.7 | **1.00** | 0.159 | **0.0000** | 0/25 |
| 200,000 | +36.5 | 1.00 | 0.296 | **0.0000** | 0/25 |
| 270,000 | **+67.8** | 0.80 | **0.419** | **0.0000** | 0/25 |

The pre-registered reading was: eight of twenty-five or worse closes the road as posed.
It is zero, at every one of twenty-two evaluations.

But the run is not failing to learn. The return rises from -272.6 to +67.8, every
episode threads the opening onto the hand by 175,000 transitions, and the sleeve then
advances to 42 % of the forearm. What it never does, once, is reach the upper arm.

**Wang's objective has two plateaus, not one.** At the fingertip, minus the
fingertip-to-opening distance rises to exactly zero and forearm progress starts at zero.
At the elbow, forearm progress ends at `forearm_len` and the upper-arm branch starts at
`forearm_len + 5 * 0`. Both junctions are continuous and flat. The run spent about
175,000 transitions crossing the first and the remaining 95,000 covering 42 % of the
forearm without reaching the second.

So the verdict is not "the control problem is unlearnable". It is that this objective
costs more than a workstation's budget per plateau, and there are two of them before the
metric moves at all.

## 2. What resolves above the noise

ADR 0009 requires a branch score whose margin clears the environment's noise before its
gate can ever fire. Measured on 24 states the policy visits: the base action is the
policy's own, a candidate is that action displaced by `sigma` along one random direction
per state and held for `hold` decisions, each repeated five times from one exactly
restored state, window sixteen decisions.

Median margin over pooled within-action standard deviation, and the fraction of states
clearing three:

| score | hold 1, sigma 1.0 | hold 4, sigma 0.5 | **hold 4, sigma 1.0** | states over 3 sd |
|---|---:|---:|---:|---:|
| summed task reward | 1.45 | 2.61 | **5.29** | **75 %** |
| sustained coverage | 1.35 | 2.84 | **4.19** | 54 % |
| opening progress along the arm axis | 1.42 | 2.51 | 2.43 | 42 % |
| final upper-arm ratio | 1.15 | 1.68 | 2.19 | 38 % |
| threading fraction | degenerate | degenerate | degenerate | - |

Three conclusions, one of which contradicts the prediction that motivated the score.

* **Nothing resolves at single-decision granularity.** The best score reaches 1.45
  standard deviations for a full-magnitude displacement of one command. An operator that
  compares primitive actions cannot work here at any affordable number of repeats.
* **A four-decision burst does resolve.** Summed task reward reaches a median 5.29 and
  clears three standard deviations at three quarters of states. RAL must therefore
  compare temporally extended action segments, which is a constraint on the algorithm
  and not a tuning choice.
* **The geometric score, predicted to win, is the weakest usable one.** A smooth
  projection of the opening onto the arm axis was proposed because it needs no ray cast;
  it reaches 2.43 against the summed reward's 5.29. Summing over the window averages the
  simulator's noise, and reading one final position does not. An early two-repeat smoke
  reading of this score at "2 to 9" was noise and is withdrawn.
* **Threading probability is degenerate at these states.** Its within-action variance is
  zero at every state in the sweep, so the milestone potential ADR 0009 proposes cannot
  be estimated at states where the milestone is already decided. It must be measured at
  states near an undecided milestone, which the upper-bound run now identifies: the
  elbow.

## 3. What this does to ADR 0009

The calibration passes, conditionally, and the conditions are specific:

* the compared unit is a four-decision segment, not a command;
* the score is the summed task reward over a sixteen-decision window;
* five repeats per candidate give a usable pooled standard deviation, so one state's
  improvement over six candidates costs about 480 decisions, that is twelve seconds on
  this workstation, and a day's budget is about eight thousand state improvements;
* the gate will decline about a quarter of states even at full displacement, which is
  what it is for.

And the upper-bound run tells RAL where to spend: the milestones are the two plateaus,
and the second one, the elbow, is where a learner with free perception and a full budget
stops.
