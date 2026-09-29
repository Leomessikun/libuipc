# Dressing research review — 2026-09-29

## Scope and evidence

The owner requested a review of Opus's new changes, data, results and research
continuation. The original workspace is `research/expo-ft-dressing` at
`645bda12`; its uncommitted collection changes were read and preserved. This
review runs on `research/anticipatory-dressing` and makes no process-control,
training, evaluation or live-collector changes.

The ledger/config/process snapshot was captured at **14:08 CEST** (12:08 UTC):
`output/anticipatory_dressing/status_review_20260929/report.json` in this
worktree. It includes hashed copies of result ledgers, source files and
configs, the original working diff, a process listing, MPS fatal events and
`lost_batches.json`. `teacher_training_audit.json` counts the actual feature
arrays; their hashes and the episode provenance are recorded in the report.
This is a CPU audit, not new validation of all saved cloth states.

The main hypothesis remains: roll out the original `fmvp_sim.pt` in IPC,
learn a flow policy from those demonstrations, and test transfer to new
garments and eventually arm motions. Keep IPC-corrected r1/DAgger data as a
separately named comparison. The owner's latest clarification in the active
project session explicitly distinguishes original-checkpoint rollouts from
IPC-fine-tuned checkpoint rollouts. MaxRL remains excluded.

## New implementation changes reviewed

The collector now accepts ClothesNet assets through the common garment
interface and can invoke a batched IPC action filter. The latter is a static,
one-decision selector; today's production arguments start it after decision
40, every third decision, above a 15 N load threshold. It supplies neither
future human motion nor a learned anticipation mechanism. Its short smoke is
not evidence of successful full-episode data production or exact force replay.

The new source-ablation script and completed evaluations are useful controls,
with the caveats below. The scratch-encoder trainer now rotates the force
vector together with the cloud and translation targets, fixing the earlier
augmentation inconsistency. The policy still uses one current cloud and
executes the first action of a predicted chunk at each decision. That is a
valid receding-horizon implementation, but it is not the proposed history-based
forecasting student. More GPU concurrency needs a shared peak-memory budget
and explicit recovery of incomplete jobs before it counts as higher throughput.

## Completed results, independently recounted

No duplicate policy/body/garment keys were found in these ledgers. Grasp
failures remain task failures. Missing starts are reported separately.

| Policy | Five existing garments, 201 started units | ClothesNet, 254 started units |
| --- | ---: | ---: |
| Original FMVP | 100 (49.8%) | 52 (20.5%) |
| BC round 0 | 118 (58.7%) | Not evaluated here |
| BC continuation | 105 (52.2%) | Not evaluated here |
| Flow on 236 original-FMVP episodes | 104 (51.7%) | Not evaluated here |
| r1 | 130 (64.7%) | 68 (26.8%) |
| Flow on 236 matched r1 episodes | 132 (65.7%) | Not evaluated here |
| Flow on all 1,807 indexed r1 training episodes | 138 (68.7%) | 70 (27.6%) |
| Flow with scratch point encoder | 137 (68.2%) | Separate 132-unit comparison below |

Static sources: `policy_eval_test_fmvp_20260927`, `policy_eval_test_20260927`,
`policy_eval_test_bc0_20260929`, `policy_eval_test_bcc_20260929`,
`datasource_ablation_20260929/eval_test`, and `policy_eval_test_e2e_20260929`.
The planned grid has 205 units; four have no legal start for every policy.
Logged seeds, offsets, action scales, decision intervals and success settings
match. Prior geometry audits found different settled cloth even with matching
settings; this review does not establish identical physical starts.

ClothesNet source: `clothesnet_eval_test_20260928`, all 840 planned policy
rows (20 garments x 14 body/pose IDs x 3 policies). Twenty-six of the 280
units have no legal start. Including those setup failures gives operational
completion rates of 52/280, 68/280 and 70/280. These are held-out garments
within the four represented categories, not a held-out-category experiment.

The scratch-encoder comparison has 132 common started units: original FMVP
40, r1 47, frozen-encoder flow 44, scratch-encoder flow 24. Keep the pretrained
encoder as the practical default. This comparison also changes architecture,
point resolution and augmentation, so it does not isolate whether encoder
fine-tuning helps. Do not infer the original encoder's garment training count
from this result.

The current small original-FMVP flow has not demonstrated a clear gain over
its source checkpoint. The r1-trained flow has not demonstrated a clear gain
over r1. These observations neither prove equivalence nor establish a data
ceiling. The 60k training run's lower imitation error is still not a measured
closed-loop improvement.

## Corrections to the causal interpretation

1. **The added 6,857 states are not 6,857 IPC labels.** Reading every feature
   shard gives 68,929 states for BC round 0, and 75,786 for r1. The r1 arrays
   contain **1,505 kind-6 lookahead labels**, 70,901 ordinary kind-0 states and
   3,380 holds. The increment also contains 5,112 ordinary states and 240
   holds. The two continuations use different internal held-out bodies and
   body-dependent weights. Eight epochs do not imply an equal number of
   optimizer updates. Thus r1 versus BC continuation supports a benefit of
   the changed training recipe; it does not isolate label correction alone.
   A label-only control must use the same visited states, body split,
   weighting, initialization, updates and checkpoint rule, with nominal
   checkpoint actions versus IPC-selected actions on those states.
2. **The source-matched flow comparison matches episodes, not state count.**
   Both train on 236 episodes from 57 bodies on shared successful units.
   FMVP contributes 53,745 transitions and r1 contributes 75,731. Both use
   255 r1 episodes for validation. Selecting the FMVP student's checkpoint
   against r1 actions favors r1-like behavior. Use task-based validation or
   a preregistered checkpoint schedule for the source comparison. Do not
   remove teacher-exclusive units from the main coverage experiment.
3. **Body/pose ID 3047 appeared in teacher training.** It is held out from
   the flow training index but occurs in both BC and r1 teacher training
   arrays. The current evaluation is not fully held out across the entire
   training chain. A transparent sensitivity calculation removing this ID
   gives static FMVP/r1/flow = 95/126/134 of 196, and ClothesNet = 45/61/63
   of 234. The qualitative gap remains. Retain the original results and
   reserve genuinely new IDs for final claims.
4. Failure of every policy on a garment does not establish an invalid asset.
   Use independent topology, placement and physical-validity criteria before
   excluding an asset; report coverage and task failure separately.

## Data and live work

| Source | Verified ledger/index count | Meaning |
| --- | ---: | --- |
| r1 v4 | 810 accepted records | Five garments |
| r1 v5, complete | 2,573 accepted records | Five garments; not automatically consumed by existing flow |
| Frozen flow index | 1,807 train / 255 validation / 299 test episodes | Existing evaluated model's data |
| Original FMVP, v5 settings | 322 accepted episodes, 73,250 transitions, 69 bodies | Main-route data; all paths exist and avoid current validation/test body IDs |
| ClothesNet training assets | 40 selected, 16 archives with `k300` | Ten open-front long-sleeve and six collared short-sleeve garments ready by archive check |

The 40 training garments do not overlap the 20 test garments. The remaining
24 hangs are not present at the snapshot; missing files are not completed
assets. The current ClothesNet DAgger launcher includes only the 16 existing
hangs. No original-FMVP ClothesNet collector was present in the inspected
processes. Do not call the r1 ClothesNet job original-checkpoint collection.

At the snapshot, five original-FMVP collector children run. Six Cloth3D
DAgger children and four ClothesNet DAgger children are in stopped process
state, as are all three dispatchers. The other session's commands explain
the stops and report GPU use approaching 93 GiB; that is session telemetry,
not an independent memory measurement by this audit. None was signaled here.

The MPS server records a fatal event at 13:58:12 CEST. Child logs include
CUDA error 700, illegal memory access, and some jobs exit without result
rows. At 14:08, the three new collection roots have **zero completed attempt
records**. Their top logs report **55 lost units across 11 batches**:
21 original-FMVP units and 34 Cloth3D DAgger units. These are infrastructure
losses, not dressing failures. Partial lookahead logs exist but are not
complete training episodes.

**14:15 CEST follow-up:** one original-FMVP tshirt_26 batch completed with
14 episode results and five successes. The existing CPU archive auditor
verified all five accepted episodes' metadata, valid grasp and required
zero-action success hold without errors. The stopped dispatcher has not
merged the child results, so the root attempt ledger is still absent. Four
original-FMVP workers remain active and ten DAgger workers remain stopped.
The follow-up report and copied log are in `followup_1415/` within the snapshot
directory. These five episodes are additional to the completed 322-episode
source. Zero ledger rows must not be mistaken for zero completed child work.

`collect_scaled_batched.py` logs `lost` without enqueuing a recovery, and
opens child logs with `"w"`. Repair/replay must preserve the original logs,
capture collector/checkpoint hashes, use an isolated recovery output, and
deduplicate completed episodes before merging. The recovery manifest has
been prepared; it has not been launched.

Memory pressure is a scheduling concern, but these logs do not prove an
out-of-memory root cause. NVIDIA documents that a fatal client fault can
propagate to other clients sharing a GPU and trigger MPS recovery; investigate
the first failing client as well as peak memory. See
[MPS architecture](https://docs.nvidia.com/deploy/mps/latest/architecture.html)
and [troubleshooting](https://docs.nvidia.com/deploy/mps/latest/troubleshooting.html).
New jobs need a shared reservation for expected peak allocation; a one-time
free-memory check or independent per-launcher worker limits are insufficient.
Preserve the owner's collection priority when scheduling further work.

## Next experiments and decision rules

### 1. Restore effective collection and freeze provenance

Use completed, accepted, source-verified episodes as the throughput measure.
Recover the 55 lost units once capacity is available; finish and validate
the remaining training garment hangs. Maintain distinct original-FMVP and
r1/IPC-corrected manifests. Keep simulation faults, illegal starts and policy
failures separate. Existing live jobs and logs must not be silently replaced.

### 2. Test original-FMVP scaling and garment diversity separately

Use the full eligible original-FMVP inventory for the main experiment, rather
than only units where both teachers succeeded. Compare a deterministic BC
policy and the flow policy with matched observation encoder, inputs, data,
training budget and action execution. Include an input-matched control before
crediting flow for gains over a teacher that uses zero force input.

Run two axes, not one mixed comparison:

- Fixed five garments, nested trajectory budgets (for example 250, 500 and
  1,000 accepted training episodes when available).
- Fixed trajectory/state budget, broader garment coverage (five Cloth3D
  garments versus added independent ClothesNet training garments).

These are experiment budgets, not estimates of how many trajectories must
be sufficient. Monitor successful units and coverage per garment, and use
validation dressing success rather than imitation MSE to decide whether more
data or updates help. A garment on which FMVP never succeeds provides no
successful demonstration to this route; record that limit instead of silently
switching its source to r1. Test held-out garments, held-out arm motions and
their intersection separately. Report garment-level variation, not only
pooled body/garment trials. Use multiple training seeds for the selected
contrast. The repeatedly inspected 20-garment benchmark should be treated as
a development benchmark when selecting methods; reserve a fresh final set.

### 3. Preserve the dynamic-arm research question

The completed GRAB pass pilot is one garment/body/clip. GICP and causal pause
succeed; unmodified r1, oracle pause and yoked pause fail. This establishes
neither an anticipation benefit nor its impossibility. There is no trained
future-motion teacher or forecasting student yet.

The next bounded test keeps the short-horizon action candidates, simulator
budget, scoring, starting states and execution frequency identical and changes
only the available motion: frozen current pose, causal velocity extrapolation,
or true future motion. Include the reactive registration baseline. Use validated
clips at pre-entry and sleeve-contact phases on statically successful cases;
this need not wait for every garment to work. Check full dressing outcomes,
grasp validity and body tracking in addition to local planner scores.

If future information repeatedly improves task success, learn a short-horizon
motion distribution from causal observation history and test a student
conditioned on it against a history-only student with identical dynamic data.
GRAB provides human motion; IPC dressing rollouts provide robot actions. Avoid
asking the student to predict an unobservable future intention exactly: measure
forecast uncertainty and whether the useful teacher decisions are causally
predictable. Static pretraining is the foundation; dynamic data and this
controlled information comparison address the original anticipation question.

The possible method contribution is a demonstrated way to choose insertion,
following, waiting and recovery using future cloth/arm contact feasibility
under motion uncertainty. That mechanism is still a research hypothesis.
Flow matching, more garments, standard DAgger, and an oracle teacher followed
by distillation do not establish novelty on their own. If true future motion
does not outperform the matched causal baseline, revise that hypothesis while
continuing the independent garment-transfer experiment.
