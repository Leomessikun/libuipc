# Fresh IPC actor and dressing architecture defaults

The seed-0 64-slot run reached return 13.31 and 29/64 successes with actor
physics, versus 6.78 and 9/64 for SAC. Critic-only reached 6.58; combined
reached 12.33. These are single-training-seed results, not independent trials.
The old bounded 1024-row tangent sidecar diluted actor replacement under
uniform TD replay: effective rho fell from about .152 to .00731. Conditional
critic/teacher cosine .873 measures agreement with a learned teacher, not
accuracy against the real return gradient. Late shuffled minibatches left
45.6% of valid labels unchanged. These results motivate controlled experiments;
they do not prove dilution is the sole limitation.

## Implemented protocol

`iaql_benchmark` defaults to fresh actor batches, raw accepted-state exports,
1024 random warmup transitions, and estimator replacement. The actor reconstructs
the executed bounded action with saved Gaussian noise and asserts equality
before one update. All fresh rows carry labels; TD replay remains independent,
with no mechanics sidecar for actor-only runs. TD updates skip actor updates.
Current-action locality is measured after tanh. Logs expose actor/critic update
counts, label coverage, effective rho, action mismatch and post-update movement.
Teacher/control RNG streams are independent of policy sampling. Paired, random
norm-matched, zero and negative teacher controls are available. Randomizing the
teacher does not preserve the norm of the correction `g_teacher - g_Q`.

Terminal tasks include remaining time, mask terminal bootstrap, and require
full-horizon evaluation. Dense time-limit truncations retain bootstrapping.
Fresh changes the state distribution and update frequency: compare against
fresh rho=0 SAC at matched budgets, retaining historical replay SAC separately.
Replay mode remains available, with the corrected post-tanh gate; it cannot
bitwise reproduce the old gate bug.

Dressing launchers default to dense action-conditioned Q with a normalized
residual trunk. Dense Q and residual normalization have separate provenance;
the latter is not attributed to the dressing paper. Saved architecture wins on
resume; legacy unspecified checkpoints retain latent/plain compatibility.
Full-state IPC experiments retain explicit plain trunks for controlled tests.

## Verification and limits

The relevant CPU suite passes, including analytic actor-gradient comparisons,
same-action assertions, TD-only updates, terminal evaluation, resume handling,
and an affine-environment replay-capacity invariance test. Native CUDA smoke:
4 slots, 96 transitions, 32 warmup, 16 fresh actor / 9 critic updates. Dense
rho=.5 and terminal/residual rho=1 both have full label coverage, exactly their
configured effective rho, and zero anchor mismatch. These are smoke tests,
not learning comparisons. Artifacts: `output/iaql/fresh_smoke_20260915_rho05`
and `output/iaql/fresh_smoke_20260915_terminal` (see actual run folders).

`scripts/probe_mechanical_metric.py` is an analytic spring diagnostic, not IPC
validation. It shows that `D^T K D` can get smaller along a stiff, suppressed
response direction. Its inverse can encourage a harder push. This metric is
therefore not installed as a safety mechanism or actor default.

Open: multi-seed learning curves; simplex cloth-body friction coupling;
visual/history actor integration; real-contact metric tests; counterfactual
response pretraining and its finite-radius validity. Existing replay snapshots
lack IPC derivatives and tracked response vertices, so those cannot be
silently reused as differential datasets.
