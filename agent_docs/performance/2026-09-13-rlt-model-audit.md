# Independent RLT model audit

> Historical audit of `dc953a34`. See the [final review and implementation status](2026-09-13-pretraining-infrastructure-review.md) for fixes and remaining limitations.

Audited revision: dc953a34bc54c68f65d1601ca373441ef60b7d29, 2026-09-13. Scope: rlt.py, related model heads/tests and model/objective claims. Root separately audits replay, SAC and context matching. CPU only. Probe script and captured results: /tmp/codex-rlt-audit/probe_model.py and probe_model.json. Findings below describe the audited revision before fixes.

## P1: successor and reward predictions omit the action that causes them

`TrajectoryPretrainingHead.forward` applies privileged/reward readouts only to state_t. `pretraining_step` correctly constructs state_t from observations through t and commands through t-1, so action_t is absent from these predictions. Different recorded action_t values at the same history produce identical successor predictions. For mixed/random-policy data, squared error therefore targets a behavior-dependent average transition rather than candidate-action dynamics. This is a structural issue, not a failed optimizer run.

Fix: preserve actor history, but condition privileged/reward heads on [state_t, recorded_action_t]. Keep policy action prediction separate and causal. The action-regression term is behavior cloning/conditional action regression even if the trajectories failed; training on failures does not make it something else. Make its weight explicit and zero by default when the purpose is predictive representation pretraining. Check action-dependent opposite-outcome data and prefix causality.

## P2: objective scaling and masking are unsuitable for general padded trajectory labels

Loss divides summed squared errors by number of valid timesteps, not number of selected elements. Increasing privileged-state dimension changes relative task weighting implicitly. All three losses are then summed equally, despite unrelated units.

The arithmetic mask is applied after subtracting/squaring: invalid padded NaN targets produce NaN loss and nonfinite gradients. CPU probe reproduced this with valid=[False,True], finite state and NaN only in padded action/priv/reward. Existing tests change only finite padding values and miss this failure. Production replay currently uses finite zero padding; this failure matters at the head API and for future unavailable labels.

Fix: sanitize invalid state/action inputs before linear layers; select valid predictions/targets before loss arithmetic; compute per-target element means; use named nonnegative weights. Distinguish absent labels from valid zero if future force targets require a separate validity mask.

## P2: streaming cache dtype is hard-coded implicitly

`init_state` creates KV buffers with torch.zeros and no dtype, so buffers stay float32. A model converted to float64 or bfloat16 fails on its first step with “Index put requires the source and destination dtypes match.” Reproduced for both on CPU. Allocate from the model parameter dtype/device and test streamed/window parity in another supported dtype. Explicit autocast introduces a further execution-dtype contract; parameter dtype alone does not promise arbitrary mixed-precision streaming support.

## P2 claims / performance: local cache is masked, not evicted

Streaming decoder KV buffers allocate max_len positions and retain all historic entries, while attention masks enforce W. Probe: W=8, max_len=300 allocates 300 positions in each decoder cache. This preserves mathematical attention semantics but is not the report's bounded W-1 persistent local storage. Step attention also receives the full allocated buffer, including future unused slots. Existing docs saying only the last W-1 positions are retained are inaccurate. Either describe this as a straightforward full-buffer reference implementation or implement/test actual eviction separately. Do not infer the report's memory bound or streaming cost from masking alone.

## P3: supported-mask contract is unchecked

`first_valid` detects each False-to-True edge, so a noncontiguous valid mask appears to restart s_star, while global and earlier valid attention entries remain accessible. For [True,False,True], perturbing token0 changes token2 by 0.816 in the probe. The module documents left-contiguous padding, so this is not a demonstrated production error. Prefer rejecting holes/right padding to silently creating partial-reset semantics. It must not be repurposed as a multi-episode reset mask without complete memory isolation.

## Checks that passed / scope limits

Core merge, decoder sublayer order, prefix memory mask, W including current token, optional tied modules, and alternate-token branching match the stated report computation on finite supported inputs. An independent branch probe compares all alternate positions against separate prefix-splice runs, including backward through original tokens, alternatives and parameters. Max forward error 8.88e-16; token-gradient errors <=3.67e-15. Maximum parameter absolute difference 2.24e-8 occurs with large initial-state derivatives, so relative tolerance is the meaningful comparison. This is stronger than checking merely that gradients are nonzero.

The existing full RLT/history test selection was run separately; do not interpret the historical 323-test count as verified whole-suite evidence from this audit. `step` is explicitly inference-only and does not provide training BPTT. Window resets differ from full-episode replay; root handles this integration issue.

## Primary-source attribution

[Upstream English report, pinned 1bee93a9](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/blob/1bee93a9b01c21bea0c7a50ce3f6619f24731e19/Recurrent_Looped_Transformer.pdf): §§2.3–2.6 define complete recurrent state and encoder/global/local attention; §§3.1–3.3 distinguish conditional execution equivalence, structural depth and cost; §5.1 defines discrete autoregressive likelihood; §5.4 and Appendix C distinguish full gradients, detached state and parameter-stale caches. Our continuous action/state/reward objective is a robotics adaptation, not that exact language-model objective. The report claims no measured quality or throughput benefit. Its temporal depth does not imply extra iterative compute per robot decision.

Bounded fix scope authorized after audit: rlt.py and test_rlt.py only; root owns documentation, CLI and integration corrections.
