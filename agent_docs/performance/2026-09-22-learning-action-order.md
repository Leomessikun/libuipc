# Learning the effect of action order: a concrete research candidate

Date: 2026-09-22. Branch: `research/expo-ft-dressing`.
Scope: literature, existing-source inspection and analytic derivation only.
No tests, numerical probes, simulation, training or empirical re-analysis.

## Research decision

The old Stage 0–3 execution plan remains retired. Correct objective semantics
are necessary; a new from-scratch IPC SAC training comparison is not a necessary
prerequisite for research. Teacher/student imitation, goal imitation and DAgger
remain excluded. SAC and IPC are not requirements.

The preceding [operation-model assessment](2026-09-22-segment-decision-research.md)
specified a conventional model-based planning architecture. Its selection as a
new research route is withdrawn: the review did not establish a new mechanism.

This note instead specifies one narrower candidate: **learn which ordering of
the same gripper movements improves the task outcome**. Condition comparisons on
the same starting state, movement inventory and duration. Update the deployed
policy directly from consequences. There is no successful teacher to imitate,
no separate student, and no prerequisite of learning full cloth dynamics.

The proposed contribution would be an exploration and credit-assignment mechanism
for indirectly controlled contact. It is not a new policy-gradient identity.
Its physical premise and benefit in dressing remain unverified. This is a
research candidate, not an implementation or a claim that dressing is solved.

## What motivates it, and what does not

The existing [action-selection record](2026-09-20-action-selection-evidence.md)
shows little benefit from the tested local action perturbations and late macro
selectors. The [long continuation record](2026-09-21-feasible-segment-result.md)
shows different early interventions can produce different later validity under
the incumbent. Those are reasons to question the existing exploration choices.
They do not establish an action-order effect.

In particular, policy/lift/outward are different movements with different
endpoints. The macro library includes retreat followed by outward, but the
reviewed records do not provide a controlled comparison with the reversed
sequence using the same command inventory. The 48 long continuations also all
have zero valid completion. They cannot supply a measured completion advantage
for the candidate below.

The physical hypothesis is that changing a path can change the subsequent cloth
configuration, contact and tension even when the gripper reaches the same place.
This is plausible, not a diagnosis of this codebase. Human–textile hysteresis has
been measured in a soft wearable robot, but that system is not our simulated
dressing task; its result does not establish hysteresis of our constitutive
model or usefulness of the proposed commands.
([McCann et al., author-hosted publication](https://biodesign.seas.harvard.edu/publication/body-textile-hysteresis-estimation-personalized-physical-human-robot-interaction))

An elementary illustration, not a cloth model, is

\[
\dot x=u_x,\qquad \dot y=u_y,\qquad \dot z=xu_y.
\]

Starting at zero, moving by A=(a,0) then B=(0,b) ends at (x,y)=(a,b)
with z=ab. B then A has the same endpoint and z=0. This demonstrates why an
actuator endpoint need not specify the entire system outcome. Nonholonomic
control and RL already exploit related phenomena; this illustration is not a
new controllability result. A step-based history policy can represent both
orders too. The proposed difference is how exploration and credit are organized.

## The policy and executable action

Use the available observation/action history h, including time remaining and
the already-violated flag. A finite history encoder is an approximation, not a
guaranteed sufficient belief state. Privileged simulator state is needed for an
exact branch restore, not as a deployment input.

For the minimal definition, hold orientation fixed and use translations in a
fixed world frame. Choose two displacement vectors w=(v_A,v_B). Each movement
uses L decisions with increments v_A/L or v_B/L; H=2L is fixed. An equivalent
parameterization separates net displacement d from path shape r:

\[
v_A=d/2+r,\qquad v_B=d/2-r.
\]

The two orders are A then B (o=1), and B then A (o=0). They have the same:

- commanded net displacement d;
- duration 2L and multiset of commanded increments;
- commanded path length ||v_A||+||v_B||.

The action distribution factorizes as

\[
\pi_{\alpha,\beta}(w,o\mid h)
=\pi_\alpha(w\mid h)\,
\operatorname{Bernoulli}(o;p_\beta(h,w)),
\quad p_\beta=\operatorname{sigmoid}(\ell_\beta).
\]

Use distinct parameter blocks for the derivation. Let pi_alpha be a distribution
over pulse pairs named in a fixed lexicographic order in the declared frame;
its logged probability must be for that distribution. Swapping their names and
flipping o otherwise describes the same path, not another physical action.
Identical pulses have zero order contrast. Re-observe and choose another segment
after execution. This gives six continuous segment parameters and one order choice;
it restricts the path family, and does not prove that the best dressing motion
is representable. Orientation control is outside this minimal candidate.

Bounds apply to each pulse before execution. Neither zero net displacement nor
the common endpoint guarantees validity between endpoints. Clipping, rejected
commands, compliance and tracking lag can also destroy equality of the *executed*
endpoints. A future implementation must retain both commanded and executed
motions. An observed difference is an effect of the full command order, possibly
including controller intervention; it does not isolate friction from geometry,
inertia or other causes. Smoothing cannot silently change the matched quantities.

## Objective and the quantity being learned

For the completion metric in the continuation record, let

\[
G=\mathbf 1\!\left[
\max_{0\le t\le T}e_t\le 0.02\ {\rm m}
\;\land\;
\min_{T-11\le t\le T}\mathrm{coverage}_t\ge 0.7
\right],\qquad J(\pi)=\mathbb E_\pi[G].
\]

Here e is the recorded maximum anchor-tracking error. This remains a simulator
proxy, not a physical grasp or human-safety certificate. A restored branch needs
the prefix violation flag to evaluate this *whole-episode* objective. An unknown
prefix cannot be declared valid. A timeout is not a measured G=0.

Define Q_1^pi(h,w) and Q_0^pi(h,w) by executing the respective order, then
continuing with the same current policy pi until the task horizon. The central
quantity is

\[
\Delta^\pi(h,w)=Q_1^\pi(h,w)-Q_0^\pi(h,w).
\]

It measures the preference between two orders at fixed w. It is conditional on
the continuation policy, not a permanent good/bad label for either movement.
With M=(Q_1+Q_0)/2,

\[
\mathbb E[G\mid h,w]=M+(p-1/2)\Delta.
\]

This decomposition separates the common outcome from the order preference.
The six continuous parameters still have to be learned; order learning does
not replace that task or solve perception aliasing.

## A fully specified update, with its limitations

For one history and sampled movement pair, the order contribution to the
policy gradient is

\[
g_\beta(h,w)=p(1-p)\Delta^\pi(h,w)\nabla_\beta\ell_\beta(h,w).
\]

If two complete continuations provide unbiased returns R_1 and R_0, substitute
R_1-R_0 for Delta. The movement-distribution contribution is

\[
\widehat g_\alpha(h,w)=
\big[pR_1+(1-p)R_0-b(h)\big]
\nabla_\alpha\log\pi_\alpha(w\mid h).
\]

Treat the sampled returns and baseline as constants in this score update.
The second expression accounts for learning the movements, rather than merely
learning which of two fixed macros to choose. It does not imitate either path.

Summing these contributions over segment histories gives the usual finite-horizon
policy gradient when histories and w follow the current policy, continuations
use that policy, and the necessary score-function regularity conditions hold.
Future policy dependence is accounted for by the later decision terms. Arbitrary
archived checkpoints instead define a context-weighted local improvement
surrogate; they are not an unbiased full-policy gradient without correcting the
sampling distribution. Simply replaying the old 2,424 branches is insufficient.

A concrete collection/update definition avoids an exponentially branching tree:

1. Hold the current policy fixed. For a fixed episode horizon T=KH, choose one
   segment index j uniformly from 0,...,K-1, independently of its future outcome.
2. Execute one ordinary on-policy episode. At j, retain the complete simulator
   checkpoint, observation/action history, prefix event flag, sampled w and o.
   Keep the realized complete-episode return for that order.
3. From that checkpoint, execute the opposite order with the same w, then the
   same policy. Its complete outcome supplies the other return. Other segment
   decisions on this extra continuation are not extra unweighted training
   contexts: that would change the state distribution.
4. Multiply the two displayed local gradient estimates by K to correct for
   selecting one of K decision indices. Average across independent episodes
   and take a policy step. The derivative is exact in expectation under the
   stated assumptions; a finite noisy policy step need not improve performance.

The ordinary episode supplies one side of the comparison. Only the opposite
continuation is additional. Ignoring restore cost and assuming both execute to T,
this uses T+(T-jH) physical decisions: a mean of 3T/2+H/2 over j, compared with T
for one ordinary episode. This is a decision-count identity, not a runtime
prediction; contact states have different costs. A computational cutoff creates
missing outcomes, not free completed labels. No collection was performed here.

Pairing is not required for the basic actor. With a single sampled o, an ordinary
on-policy estimate is

\[
\widehat g_\beta=(R-b(h,w))(o-p)\nabla_\beta\ell_\beta.
\]

The baseline may depend on w but not the sampled o. It must not be subtracted
unchanged from the w-score update: that generally biases movement learning.
Fit a baseline from separate/prior data or handle its sample dependence; merely
stopping its gradient does not remove dependence on the current outcome.
These are established action-dependent baseline principles, not our invention.
([Wu et al., ICLR 2018](https://arxiv.org/html/1803.07246))

The paired estimator explicitly averages over the two order choices. It does
not require differentiating cloth dynamics, but still uses policy gradients.
For fixed (h,w), let s=grad_beta ell, let sigma_o^2 be the return variance and
C=Cov(R_1,R_0). Its covariance is

\[
\operatorname{Cov}(\widehat g_\beta\mid h,w)
=p^2(1-p)^2(\sigma_1^2+\sigma_0^2-2C)\,ss^\top.
\]

Independent continuations have C=0. A shared seed does not establish positive
covariance in the nondeterministic solver. Pairing spends two continuations and
does not have a distribution-free advantage per workstation second. Enumerating
a binary action and subtracting returns is standard conditional estimation.

For the single-order scalar-logit estimate, the variance-minimizing baseline is
b*=(1-p)Q_1+pQ_0, not generally E[G|h,w]=pQ_1+(1-p)Q_0. Thus even the claim that
"subtract the conditional mean and variance must improve" would be too strong.
At p=1/2 with this oracle baseline and independent noise, two ordinary samples
averaged have the same conditional variance as one matched pair. The proposed
benefit must come from learning useful movement/order structure, not an invented
universal variance reduction theorem.

## The sparse-success obstruction remains

If both observed returns are zero, their observed order contrast is zero.
Reparameterizing actions or adding a baseline supplies no missing success
evidence. This describes the available 48 long branches for the binary objective;
it does not prove that all reachable policies have zero success probability.

Validity-weighted coverage supplies a different, denser objective. It must be
named as a surrogate: improving it can favor valid but incomplete dressing and
does not prove improved completion probability. No surrogate training was
selected or implemented here.

Consequently this candidate alone is not an established solution to sparse
success, expensive physics, partial observability, or actuator limitations. It
would be misleading to replace the old two-day training promise with another
one based only on fewer policy decisions.

## Closest prior art and the permissible claim

Primary sources checked on 2026-09-22. Scope descriptions below concern the
specified sources, not an exhaustive novelty certification.

| Source | What it already covers | Consequence for this candidate |
|---|---|---|
| [Deep Black-Box RL with Movement Primitives, Otto et al.](https://proceedings.mlr.press/v205/otto23a.html), [method](https://arxiv.org/html/2210.09622) | A contextual policy chooses trajectory parameters and learns from returns. | Choosing a short path instead of a per-step command, without a teacher, is already established. |
| [TCE, Li et al., ICLR 2024](https://arxiv.org/html/2401.11437), Sections 2–3 | Correlated movement-primitive exploration and segment-wise policy updates using step information. | Temporal correlation, smoothness and segment credit cannot be the novelty claim. Its method does not supply our measured matched-order effect; neither have we supplied one. |
| [Wu et al., ICLR 2018](https://arxiv.org/html/1803.07246), Section 4 | Factor-conditioned unbiased baselines and their variance analysis. | Conditioning order credit on the chosen movements is an application of established estimation theory. |
| [Learning Sequences of Manipulation Primitives, Vuong et al.](https://arxiv.org/abs/2011.00778), abstract | RL discovers sequences of contact-aware primitives for assembly. | Learning action order or using contact primitives in general is not new. |
| [Nonholonomic Yaw Control with Model-based RL, Lambert et al.](https://arxiv.org/html/2009.01221), Sections III and V | Learned control exploits sequential effects beyond direct instantaneous actuation. | Noncommutativity, loops and Lie-bracket motivation are prior art. Their smooth-system derivation cannot certify hybrid cloth contact. |
| [On-body textile hysteresis, McCann et al.](https://biodesign.seas.harvard.edu/publication/body-textile-hysteresis-estimation-personalized-physical-human-robot-interaction), abstract | History-dependent textile behavior is modeled for wearable assistance. | Physical plausibility is not evidence of dressing policy improvement or of the current simulator's material behavior. |

The narrow candidate claim is therefore: **task-conditioned exploration over
matched reorderings can expose and teach useful contact consequences more
efficiently than ordinary trajectory-parameter exploration**. That is a
falsifiable mechanism hypothesis, not a result. Merely implementing movement
primitives plus the equations above would not establish a new RL algorithm.
BBRL and TCE are direct methodological neighbors; comparing only against the
abandoned SAC configuration would not support this claim.

This pass completes the action definition, objective, direct policy update,
estimation/cost limits and primary-source overlap screen. It does not establish
that order is this task's bottleneck. The preceding broad model-planning route
and the original teacher/student plan are not reinstated. Existing code and
data are preserved; no experiment or training launch follows from this note.
