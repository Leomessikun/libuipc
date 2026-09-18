# CS 185/285 reading record and implications for dressing

Reviewed 2026-09-18. This is an original reading record, not a reproduction of
the course. The resulting recommendation is in the
[research direction assessment](2026-09-18-cs285-research-direction.md).

## Coverage and limits

The public site currently presents the Spring 2026 offering. Read the extracted
text of its five public navigation pages and **every page of all 44 directly
linked current-course PDFs: 1,065 pages**. This comprises 25 lecture decks
(844 pages), 10 discussion documents (119), five homework documents (49),
three project documents (50), and one visualization handout (3).

Text extraction does not fully preserve equations, diagrams, animations or
embedded media. Selected equation/diagram pages were also rendered and visually
inspected: **63 lecture pages**, listed below. This is not a claim that every
visual element on all 1,065 pages was inspected. Lecture videos were not watched;
login-only bCourses material, linked textbooks, all historical offerings, and
every paper cited by the slides are outside this review. The calendar wrapper
was read, but its embedded Google calendar could not be retrieved.

The [source manifest](2026-09-18-cs285-source-manifest.csv) records each PDF's URL,
page count and SHA-256 at retrieval. Downloaded PDFs, per-page extracted text,
HTML and rendered images remain in the ignored local directory
`output/research/cs285_2026/`; copyrighted course materials are not committed.
The notes below are project-specific interpretations, not claims made by the
instructor about our environment. Page references are one-based PDF pages.

## Public pages

| Page | What was checked | Consequence for this review |
|---|---|---|
| [Home](https://rail.eecs.berkeley.edu/deeprlcourse/) | Current lecture, discussion, homework and project links | Defines the finite current-course corpus above. |
| [Calendar](https://rail.eecs.berkeley.edu/deeprlcourse/calendar/) | Public wrapper and embedded-calendar reference | No inference from an unavailable calendar about course progress. |
| [Resources](https://rail.eecs.berkeley.edu/deeprlcourse/resources/) | Previous offerings, textbooks, supplementary resources | These are further reading, not additional current-course PDFs silently counted as read. |
| [Syllabus](https://rail.eecs.berkeley.edu/deeprlcourse/syllabus/) | Course scope, prerequisites, assignments and project expectations | A course project suggestion is not evidence of research novelty. |
| [Staff](https://rail.eecs.berkeley.edu/deeprlcourse/staff/) | Instructor and teaching-team information | No technical inference depends on staff details. |

## All lecture decks

| Lecture | Pages | Main material reviewed and implication for this project |
|---|---:|---|
| [1: Introduction](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-1.pdf) | 38 | Problem formulation and deep RL landscape. Separate the desired dressing behavior, available information, and learning machinery before selecting an algorithm. |
| [2: Behavioral cloning](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf) | 39 | Supervised imitation, distribution shift, DAgger. Existing trajectories support a baseline, but successful imitation on recorded observations need not recover from learner-induced stalls. |
| [3: Behavioral cloning II](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-3.pdf) | 32 | History, causal confusion, multimodal actions, chunking and pretraining. Pages 5–8 motivate checking information availability; pages 21–23 distinguish broad pretraining from task adaptation. Do not imitate all failed actions indiscriminately. |
| [4: RL basics](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf) | 41 | MDPs, objectives and the collect/evaluate/improve cycle. Pages 21 and 35–38 motivate separate simulator and learner cost accounting; accurate transitions alone do not provide efficient exploration. |
| [5: Policy gradients](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) | 23 | Likelihood-ratio gradients, causality, baselines and variance. Policy learning does not require differentiating the physics solver. |
| [6: Actor-critic](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf) | 33 | Value baselines, bootstrapping, bias/variance and advantage estimation. Poor rollout return does not, by itself, identify critic bias as the cause. |
| [7: Value-based RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf) | 20 | Bellman evaluation/improvement and Q-learning. Actor performance depends on the ordering of available actions, not merely average prediction error. |
| [8: Practical Q-learning](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-8.pdf) | 36 | Replay, target networks, continuous-action maximization and practical debugging. More updates can amplify errors; candidate sampling and ranking are established techniques. |
| [9: Advanced policy gradients I](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-9.pdf) | 16 | Off-policy estimation and importance weighting. Changing the collection/reset distribution must be explicit in the learning objective and evaluation. |
| [10: Advanced policy gradients II](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf) | 33 | Policy improvement, trust regions and practical constrained updates. Stable small steps do not repair an invalid task metric or missing deployment information. |
| [11: Variational inference](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-11.pdf) | 18 | Latent-variable inference and variational objectives. A latent representation needs a justified downstream role, not just a smaller reconstruction loss. |
| [12: Variational inference in RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-12.pdf) | 37 | Generative/latent modeling and its RL connections. Representation quality must be tested under the policy's visited distribution. |
| [13: Control as inference](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-13.pdf) | 34 | Entropy-regularized control and inference assumptions. Allowing an inference procedure to select favorable physical transitions is not a deployable controller. |
| [14: LLM RL and partial observability](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf) | 51 | Sequence policies and POMDPs. Pages 40–51 are especially relevant: history can support a sufficient decision state, and some actions gather information. This does not establish that memory is our current bottleneck. |
| [15: Model-based RL I](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-15.pdf) | 22 | Learned dynamics, planning and uncertainty. Pages 17–21 distinguish uncertainty sources; snapshot variability must not automatically be called epistemic uncertainty or physical chaos. |
| [16: Model-based RL II](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-16.pdf) | 38 | Differentiating models, Dyna, short rollouts and latent dynamics. Pages 17–24 show several ways to use a simulator; long differentiable trajectories are only one option. |
| [17: Offline RL I](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-17.pdf) | 25 | Learning from fixed data and distributional shift. A large stored dataset is useful only through a compatible observation/action/reward contract; absent sampled alternatives do not prove generalization impossible. |
| [18: Offline RL II](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-18.pdf) | 35 | Conservative/constrained learning, offline-to-online RL, RLPD, IQL/IDQL, FQL and latent actions. Pages 24–32 make these strong baselines, not our algorithmic contributions. |
| [19: Exploration](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-19.pdf) | 23 | Exploration bonuses and uncertainty. For dressing, novel cloth motion need not be progress toward wearing the sleeve; that relevance must be measured. |
| [20: RL theory](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-20.pdf) | 28 | Sample complexity and theoretical assumptions. Finite/tabular guarantees do not automatically transfer to neural critics, deformable contact or our reset mechanism. |
| [21: Review I](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-21.pdf) | 34 | Review of supervised learning, policy/value learning and core derivations. Used to check objective distinctions rather than extract another proposed algorithm. |
| [22: Review II](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-22.pdf) | 73 | Extended review across inference, models and offline learning. Reinforces separating optimization, approximation and data-distribution errors. |
| [23: Advanced exploration](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-23.pdf) | 18 | Skills, empowerment and structured exploration. Skill discovery or returning to promising states is established prior art. |
| [24: Multi-task RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-24.pdf) | 53 | Multi-task learning, goal conditioning, successor features, hierarchy and meta-RL. Pages 23–31 motivate preserving control-relevant information; pages 46–53 connect history to adaptation. Neither idea is new by itself. |
| [25: Challenges and open problems](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-25.pdf) | 44 | Cost, generalization, adaptation, experience reuse and applications. Pages 25–36 support investigating a concrete failure mechanism with reusable experience, rather than assuming scale will repair it. |

## All discussions, assignments and project documents

| Document | Pages | Project-specific reading note |
|---|---:|---|
| [Section 1: PyTorch](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-1.pdf) | 62 | Autograd, networks and optimization. A physics derivative and the ordinary neural actor gradient are different computation paths. |
| [Section 2.1: Probability](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-2-1.pdf) | 4 | Conditional expectation and estimation foundations. Relevant to distinguishing privileged-state targets from observable-history values. |
| [Section 2.2: BC shift](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-2-2.pdf) | 5 | Compounding imitation errors. Randomly splitting overlapping windows does not measure recovery generalization. |
| [Section 3: PG/actor-critic](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-3.pdf) | 8 | Gradient estimators and baselines. Return targets must specify their continuation policy. |
| [Section 4: DQN/SAC](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-4.pdf) | 5 | Bellman and entropy-regularized updates. IPC accuracy does not change SAC's objective automatically. |
| [Section 5: Advanced PG](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-5.pdf) | 14 | Importance sampling, constrained updates and practical approximations. Do not call arbitrary replay updates unbiased on-policy gradients. |
| [Section 6: VI](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-6.pdf) | 8 | Inference/control derivations. Check equations against definitions rather than copy slide notation into code. |
| [Section 7: IRL/LLM RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-7.pdf) | 4 | Reward learning and sequence optimization. An attractive learned reward would still require physical-outcome validation. |
| [Section 8: Model-based RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-8.pdf) | 4 | MPC, learned dynamics and planning. Short simulator interventions need not imply a deployed MPC planner. |
| [Section 9: Offline RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-9.pdf) | 5 | Offline constraints and policy extraction. Separate a behavior prior from the reward-improved actor. |
| [HW1: Imitation](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw1.pdf) | 4 | BC and DAgger implementation/evaluation. Retain a simple imitation control using the same data. |
| [HW2: Policy gradients](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf) | 10 | Variance, baselines and learning curves. Count episodes/configurations as well as transitions. |
| [HW3: Q-learning and AC](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) | 9 | DQN and SAC implementation. Coursework allowances for showing a best seed are not an acceptable research comparison. |
| [HW4: LLM RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf) | 15 | Sequence RL and GRPO. Comparing sampled outcomes is useful, but text-prefix branching costs do not describe a cloth-state restore. |
| [HW5: Offline RL](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5.pdf) | 11 | SAC+BC, IQL and FQL. FQL's flow prior, one-step actor and Q objective are known components; its regularization is a material hyperparameter. |
| [Offline-to-online project](https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/offline_to_online_rl_default_final_project.pdf) | 21 | Existing baselines include FQL, prior-data RL, warm starts, latent RL and action chunking. Course-scale transition budgets are not workstation-time budgets for IPC cloth. |
| [LLM RL project](https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/llm_rl_default_final_project.pdf) | 25 | Data, optimization and judge/reward evaluation. Actual outcomes must be checked separately from the optimized proxy. No evidence here that adding an LLM solves dressing. |
| [Final project outline](https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/final_project_outline.pdf) | 4 | Question, method, comparisons and reporting. A defensible negative finding differs from a successful new algorithm claim. |
| [Visualization handout](https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/viz.pdf) | 3 | Learning-curve presentation and uncertainty. Show individual seeds and failures; repeated episodes from four cells do not establish broad robustness. |

## Visually inspected lecture pages

Rendered locally at readable resolution and inspected as page contact sheets:

- Lecture 14: 40–51 (12 pages).
- Lecture 15: 15–21 (7).
- Lecture 16: 16–24 (9).
- Lecture 18: 24–32 (9).
- Lecture 24: 23–31 and 46–53 (17).
- Lecture 25: 16, 20, 25–28, 35–36 and 44 (9).

This checked the key history/POMDP, uncertainty, gradient-versus-rollout,
offline-to-online, successor-feature and open-problem figures used in the
recommendation. Additional literature reading is documented separately in the
research assessment, with abstract-only checks distinguished from method reads.
