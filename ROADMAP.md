# Roadmap — Phase 2: Isaac Lab

Days 1–17 built every layer of a physical AI system by hand: kinematics,
physics, the environment loop, and the learning algorithm. Phase 2 trades the
hand-built miniatures for the real thing — **NVIDIA Isaac Lab** running GPU
simulation, training a **Franka arm on a manipulation task**.

Phase 2 is not hypothetical: it is already running in the sibling repo
[grasping-twin](https://github.com/welidemezene/grasping-twin), which carries
the training scripts, checkpoints, and motion replays. This repo remains the
study log — the roadmap below turns what is happening there into the same
day-by-day mastery record as Days 1–17.

---

## The verified stack (audited 2026-08-03)

The simulation rig lives in **WSL2 Ubuntu**, not on the Windows side:

- **Training rig (the one that works):** Docker image
  `grasping-twin-isaaclab:latest` (52 GB), built on the pinned
  `nvcr.io/nvidia/isaac-sim:4.5.0` container — the workaround for the
  Ubuntu 26.04 glibc mismatch. All week-2/3 training ran here; it produced
  stage-9 motion captures as recently as this morning.
- **Native env `~/env_isaacsim`:** Isaac Sim 4.5.0 + Isaac Lab 2.1.0,
  torch 2.11.0+cu128, `torch.cuda.is_available() == True` on the
  RTX 5070 Ti. Works for scripts and evaluation.
- **Native env `~/env_isaaclab`:** Isaac Lab 2.3.2.post1 **without** Isaac
  Sim — `import isaaclab` dies on `ModuleNotFoundError: isaacsim`. A leftover
  from the first install attempt; repair it or delete it, but don't build on
  it.
- **Windows `env_isaacsim_win`:** empty venv (pip + setuptools only) —
  another abandoned start, safe to delete.
- Disk is not a constraint (823 GB free in WSL); WSL RAM cap already tuned
  via `.wslconfig`, and 4096 parallel envs are confirmed to fit this laptop.

## Where the training actually stands (grasping-twin)

- **Week 2:** Franka-lift PPO trained in the container up to ~4.5 M steps,
  checkpointed throughout.
- **Week 3:** curriculum + custom grasp reward reached ep_rew_mean 88 — but
  motion replays exposed the truth: **the cube was batted upward, not
  grasped**. Anti-exploit rewards (height only pays while the cube is held),
  warm-starts, and a surgical gripper-head reset followed (stages 4–9).
- **The open problem:** at stage 9 (~3 M steps) the gripper reaches within
  2 cm of the cube, but *the fingers never fully shut* and the cube never
  rises. This — closing the hand — is the current frontier.

---

## The days

Each day ends with something runnable plus a mastery write-up, same as
Phase 1. Days 18–20 document work already lived through in grasping-twin;
days 21–24 push the frontier.

### Day 18 — The rig
Write up the simulation stack as it actually exists: why native installs
failed (glibc, Blackwell wheels), why the pinned 4.5.0 container won, how
the pieces (WSL2, Docker, NVIDIA Container Toolkit, mounted repo) fit.
**Done when:** the mastery doc lets a stranger rebuild the rig, and the two
dead venvs are repaired or removed.

### Day 19 — Anatomy of a real task
Read the Franka lift task source inside Isaac Lab: environment config,
observation terms, reward terms, resets, randomization hooks.
**Done when:** the write-up maps every block to the day that built it by
hand — observations → Day 7, shaped reward → Day 6, partial resets →
Day 11, noise → Day 12, env API → Day 15, the articulated arm → Day 17.
This is the payoff day for Phase 1.

### Day 20 — The reward-hacking post-mortem
Tell the batted-not-grasped story with evidence: the ep_rew_mean-88 policy
that gamed the height reward, the motion replays that exposed it, and the
anti-exploit reward that closed the loophole.
**Done when:** the write-up pairs each exploit with the reward term that
enabled it — the production sequel to Day 6's sparse-vs-dense lesson.

### Day 21 — Close the hand
The frontier. Diagnose why the gripper action saturates open despite the
head reset (action scaling? penetration penalties? reward too indirect?)
and get fingers physically shut on the cube in replay.
**Done when:** a motion capture shows closed fingers with the cube held —
even briefly. Lift height doesn't matter yet.

### Day 22 — Lift while holding
Extend the curriculum from "fingers shut" to "cube rises while held":
height reward gated on grasp, measured in the replay viewer.
**Done when:** cube z climbs while gripper-cube distance stays at contact,
checkpoint + curve + replay committed.

### Day 23 — Domain randomization at scale
Randomize mass, friction, and cube spawn (the production Day 12) on the
working grasp policy; measure what robustness costs in convergence.
**Done when:** clean vs. randomized runs are compared on one plot with a
verdict.

### Day 24 — Phase 2 retrospective
Evaluate the best checkpoint, and record what the hand-built Phase 1
versions got right and wrong about the real system, in the style of
`Day1_to_12_Architect_Review.md`.
**Done when:** the review is committed and the README status is updated.

---

## Phase 3 (parked, not planned in detail)

Build the **Ethiopian coffee-cherry picker** — the task every mock in this
repo has been rehearsing — as a custom Isaac Lab environment: own scene USD
(Day 5), own observation and reward managers, own success criteria. The
grasping-twin gripper work is its direct prerequisite: a picker that cannot
close its hand cannot pick. Scoping happens after Day 24.

---

## Working agreements (unchanged from Phase 1)

- One day = one commit series, ending green on CI.
- Scripts stay standalone; no cross-file imports.
- Every day gets a module docstring and a mastery write-up.
- Curves or it didn't happen: training claims are backed by committed PNGs
  (or motion replays, which grasping-twin week 3 proved are the stronger
  evidence).
