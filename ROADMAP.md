# Roadmap — Phase 2: Isaac Lab

Days 1–17 built every layer of a physical AI system by hand: kinematics,
physics, the environment loop, and the learning algorithm. Phase 2 trades the
hand-built miniatures for the real thing — **NVIDIA Isaac Lab** running GPU
simulation — and trains a **Franka arm on a manipulation task**. The goal of
the phase is a committed, converging training curve produced on this machine.

The rule of the repo does not change: every day still ends with something
runnable plus a mastery write-up. What changes is that the components are no
longer mocks — the physics is PhysX, the arm is a real URDF/USD Franka, and
the PPO is the production implementation whose math Day 14 already derived.

---

## Hardware & setup reality check

- **This machine:** Windows 11, NVIDIA RTX 5070 Ti Laptop GPU (Blackwell).
  Isaac Sim 5.x supports RTX 50-series; training runs headless
  (`--headless`), which is the right mode for a laptop anyway.
- **Two install paths.** Native Windows (Isaac Lab ships `isaaclab.bat`) or
  WSL2 Ubuntu — the plotting script in this repo was already written for the
  headless/WSL workflow. Pick one on Day 18 and write down why.
- **Python version gotcha:** Isaac Sim/Isaac Lab pin a specific Python
  (3.10/3.11 era), not the system 3.14. The install goes into its own
  conda/venv environment; nothing in this repo's Day 1–17 scripts changes.
- **PyTorch gotcha:** Blackwell GPUs need recent CUDA-12.8-class wheels. If
  `torch.cuda.is_available()` is False inside the Isaac environment, the
  wheel is wrong — fix that before blaming the simulator.
- **Disk:** Isaac Sim is a ~50 GB class install. Check free space first.

---

## The days

Each day has a deliverable and a definition of done, same as Phase 1.

### Day 18 — Installation & first light
Install Isaac Sim + Isaac Lab, verify the GPU is visible, and run the
smallest included example (empty stage, then the cartpole training demo).
**Done when:** the cartpole example trains headless for a few hundred
iterations without error, and the mastery note records the chosen install
path and every trap hit along the way.

### Day 19 — Anatomy of a real task
No training. Read the source of the Franka lift task (`Isaac-Lift-Cube-
Franka-v0`): the environment config, observation terms, reward terms, reset
logic, and randomization hooks.
**Done when:** the write-up maps every block of the task config to the day
that built it by hand — observations → Day 7, shaped reward → Day 6, partial
resets → Day 11, sensor/domain noise → Day 12, the env API → Day 15, the
articulated arm → Day 17. This is the payoff day for Phase 1.

### Day 20 — First Franka training run
Train the simple reach task (`Isaac-Reach-Franka-v0`) headless with the
rsl_rl PPO workflow. Read the run back with `plot_training_curves.py` — the
tool this repo built for exactly this moment.
**Done when:** a converging reward-curve PNG is committed alongside the run's
hyperparameters (num_envs, learning rate, iterations, wall-clock time).

### Day 21 — The manipulation task
Train `Isaac-Lift-Cube-Franka-v0` — the phase's headline goal. Expect this
to take real wall-clock time; tune `--num_envs` to what the 5070 Ti fits.
**Done when:** the policy lifts the cube in evaluation, the checkpoint is
saved, and the curve PNG + metrics are committed.

### Day 22 — Reward surgery
Change the reward, retrain, and compare. Add or re-weight one term (e.g. an
energy/action penalty, exactly like Day 6's `w_energy`) and overlay the
before/after curves with the plotter's multi-run mode.
**Done when:** the write-up explains what the modified term changed in the
learned behavior, with both curves as evidence.

### Day 23 — Domain randomization at scale
Turn up Isaac Lab's randomization (mass, friction, observation noise — the
production version of Day 12) and measure what it costs in convergence speed
versus what it buys in robustness.
**Done when:** two runs (clean vs. randomized) are compared on one plot with
a written verdict.

### Day 24 — Phase 2 retrospective
Evaluate the best checkpoint, record what the hand-built Phase 1 versions
got right and wrong about the real system, and write the retrospective in
the style of `Day1_to_12_Architect_Review.md`.
**Done when:** the review is committed and the README status is updated.

---

## Phase 3 (parked, not planned in detail)

Build the **Ethiopian coffee-cherry picker** — the task every mock in this
repo has been rehearsing — as a custom Isaac Lab environment: own scene USD
(Day 5), own observation and reward managers, own success criteria. Then
sim-to-real notes. Scoping happens after Day 24, not before.

---

## Working agreements (unchanged from Phase 1)

- One day = one commit series, ending green on CI.
- Scripts stay standalone; no cross-file imports.
- Every day gets a module docstring and a mastery write-up.
- Curves or it didn't happen: training claims are backed by committed PNGs.
