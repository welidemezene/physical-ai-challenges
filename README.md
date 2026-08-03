# Physical AI Challenges

A build-from-scratch study log in **physical AI** — robotics simulation and reinforcement
learning — written one day at a time. Every day pairs a runnable PyTorch script with a
"mastery" write-up explaining the concept behind it, so the repo is both working code and
a record of the reasoning.

The goal is not to call library functions. It is to build the **Reality Engine** underneath
them: the kinematics, the physics, the environment loop, and the learning algorithm.

---

## The Three Pillars

The work is organized around three layers of a physical AI system.

### Pillar 1 — The Body (Physics & Kinematics)
*Days 1–4.* Building the physical machine.

- Vectorized matrix math in PyTorch — no `for` loops, so a million robots step at once.
- **SE(3) homogeneous transforms:** the 4×4 matrix holding rotation (3×3) and translation
  (3×1) that gives forward kinematics.
- **The Jacobian:** the map from desired end-effector motion back to joint velocities
  (inverse kinematics).
- **Differentiable physics:** using autograd as a calculus engine — write `F = ma`, let
  PyTorch produce the gradients.

### Pillar 2 — The World (Simulation & Architecture)
*Days 5, 7, 11, 12, 15.* Building the environment the robot lives in.

- The **Gymnasium loop** — `reset()` / `step(actions)` as the heartbeat of the system.
- **Partial resetting** with `torch.where` — recycle only the finished environments without
  stalling the survivors.
- **Observation filtering** — normalize angles to `[-1, 1]`, `torch.clamp` away velocity
  spikes, inject Gaussian noise to model imperfect sensors.
- **OpenUSD** as the composition layer for scene description.
- **Domain randomization** — randomize gravity, friction, and sensing so the policy is
  anti-fragile rather than overfit to a perfect simulator.

### Pillar 3 — The Brain (Intelligence & Learning)
*Days 6, 8, 9, 10, 13, 14.* Building the network that controls the body.

- **MLP backbone** — sensors in, torques out.
- **Policy gradients** — `loss = -(log_prob * reward)`, the slope that rewires the weights.
- **Dense vs. sparse rewards** — engineering a reward that actually admits learning.
- **The entropy bonus** — paying the agent to keep exploring instead of settling early.
- **Actor–Critic** — the critic's baseline subtracted from actual reward to get the
  *advantage*, so luck isn't mistaken for skill.
- **PPO** — the clipped surrogate objective that keeps updates from destroying the policy.
- **Temporal intelligence** — observation stacking so the agent can perceive motion.

### Contact & Articulated Dynamics
*Days 16–17.* Where the body meets the world.

- **Friction cone** constraints and a contact solver.
- **Articulated body dynamics** and the equation of motion.

---

## Day Index

| Day | Topic | Code | Notes |
|----:|-------|------|-------|
| 1 | Vectorized matrix math | `day1_matrix.py` | |
| 2 | SE(3) transform chains | `day2_se3_chain.py` | `SE3_Math_Primer.md` |
| 3 | The Jacobian | `day3_jacobian.py` | `Jacobian_Primer_Day3.md` |
| 4 | Differentiable physics | `day4_differentiable_physics.py` | `Differentiable_Physics_Primer_Day4.md` |
| 5 | OpenUSD architecture | `day5_usd_script.py` | `day5_usd_architecture.md`, `Day5_Mastery.md` |
| 6 | MDPs & reward design | `day6_mdp_reward.py`, `day6_vectorized_reward.py` | `Day6_MDP_Mastery.md` |
| 7 | Observation spaces | `day7_observation_space.py` | `Day7_Observation_Mastery.md` |
| 8 | Neural policies | `day8_neural_policy.py` | `Day8_DRL_Mastery.md` |
| 9 | Policy gradients | `day9_policy_gradient.py`, `day9_rl_training_loop.py` | `Day9_PolicyGradient_Mastery.md` |
| 10 | Actor–Critic | `day10_actor_critic.py` | `Day10_ActorCritic_Mastery.md` |
| 11 | Gymnasium loop | `day11_gymnasium_loop.py` | `Day11_Gymnasium_Mastery.md` |
| 12 | Domain randomization | `day12_domain_randomization.py` | `Day12_DomainRandomization_Mastery.md` |
| 13 | Observation stacking | `day13_observation_stacking.py` | `Day13_TemporalIntelligence_Mastery.md` |
| 14 | PPO clipped objective | `day14_ppo_clipped.py` | `Day14_PPO_Mastery.md` |
| 15 | Gymnasium API integration | `day15_gymnasium_interface.py` | `Day15_GymnasiumAPI_Mastery.md` |
| 16 | Contact dynamics & friction cone | `day16_friction_solver.py` | `Day16_ContactDynamics_Mastery.md` |
| 17 | Articulated body dynamics | `day17_articulated_dynamics.py` | `Day17_ArticulatedDynamics_Mastery.md` |

A consolidated retrospective covering the first twelve days lives in
[`Day1_to_12_Architect_Review.md`](Day1_to_12_Architect_Review.md), and
[`RL_vs_DiffPhysics_Primer.md`](RL_vs_DiffPhysics_Primer.md) compares the two approaches to
learning control.

---

## Stack

- **PyTorch** — tensors, autograd, and the networks
- **Gymnasium** — environment API
- **OpenUSD** — scene description
- **NVIDIA Isaac Lab** — GPU simulation for the manipulation phase

---

## Running

Each day is a standalone script with no cross-file imports:

```bash
python day14_ppo_clipped.py
```

---

## Status

Days 1–17 complete. Phase 2 — full GPU simulation, training a Franka arm on a
manipulation task in Isaac Lab — is underway in the sibling repo
[grasping-twin](https://github.com/welidemezene/grasping-twin), where a lift policy has
been trained through a nine-stage curriculum and the current frontier is making the
gripper truly grasp rather than bat the cube. The day-by-day plan for recording that
phase here, with a definition of done for each day, lives in [`ROADMAP.md`](ROADMAP.md).
