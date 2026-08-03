"""Day 15 — Gymnasium API integration.

Assemble the previous days into one professional environment class: action
and observation spaces, reset() with randomized targets, and a step() that
runs physics, shaped reward, sensor noise, termination flags, and Day 11
partial resets in a single pass. The interface is the point — the brain
talks reset()/step() and never touches the physics directly, which is what
lets the same PPO code drive any simulator.
"""

import torch
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: GYMNASIUM API INTEGRATION (DAY 15) ")
print("========================================================\n")

# This is the "Universal Adapter" that connects your Brain to any Robot Body.
# Every simulation framework in the world (Isaac Lab, MuJoCo, Gazebo) follows this pattern.

class EthiopianCoffeePickerEnv:
    """
    The Professional Gymnasium Interface.
    
    This class is the "Middleman" between your PPO Brain (Day 14) 
    and the Physics Engine (Day 4). The brain never talks to physics directly.
    
    Think of it as the "USB Standard."
    - USB doesn't care if you plug in a keyboard or a hard drive.
    - The Brain doesn't care if it's controlling a coffee arm or a walking robot.
    """
    
    def __init__(self, num_envs=1_000_000):
        self.num_envs = num_envs
        
        # --- PILLAR 1: DEFINE THE SPACE (The "Rules of the Game") ---
        
        # ACTION SPACE: What can the robot DO?
        # 7 joint torques, each clamped to [-1.0, 1.0] (normalized).
        # The environment internally scales this to the real motor torque range.
        self.action_dim = 7
        self.action_low = -1.0
        self.action_high = 1.0
        
        # OBSERVATION SPACE: What can the robot SEE?
        # 7 joint angles + 7 joint velocities + 3 target positions + 4 quaternion orientation = 21
        self.obs_dim = 21
        
        # --- PILLAR 2: INITIALIZE THE ROBOT STATE ---
        self.joint_angles = torch.zeros(self.num_envs, 7)
        self.joint_velocities = torch.zeros(self.num_envs, 7)
        self.target_positions = torch.zeros(self.num_envs, 3)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32)
        self.max_episode_length = 500
        
        print(f"[+] Environment initialized: {self.num_envs:,} parallel worlds.")
        print(f"    Action Space:      Box({self.action_dim},) in range [{self.action_low}, {self.action_high}]")
        print(f"    Observation Space: Box({self.obs_dim},) -> Angles + Velocities + Target\n")

    def reset(self, env_ids=None):
        """
        The "New Game" button.
        Puts the robot back at the starting position with random noise (Domain Randomization).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        
        n = len(env_ids)
        
        # 1. Reset joint angles to random starting position
        self.joint_angles[env_ids] = torch.empty(n, 7).uniform_(-0.5, 0.5)
        self.joint_velocities[env_ids] = torch.zeros(n, 7)
        
        # 2. Randomize the target (the coffee cherry's position)
        # This is Domain Randomization (Day 12): the cherry is never in the same place twice
        self.target_positions[env_ids] = torch.empty(n, 3).uniform_(0.3, 1.2)
        
        # 3. Reset episode counter
        self.episode_lengths[env_ids] = 0
        
        return self._get_observations(env_ids)

    def _get_observations(self, env_ids=None):
        """
        Builds the Observation Tensor (Day 7).
        Normalizes raw data + adds Sensor Noise (Day 12) before sending to the Brain.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        
        # Normalize angles to [-1, 1] range
        normalized_angles = self.joint_angles[env_ids] / 3.14
        
        # Clip extreme velocities (Day 7: clamp to [-5, 5])
        clipped_velocities = torch.clamp(self.joint_velocities[env_ids], -5.0, 5.0)
        
        # Calculate distance to target (3D positions)
        target = self.target_positions[env_ids]
        
        # Combine all sensors into one flat observation vector
        obs = torch.cat([normalized_angles, clipped_velocities, target], dim=-1)
        
        # Apply Sensor Noise (Day 12: the robot's senses are imperfect)
        noise = torch.randn_like(obs) * 0.01
        obs = obs + noise
        
        return obs

    def step(self, actions):
        """
        THE HEARTBEAT of the simulation.
        
        This single function integrates:
        - Day 2: Forward Kinematics
        - Day 4: Differentiable Physics (F=ma)
        - Day 6: Reward Function
        - Day 12: Sensor Noise
        """
        # 1. SAFETY: Clip the raw actions to the valid range
        actions = torch.clamp(actions, self.action_low, self.action_high)
        
        # 2. PHYSICS (Day 4): Apply torques -> update velocities -> update positions
        dt = 0.016  # 1/60th second
        gravity_effect = 9.81 * torch.sin(self.joint_angles) * 0.01
        self.joint_velocities = self.joint_velocities + (actions - gravity_effect) * dt
        self.joint_velocities = torch.clamp(self.joint_velocities, -10.0, 10.0)
        self.joint_angles = self.joint_angles + self.joint_velocities * dt
        
        # 3. REWARD (Day 6): How close is the end-effector to the coffee cherry?
        # Simplified: use joint angle sum as proxy for end-effector position
        end_effector_pos = self.joint_angles.sum(dim=-1, keepdim=True).expand(-1, 3)
        distance_to_target = torch.norm(self.target_positions - end_effector_pos, dim=-1)
        
        # Dense reward: reward for being close, penalize for being far
        rewards = 1.0 / (distance_to_target + 0.1) - 0.1 * torch.norm(actions, dim=-1)
        
        # 4. DONE CONDITIONS
        self.episode_lengths += 1
        
        # Robot is "done" if it reaches the target OR runs out of time
        reached_target = distance_to_target < 0.05
        timed_out = self.episode_lengths >= self.max_episode_length
        
        # SAFETY: Done if torque is dangerously high (Safety Engineering)
        unsafe_torque = (torch.abs(actions) > 0.99).any(dim=-1)
        
        terminated = reached_target
        truncated = timed_out | unsafe_torque
        dones = terminated | truncated
        
        # 5. PARTIAL RESET (Day 11): Reset only the dead robots, not all of them
        dead_env_ids = torch.where(dones)[0]
        if len(dead_env_ids) > 0:
            self.reset(dead_env_ids)
        
        # 6. GET NEW OBSERVATIONS for all envs
        obs = self._get_observations()
        
        return obs, rewards, terminated, truncated

# --- THE FULL LOOP BENCHMARK ---
env = EthiopianCoffeePickerEnv(num_envs=10_000)

print("[!] Running 100 steps of the full integrated simulation loop...\n")
obs = env.reset()
start_time = time.time()

for step in range(100):
    # The PPO Brain generates actions (mock random for now)
    actions = torch.empty(env.num_envs, env.action_dim).uniform_(-1.0, 1.0)
    obs, rewards, terminated, truncated = env.step(actions)

end_time = time.time()

benchmark_ms = (end_time - start_time) * 1000
sps = (100 * env.num_envs) / (end_time - start_time)

print("========================================================")
print(f" SUCCESS: Completed 100 steps x {env.num_envs:,} environments.")
print(f" Total Benchmark Time: {benchmark_ms:.2f} ms")
print(f" Throughput: {sps:,.0f} Steps Per Second (SPS)")
print(f" Observation Shape: {obs.shape} -> Ready for PPO Brain")
print("========================================================\n")
