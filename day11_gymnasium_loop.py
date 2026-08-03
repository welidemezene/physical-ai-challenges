"""Day 11 — Vectorized environment loop.

Implement the Gymnasium reset()/step() heartbeat for a million pendulums as
pure tensor math: semi-implicit physics, a dense reward, done flags, and —
the key trick — partial resets that re-initialize only the robots that died
while the survivors keep stepping. Benchmarks the loop in frames per second,
the metric GPU simulators live by.
"""

import torch
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: VECTORIZED GYMNASIUM LOOP (DAY 11)")
print("========================================================\n")

class MercenaryEnv:
    def __init__(self, num_robots=1_000_000):
        self.num_robots = num_robots
        self.dt = 0.016  # 1/60th of a second per step
        self.gravity = 9.81
        
        # We need to store the State (Position and Velocity) of all 1,000,000 robots
        self.positions = torch.zeros(self.num_robots)
        self.velocities = torch.zeros(self.num_robots)
        
        # Start the simulation by randomly placing the robots
        self.reset()
        print(f"[+] Initialized Isaac-Style Environment for {self.num_robots:,} parallel robots.")

    def reset(self, env_ids=None):
        """ 
        PARTIAL RESETTING: The secret to 4090 performance.
        If env_ids is None, reset EVERYONE (Full Reset).
        If env_ids is a list (e.g., [5, 10, 99]), reset ONLY those specific robots.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_robots)
            
        # Reset the chosen robots to a random starting angle and zero velocity
        self.positions[env_ids] = torch.empty(len(env_ids)).uniform_(-3.14, 3.14)
        self.velocities[env_ids] = 0.0
        
        return self._get_observations()

    def _get_observations(self):
        # The AI needs to see Position and Velocity to make a decision
        return torch.stack([self.positions, self.velocities], dim=-1)

    def step(self, actions):
        """
        The Core Physics Loop:
        1. Apply Torque -> Changes Velocity
        2. Apply Velocity -> Changes Position
        """
        # 1. Physics Engine (Differentiable)
        # v_new = v_old + (torque - gravity_effect) * dt
        gravity_torque = self.gravity * torch.sin(self.positions)
        self.velocities = self.velocities + (actions - gravity_torque) * self.dt
        
        # p_new = p_old + v_new * dt
        self.positions = self.positions + self.velocities * self.dt
        
        # 2. The Reward (Dense)
        # We want the pendulum to stand straight up (Position = 0.0, Velocity = 0.0)
        dist_to_target = torch.abs(self.positions)
        rewards = -dist_to_target - 0.1 * torch.abs(self.velocities)
        
        # 3. The "Done" Flag (Death Condition)
        # If the pendulum spins way out of control (past 5.0 radians), it "Dies".
        dones = torch.abs(self.positions) > 5.0
        
        # 4. The Architect's Secret: Automatic Partial Reset
        # Find exactly which robots died, and reset ONLY them immediately.
        dead_robot_ids = torch.where(dones)[0]
        if len(dead_robot_ids) > 0:
            self.reset(dead_robot_ids)
            
        return self._get_observations(), rewards, dones

# --- THE 1,000 TIMESTEP BENCHMARK ---
env = MercenaryEnv()

# Mock Brain: The Actor outputs random torques for now
print("[!] Starting 1,000 timestep physics simulation...")
start_time = time.time()

total_resets = 0
for step in range(1000):
    # The brain calculates 1,000,000 actions at once
    # Let's pretend the brain outputs random torques between -5.0 and +5.0
    mock_actions = torch.empty(env.num_robots).uniform_(-5.0, 5.0)
    
    # The Physics engine steps forward 1/60th of a second
    obs, rewards, dones = env.step(mock_actions)
    
    # Keep track of how many robots died and had to be reset
    total_resets += dones.sum().item()

end_time = time.time()

# --- RESULTS ---
benchmark_ms = (end_time - start_time) * 1000
fps = (1000 * env.num_robots) / (end_time - start_time)

print("========================================================")
print(f" SUCCESS: Simulated 1,000 steps for 1,000,000 robots.")
print(f" Total Time: {benchmark_ms:.2f} ms")
print(f" Simulator Speed: {fps:,.0f} Frames Per Second (FPS)")
print(f" Total Partial Resets Handled: {total_resets:,}")
print("========================================================\n")
