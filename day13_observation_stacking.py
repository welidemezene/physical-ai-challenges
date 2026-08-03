"""Day 13 — Temporal observation stacking.

Give the agent short-term memory: a rolling buffer keeps the last 5
observation frames per robot and flattens (N, 5, 14) into the (N, 70) tensor
an MLP can read. A single frame cannot show motion — stacking history is how
a feed-forward network perceives velocity, direction, and change.
"""

import torch
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: TEMPORAL INTELLIGENCE (DAY 13) ")
print("========================================================\n")

class TemporalBuffer:
    def __init__(self, num_robots=1_000_000, obs_dim=14, history_length=5):
        """
        The Rolling Memory Buffer
        Instead of the robot just seeing the "Eternal Now", it sees the last 5 frames.
        """
        self.num_robots = num_robots
        self.obs_dim = obs_dim
        self.history_length = history_length
        
        # Shape: (1,000,000 robots, 5 frames, 14 sensors)
        self.buffer = torch.zeros(self.num_robots, self.history_length, self.obs_dim)
        
    def add_observation(self, new_obs):
        """
        Shifts the history back by 1 frame, and adds the new frame to the front.
        This is a highly optimized rolling window operation on the GPU.
        """
        # 1. Shift old memories into the past
        # We take frames 0 to 3, and move them to slots 1 to 4. 
        # (The oldest memory in slot 4 gets deleted)
        self.buffer[:, 1:] = self.buffer[:, :-1].clone()
        
        # 2. Insert the brand new memory into the "Present" (slot 0)
        self.buffer[:, 0] = new_obs
        
    def get_stacked_tensor(self):
        """
        The MLP Brain (Day 8) cannot read 3D arrays (Robots, History, Sensors).
        It needs a flat 2D array. So we "Flatten" the 5 frames of 14 sensors 
        into a single massive array of 70 sensors.
        """
        # Flattens (1000000, 5, 14) -> (1000000, 70)
        stacked_obs = self.buffer.view(self.num_robots, -1)
        return stacked_obs

# --- THE TEMPORAL BENCHMARK ---
num_robots = 1_000_000
obs_dim = 14
frames_of_history = 5

print(f"[+] Initializing Memory Buffer for {num_robots:,} parallel agents...")
memory_buffer = TemporalBuffer(num_robots, obs_dim, frames_of_history)

print(f"[!] Simulating 10 timesteps of observation gathering...")
start_time = time.time()

for step in range(10):
    # Simulate a new observation coming from the physics engine
    # Each step, the robot gets slightly closer to the target
    mock_new_observation = torch.ones(num_robots, obs_dim) * step
    
    # Store it in the memory buffer
    memory_buffer.add_observation(mock_new_observation)

end_time = time.time()

# Get the final flattened tensor to feed to the Actor-Critic
final_tensor_for_brain = memory_buffer.get_stacked_tensor()

# --- RESULTS ---
benchmark_ms = (end_time - start_time) * 1000

print("========================================================")
print(f" SUCCESS: Processed 10 memory cycles in {benchmark_ms:.3f} ms.")
print(f"\n SHAPE VERIFICATION:")
print(f"  Internal Buffer Shape: {memory_buffer.buffer.shape} -> (Robots, Frames, Sensors)")
print(f"  Final Brain Input Shape: {final_tensor_for_brain.shape} -> (Robots, Total Sensors)")

print(f"\n Example Robot [0] Final Flattened Input:")
# Since it stores the present at index 0, and past at index 1, 2, 3...
# You will see the numbers counting DOWN: 9, 8, 7, 6, 5
print(f"  {final_tensor_for_brain[0, 0:5].tolist()}... (Showing first 5 values)")
print("========================================================\n")
