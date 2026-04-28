import torch
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: OBSERVATION TENSOR (DAY 7) ")
print("========================================================\n")

num_robots = 1_000_000

# 1. RAW SENSOR DATA (The "Lies")
# Let's pretend the sensors are giving us chaotic, raw data
print(f"[+] Gathering raw sensor data for {num_robots:,} robots...")
raw_angles = torch.empty(num_robots).uniform_(-torch.pi, torch.pi) # From -3.14 to 3.14
raw_velocities = torch.empty(num_robots).uniform_(-20.0, 20.0)      # Crazy fast speeds
target_positions = torch.rand(num_robots, 3) * 10                   # Target is 0 to 10m away
hand_positions = torch.rand(num_robots, 3) * 10                     # Hand is 0 to 10m away

# 2. THE FILTER: NORMALIZATION & CLIPPING
print("[!] Filtering and Normalizing data for Neural Network ingestion...\n")
start_time = time.time()

# A. Normalize Angles: [-pi, pi] -> [-1, 1]
# We simply divide by Pi.
normalized_angles = raw_angles / torch.pi

# B. Clip Velocities: Prevent the AI from seeing crazy spikes
# We clamp the data strictly between -5.0 and +5.0
clipped_velocities = torch.clamp(raw_velocities, min=-5.0, max=5.0)

# C. Calculate Relative Distance
# The AI doesn't need to know global coordinates. It just needs to know: 
# "How far is the target from my hand?"
# We use Euclidean distance calculation (L2 Norm)
distances = torch.norm(target_positions - hand_positions, p=2, dim=1)

# 3. THE OBSERVATION TENSOR (O)
# We stack these 3 values together to feed directly into the Neural Network
# Shape will be: (1000000, 3)
observation_tensor = torch.stack([normalized_angles, clipped_velocities, distances], dim=1)

end_time = time.time()

# 4. RESULTS
benchmark_ms = (end_time - start_time) * 1000
print("========================================================")
print(f" SUCCESS: Processed {num_robots:,} Observation Tensors in {benchmark_ms:.2f} ms.")

print(f"\n Example Robot [0] Sensor Output:")
print(f"  Raw Data: Angle={raw_angles[0]:.2f} rad, Vel={raw_velocities[0]:.2f} m/s, Dist={distances[0]:.2f}m")
print(f"  Neural Net Input: {observation_tensor[0].tolist()}")
print("========================================================\n")
