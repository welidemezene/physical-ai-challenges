import torch
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: VECTORIZED REWARD SHAPING (DAY 6) ")
print("========================================================\n")

num_robots = 1_000_000

# 1. THE STATE TENSORS
print(f"[+] Initializing memory for {num_robots:,} agents...")
# dist_prev: How far were they 1 millisecond ago? (0.5m to 2.0m)
dist_prev = torch.empty(num_robots).uniform_(0.5, 2.0)

# dist_current: How far are they right now? 
# (Some moved closer, some moved further away)
dist_current = dist_prev + torch.empty(num_robots).uniform_(-0.05, 0.05)

# effort: How much torque did they apply to the motors? (0 to 10 Nm)
effort = torch.empty(num_robots).uniform_(0.0, 10.0)

start_time = time.time()

# 2. THE POTENTIAL-BASED REWARD (The Magnetic Pull)
# Rt = Φ(s_t) - Φ(s_t-1)
# Potential Φ is negative distance. Moving from -2.0 to -1.9 is +0.1 reward!
# This rewards PROGRESS, not just absolute position.
print("[!] Calculating Dense Reward Tensors (Progress vs. Effort)...")
progress_reward = (-dist_current) - (-dist_prev)

# Penalize high effort to encourage smooth, efficient movements
effort_penalty = -0.1 * effort 

# 3. THE SPARSE BONUS (The Final Goal)
# If a robot reaches within 1cm (0.01m), give it a massive +100 bonus.
# We use `torch.where` to vectorize the 'if' statement without slow for-loops.
print("[!] Injecting Sparse Success Bonuses...")
success_bonus = torch.where(dist_current < 0.01, torch.tensor(100.0), torch.tensor(0.0))

# 4. TOTAL REWARD CALCULATION
final_reward = progress_reward + effort_penalty + success_bonus

end_time = time.time()

# 5. THE RESULTS
benchmark_ms = (end_time - start_time) * 1000
print("========================================================")
print(f" SUCCESS: Evaluated {num_robots:,} Brains in {benchmark_ms:.2f} ms.")

# Find a successful robot to print out
success_indices = torch.where(dist_current < 0.01)[0]
if len(success_indices) > 0:
    idx = success_indices[0].item()
    print(f"\n Example Robot [Index {idx}] (SUCCESSFUL PICK):")
    print(f"  Distance: {dist_prev[idx]:.3f}m -> {dist_current[idx]:.3f}m")
    print(f"  Effort Used: {effort[idx]:.2f} Nm")
    print(f"  Total Reward: {final_reward[idx]:.2f}")
else:
    print(f"\n Example Robot [0] (IN PROGRESS):")
    print(f"  Distance: {dist_prev[0]:.3f}m -> {dist_current[0]:.3f}m")
    print(f"  Effort Used: {effort[0]:.2f} Nm")
    print(f"  Total Reward: {final_reward[0]:.2f}")
print("========================================================\n")
