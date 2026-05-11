import torch
import torch.nn as nn
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: PPO CLIPPED OBJECTIVE (DAY 14) ")
print("========================================================\n")

# --- PPO CORE MATH ---
def compute_ppo_loss(
    old_log_probs,    # The Brain's confidence in the action BEFORE the update
    new_log_probs,    # The Brain's confidence in the action AFTER the update
    advantages,       # Day 10: Was this action better or worse than expected?
    clip_epsilon=0.2  # The "Safety Cage" - how much the brain is allowed to change
):
    """
    PPO Clipped Objective.
    This is the exact loss function used by NVIDIA Isaac Lab.
    
    The Core Rule:
    "You may update your brain's weights, but you are NOT allowed to
    make a change so big that your behavior becomes completely unrecognizable."
    """
    # 1. COMPUTE THE PROBABILITY RATIO
    # log_prob_new - log_prob_old is equivalent to log(pi_new / pi_old)
    # If the ratio = 1.0, the brain didn't change. 
    # If ratio = 2.0, the brain doubled its confidence in this action.
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # 2. THE RAW POLICY GRADIENT (Day 9 Math)
    # If the action was good (positive advantage), scale up its probability.
    # If it was bad (negative advantage), scale it down.
    raw_loss = ratio * advantages
    
    # 3. THE PPO SAFETY CAGE (The "Clip")
    # torch.clamp() forces the ratio to stay between 0.8 and 1.2
    # (i.e. 1 ± clip_epsilon where epsilon = 0.2)
    # This prevents the brain from making reckless, giant updates.
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    clipped_loss = clipped_ratio * advantages
    
    # 4. THE FINAL PPO OBJECTIVE
    # We take the MINIMUM of the raw and clipped losses.
    # "If the raw loss is too greedy, we use the clipped (conservative) version."
    ppo_loss = -torch.min(raw_loss, clipped_loss).mean()
    
    return ppo_loss, ratio

# --- THE TRAINING SCENARIO ---
num_robots = 1_000_000

print(f"[+] Simulating PPO update for {num_robots:,} parallel agents...")

# Simulate the "old" policy that generated the experience
old_log_probs = torch.empty(num_robots).uniform_(-2.0, -0.1)

# Simulate the new policy after 1 gradient step
# (Slightly different from old, but most are similar)
new_log_probs = old_log_probs + torch.empty(num_robots).uniform_(-0.3, 0.3)

# Simulate the Advantages from the Critic (Day 10)
advantages = torch.empty(num_robots).uniform_(-5.0, 10.0)

start_time = time.time()
ppo_loss, ratios = compute_ppo_loss(old_log_probs, new_log_probs, advantages)
end_time = time.time()

# --- RESULTS ---
benchmark_ms = (end_time - start_time) * 1000

# Measure how many robots were clipped (safely constrained)
clipped_fraction = ((ratios < 0.8) | (ratios > 1.2)).float().mean().item()

print("========================================================")
print(f" SUCCESS: PPO Loss computed for {num_robots:,} agents in {benchmark_ms:.3f} ms.")
print(f"\n PPO Metrics:")
print(f"  Final PPO Loss: {ppo_loss.item():.4f}")
print(f"  Safety Clipping Rate: {clipped_fraction * 100:.1f}% of robots were constrained")
print("  (A healthy clip rate is 5-20%. Too low = not learning. Too high = unstable.)")
print("========================================================\n")
