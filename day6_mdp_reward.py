import numpy as np

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: MARKOV DECISION PROCESS (DAY 6) ")
print("========================================================\n")

# 1. THE STATE (S) - The Snapshot of Reality
# The Robot is trying to pick a Coffee Cherry.
distance_to_cherry = 2.0  # meters
force_on_branch = 0.0     # Newtons
energy_used = 15.0        # Joules

print(f"[+] Current State Snapshot (S):")
print(f"    Distance to Target: {distance_to_cherry}m")
print(f"    Force on Branch:    {force_on_branch}N")
print(f"    Energy Consumed:    {energy_used}J\n")


# 2. REWARD SHAPING (The Inventor's Secret)
# If we only give a reward when distance == 0, the AI will never learn.
# We must "Shape" the reward so the AI feels a "magnetic pull" toward the cherry.

def calculate_reward(distance, force, energy):
    # WEIGHTS (W) - How important is each factor?
    w_distance = 10.0   # Highest priority: Get close to the cherry
    w_force = 5.0       # High priority: Do not snap the branch
    w_energy = 0.1      # Low priority: Save battery
    
    # 1. Distance Penalty (Closer = Less Negative Penalty)
    # By making the reward negative based on distance, the AI naturally 
    # learns that moving closer INCREASES its score.
    reward_distance = -w_distance * distance
    
    # 2. Damage Penalty
    # If the robot pulls too hard (> 10 Newtons), we severely punish it.
    reward_force = 0
    if force > 10.0:
        reward_force = -w_force * (force - 10.0)
        
    # 3. Energy Penalty (Prevents the robot from shaking violently)
    reward_energy = -w_energy * energy
    
    # 4. The "Sparse" Success Reward
    # A massive bonus only given when the task is perfectly completed.
    reward_success = 0
    if distance < 0.05 and force <= 10.0:
        reward_success = 1000.0  
        
    # THE FINAL MDP REWARD CALCULATION
    total_reward = reward_distance + reward_force + reward_energy + reward_success
    return total_reward

# 3. SIMULATING 3 ACTIONS (A)
print("[!] Evaluating 3 Possible Actions (A) through the Policy (π)...\n")

# Action 1: Do Nothing (Distance stays 2.0m, uses 5J energy)
r1 = calculate_reward(2.0, 0.0, 5.0)
print(f" Action 1 (Do Nothing):      Reward = {r1:.2f}")

# Action 2: Rip the Cherry off violently (Distance 0m, Force 50N, uses 50J)
r2 = calculate_reward(0.0, 50.0, 50.0)
print(f" Action 2 (Violent Pull):    Reward = {r2:.2f}")

# Action 3: Gentle Approach (Distance 0.5m, Force 2N, uses 10J)
r3 = calculate_reward(0.5, 2.0, 10.0)
print(f" Action 3 (Gentle Approach): Reward = {r3:.2f}\n")


print("========================================================")
if r3 > r1 and r3 > r2:
    print(" SUCCESS: The AI Policy chose the Gentle Approach!")
    print(" Reward Shaping successfully guided the robot.")
print("========================================================\n")
