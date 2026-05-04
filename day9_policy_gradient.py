import torch

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: POLICY GRADIENT (DAY 9) ")
print("========================================================\n")

# 1. SIMULATING A BATCH OF EXPERIENCE
# Imagine 5 robots just attempted to pick a coffee cherry.
num_robots = 5

# The Rewards: How well did they do?
# Robot 0 got +10 (Success!), Robot 1 got -5 (Failed), etc.
rewards = torch.tensor([10.0, -5.0, 2.0, 8.0, -10.0])

# The Log-Probabilities: How confident was the brain in the action it took?
# (These are negative numbers because the log of a probability (0.0 to 1.0) is negative)
log_probs = torch.tensor([-0.2, -1.5, -0.8, -0.3, -2.1], requires_grad=True)

# Entropy: How much "Randomness/Curiosity" did the brain have when choosing?
# Higher entropy means the robot was exploring.
entropy = torch.tensor([1.2, 0.5, 0.9, 1.1, 0.3], requires_grad=True)

print(f"[+] Loaded Experience Batch for {num_robots} robots.")
print(f"    Rewards:   {rewards.tolist()}")
print(f"    Log Probs: {log_probs.tolist()}")
print(f"    Entropy:   {entropy.tolist()}\n")


# 2. THE INVENTOR'S TASK: COMPUTING THE LOSS
def compute_loss(log_probs, rewards, entropy):
    # A. The Policy Loss: (Log Probability * Reward)
    # If reward is POSITIVE, we want to maximize the log_prob.
    # If reward is NEGATIVE, we want to minimize the log_prob.
    # PyTorch optimizers always MINIMIZE. So we put a negative sign (-) in front.
    policy_loss = -(log_probs * rewards).mean()
    
    # B. The Entropy Bonus (The Curiosity Tax)
    # We want to maximize entropy to keep the robot exploring.
    # So we MINIMIZE negative entropy.
    entropy_weight = 0.01
    entropy_loss = -entropy_weight * entropy.mean()
    
    # C. Total Loss
    total_loss = policy_loss + entropy_loss
    
    return total_loss, policy_loss, entropy_loss

# 3. RUNNING THE GRADIENT CALCULATOR
print("[!] Calculating Policy Gradient Loss...")
total_loss, policy_loss, entropy_loss = compute_loss(log_probs, rewards, entropy)

print("========================================================")
print(f" SUCCESS: Loss Calculated.")
print(f"  Policy Loss:  {policy_loss.item():.4f}")
print(f"  Entropy Loss: {entropy_loss.item():.4f}")
print(f"  TOTAL LOSS:   {total_loss.item():.4f}")

# 4. TRIGGERING BACKPROPAGATION
# This is where the magic happens. The Loss travels BACKWARD through the brain
# to update the weights of the MLP you built on Day 8.
total_loss.backward()

print("  Backpropagation Triggered. Gradients updated.")
print("========================================================\n")
