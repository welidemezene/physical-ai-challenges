import torch
import torch.nn as nn
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: DEEP NEURAL POLICY (DAY 8) ")
print("========================================================\n")

# 1. ARCHITECTING THE BRAIN (The Policy π)
class RobotPolicy(nn.Module):
    def __init__(self):
        super(RobotPolicy, self).__init__()
        
        # The Multi-Layer Perceptron (MLP)
        # Input: 14 (7 joint angles + 7 joint velocities)
        # Hidden 1: 256 neurons
        # Hidden 2: 256 neurons
        # Output: 7 (Torques to apply to the 7 joints)
        self.brain = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),           # The "Spark" - adds non-linear capability
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, observations):
        # This function is called every millisecond in the simulation
        actions = self.brain(observations)
        return actions

# 2. INITIALIZING THE BRAIN
print("[+] Booting up the Neural Network...")
policy = RobotPolicy()

# Optional: Move the brain to the GPU if you had a 4090
# policy = policy.to('cuda')

# 3. THE SIMULATION TEST (The Batch Check)
num_robots = 10
input_dim = 14

# Simulate 10 robots looking at the world (random normalized data)
print(f"[+] Simulating Observation Tensors for {num_robots} robots...")
mock_observations = torch.randn(num_robots, input_dim)

print(f"    Input Shape (O): {mock_observations.shape} -> [Robots, Sensors]")

# Feed the observations into the Brain to get the Actions
start_time = time.time()
predicted_actions = policy(mock_observations)
end_time = time.time()

print(f"    Output Shape (A): {predicted_actions.shape} -> [Robots, Motors]\n")

# 4. RESULTS
benchmark_ms = (end_time - start_time) * 1000
print("========================================================")
print(f" SUCCESS: Brain Processed {num_robots} Agents in {benchmark_ms:.3f} ms.")

# Verify the shapes matched perfectly
if predicted_actions.shape == (10, 7):
    print(" SHAPE VERIFICATION: PASSED (10, 7)")
else:
    print(" SHAPE VERIFICATION: FAILED")

print("========================================================\n")
