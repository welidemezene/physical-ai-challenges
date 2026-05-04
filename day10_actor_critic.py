import torch
import torch.nn as nn
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: ACTOR-CRITIC ARCHITECTURE (DAY 10) ")
print("========================================================\n")

# 1. ARCHITECTING THE DUAL-HEADED BRAIN
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=14, action_dim=7):
        super(ActorCritic, self).__init__()
        
        # A. THE SHARED TRUNK (The Feature Extractor)
        # Both the Actor and the Critic need to understand physics. 
        # By sharing these layers, they "teach" each other what physical stability looks like.
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # B. THE ACTOR HEAD (The Operator)
        # Job: "What action should I take?"
        # Output: 7 Torques for the 7 motors.
        self.actor = nn.Linear(256, action_dim)
        
        # C. THE CRITIC HEAD (The Auditor/Judge)
        # Job: "How good is this exact moment?"
        # Output: 1 single scalar (The expected future reward).
        self.critic = nn.Linear(256, 1)

    def forward(self, observations):
        # 1. Reality enters the trunk and becomes "Latent Features"
        latent = self.trunk(observations)
        
        # 2. The Features split into two different "Thoughts"
        action_mean = self.actor(latent)
        value = self.critic(latent)
        
        return action_mean, value

# 2. INITIALIZING THE AI
print("[+] Booting up the Actor-Critic Brain...")
brain = ActorCritic()

# 3. THE BATCH SIMULATION
num_robots = 1000
input_dim = 14

print(f"[+] Simulating Observation Tensors for {num_robots:,} parallel robots...")
mock_observations = torch.randn(num_robots, input_dim)

print(f"    Input Shape (O): {mock_observations.shape} -> [Robots, Sensors]\n")

# 4. FORWARD PASS
start_time = time.time()
predicted_actions, predicted_values = brain(mock_observations)
end_time = time.time()

# 5. THE RESULTS
benchmark_ms = (end_time - start_time) * 1000
print("========================================================")
print(f" SUCCESS: Dual-Brains Processed {num_robots:,} Agents in {benchmark_ms:.3f} ms.")

print("\n SHAPE VERIFICATION:")
if predicted_actions.shape == (num_robots, 7):
    print(" [+] Actor Head (Actions): PASSED (1000, 7)")
else:
    print(" [-] Actor Head (Actions): FAILED")

if predicted_values.shape == (num_robots, 1):
    print(" [+] Critic Head (Values): PASSED (1000, 1)")
else:
    print(" [-] Critic Head (Values): FAILED")

print("\n Example Robot [0]:")
print(f"  Critic predicted this state is worth: {predicted_values[0].item():.4f} Points")
print("========================================================\n")
