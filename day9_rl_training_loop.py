import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: THE TRAINING LOOP (DAY 9) ")
print("========================================================\n")

print("Demystifying the 'Magic' of Reinforcement Learning.")
print("This is the exact loop that runs 1,000 times a second inside Isaac Lab.\n")

# --- MOCK CLASSES TO REPRESENT YOUR PREVIOUS DAYS ---

class Environment:
    """ Day 5 & 7: The Physics Simulator and Sensors """
    def __init__(self):
        self.cherry_distance = 2.0
        
    def get_observation(self):
        # Day 7: The Robot opens its eyes and normalizes data
        return [self.cherry_distance / 10.0] # Normalized
        
    def step(self, action_torque):
        # Day 4: Differentiable Physics simulates the movement
        # If torque is positive, it moves closer to the cherry.
        self.cherry_distance -= (action_torque * 0.1)
        
        # Day 6: The Reward Function (Dense Reward)
        reward = 0.0
        done = False
        
        if self.cherry_distance <= 0.05:
            reward = 100.0  # Massive success bonus!
            done = True
        else:
            reward = -0.1   # Small time penalty for being slow
            
        return self.get_observation(), reward, done

class Brain:
    """ Day 8 & 9: The Neural Network and Policy Gradient """
    def __init__(self):
        self.weights = 0.5  # A very simple mock "Neural Network"
        
    def guess_action(self, observation):
        # Day 8: The MLP calculates the best action (torque)
        # Plus we add Randomness (Entropy) to explore!
        action = (observation[0] * self.weights) + 1.0 # Base action
        return action
        
    def learn(self, reward):
        # Day 9: The Policy Gradient!
        # The Reward acts as a volume knob to update the weights.
        learning_rate = 0.1
        self.weights += (reward * learning_rate)

# --- THE ACTUAL RL TRAINING LOOP ---

env = Environment()
robot_brain = Brain()

epochs = 30
print("[+] Starting the Reinforcement Learning Loop...\n")

for loop in range(epochs):
    # 1. PERCEPTION (Day 7)
    obs = env.get_observation()
    
    # 2. ACTION (Day 8)
    # The brain looks at the observation and fires the motors
    action = robot_brain.guess_action(obs)
    
    # 3. PHYSICS & REWARD (Day 4 & 6)
    # The environment calculates physics and tells the robot if it did a good job
    next_obs, reward, done = env.step(action)
    
    # 4. LEARNING / POLICY GRADIENT (Day 9)
    # The brain rewires its weights based on the reward (The Volume Knob)
    robot_brain.learn(reward)
    
    print(f" Loop {loop+1:02d} | Dist: {env.cherry_distance:.2f}m | Action: {action:.2f}N | Reward: {reward:.1f} | Brain Wgt: {robot_brain.weights:.2f}")
    
    if done:
        print("\n[!] TARGET REACHED! The AI has successfully learned to pick the cherry.")
        break
    
    time.sleep(0.1) # Slowing down so you can read it

print("========================================================\n")
