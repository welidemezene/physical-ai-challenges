import torch
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: DOMAIN RANDOMIZATION (DAY 12) ")
print("========================================================\n")

def apply_sensor_noise(observations, noise_level=0.05):
    """
    The Stochastic Hardener:
    Takes the "Perfect" observations from the simulator and adds Gaussian Noise
    to simulate real-world broken/dusty sensors.
    
    Args:
        observations: Tensor of shape (num_robots, obs_dim)
        noise_level: How much chaos to add (Standard Deviation)
    """
    # 1. Generate the Chaos Tensor
    # torch.randn_like creates a tensor of the exact same shape as `observations`,
    # filled with random numbers from a Normal (Gaussian) Distribution (Mean 0, Variance 1)
    chaos_tensor = torch.randn_like(observations)
    
    # 2. Scale the Chaos
    scaled_chaos = chaos_tensor * noise_level
    
    # 3. Add the Chaos to Reality (The "Anti-Fragile" fusion)
    noisy_observations = observations + scaled_chaos
    
    return noisy_observations, scaled_chaos

# --- THE SIM-TO-REAL BENCHMARK ---
num_robots = 1_000_000
obs_dim = 14  # 7 angles, 7 velocities

print(f"[+] Booting up {num_robots:,} 'Perfect' Observation Tensors...")
# Imagine all robots are standing perfectly still at angle 1.0 rad
perfect_obs = torch.ones(num_robots, obs_dim)

print(f"[!] Injecting Gaussian Sensor Noise (Noise Level = 5%)...")
start_time = time.time()

noisy_obs, applied_chaos = apply_sensor_noise(perfect_obs, noise_level=0.05)

end_time = time.time()

# --- RESULTS ---
benchmark_ms = (end_time - start_time) * 1000

print("========================================================")
print(f" SUCCESS: Randomized 1,000,000 Data Streams in {benchmark_ms:.3f} ms.")

print("\n Example Robot [0] Sensor Output (Joint 0):")
print(f"  Perfect Simulator Reality: {perfect_obs[0, 0].item():.4f} rad")
print(f"  What the Robot Actually Sees: {noisy_obs[0, 0].item():.4f} rad")
print(f"  The Sensor Glitch (Error): {applied_chaos[0, 0].item():.4f} rad")
print("========================================================\n")

# --- THE CEO CHALLENGE ---
# If you set noise_level to 1.0 (100% noise), the robot will see "3.5 rad" when it is 
# actually at "1.0 rad". It will violently jerk its arm to fix an error that doesn't exist,
# and it will destroy the motor. This is why filtering and tuning the noise_level is the
# hardest part of Sim-to-Real Engineering.
