import numpy as np
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: THE JACOBIAN GRADIENT (DAY 3) ")
print("========================================================\n")

num_robots = 1_000_000
L1 = 1.0  # Length of Link 1
L2 = 1.0  # Length of Link 2

print(f"[+] Spawning {num_robots:,} 2-Link Robots with random angles...")
theta1 = np.random.uniform(-np.pi, np.pi, num_robots)
theta2 = np.random.uniform(-np.pi, np.pi, num_robots)

print("[+] Calculating Partial Derivatives (The 2x2 Jacobian Tensor)...")
start_time = time.time()

# ---------------------------------------------------------
# THE JACOBIAN MATH (Vectorized)
# J = [ ∂X/∂θ1   ∂X/∂θ2 ]
#     [ ∂Y/∂θ1   ∂Y/∂θ2 ]
# ---------------------------------------------------------

# Create the (1000000, 2, 2) shape tensor
J = np.zeros((num_robots, 2, 2))

# ∂X/∂θ1 = -L1*sin(θ1) - L2*sin(θ1 + θ2)
J[:, 0, 0] = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2)

# ∂X/∂θ2 = -L2*sin(θ1 + θ2)
J[:, 0, 1] = -L2 * np.sin(theta1 + theta2)

# ∂Y/∂θ1 = L1*cos(θ1) + L2*cos(θ1 + θ2)
J[:, 1, 0] = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)

# ∂Y/∂θ2 = L2*cos(θ1 + θ2)
J[:, 1, 1] = L2 * np.cos(theta1 + theta2)

# ---------------------------------------------------------
# THE GRADIENT STEP (Predicting Movement)
# If the motors turn exactly 0.01 radians, how far does the hand move?
# ---------------------------------------------------------

# dq (delta q) represents the tiny movement in the joints. Shape: (2, 1)
dq = np.array([[0.01], 
               [0.01]])

# We multiply the Jacobian (2x2) by the joint movement (2x1) 
# dx = J @ dq  --> The predicted hand movement (delta X and delta Y)
dx = J @ dq

end_time = time.time()

benchmark_ms = (end_time - start_time) * 1000
print("========================================================")
print(f" SUCCESS: Jacobian Gradient calculated in {benchmark_ms:.2f} ms.")

print(f"\n Example Robot [0]:")
print(f"  Current Angles: theta1={theta1[0]:.2f}, theta2={theta2[0]:.2f}")
print(f"  Jacobian Matrix:\n{np.round(J[0], 4)}")
print(f"  If motors turn 0.01 rads, Hand shifts by: dX={dx[0, 0, 0]:.4f}m, dY={dx[0, 1, 0]:.4f}m")
print("========================================================\n")
