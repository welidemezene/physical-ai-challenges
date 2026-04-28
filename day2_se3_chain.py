import numpy as np
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: SE(3) KINEMATIC CHAIN (DAY 2) ")
print("========================================================\n")

# 1. SETUP THE ENVIRONMENT
num_robots = 1_000_000
print(f"[+] Spawning {num_robots:,} 2-Link Robots in memory...")

# Generate random angles for Shoulder (theta1) and Elbow (theta2) radians
theta1 = np.random.uniform(-np.pi, np.pi, num_robots)
theta2 = np.random.uniform(-np.pi, np.pi, num_robots)

# 2. CONSTRUCT BATCH TENSORS
print("[+] Constructing (1000000, 4, 4) Homogeneous Matrices...")

# JOINT 1 (SHOULDER): Rotates around Z-axis. Anchored at the World Origin [0,0,0]
T1 = np.zeros((num_robots, 4, 4))
T1[:, 0, 0] = np.cos(theta1);  T1[:, 0, 1] = -np.sin(theta1)
T1[:, 1, 0] = np.sin(theta1);  T1[:, 1, 1] =  np.cos(theta1)
T1[:, 2, 2] = 1.0
T1[:, 3, 3] = 1.0 # The "Homogeneous Anchor"
# Note: T1 Translation column (index 3) remains 0 because the shoulder is at (0,0,0)

# JOINT 2 (ELBOW): Rotates around Z-axis. Translated 1.0m down the X-axis from the Shoulder
T2 = np.zeros((num_robots, 4, 4))
T2[:, 0, 0] = np.cos(theta2);  T2[:, 0, 1] = -np.sin(theta2)
T2[:, 1, 0] = np.sin(theta2);  T2[:, 1, 1] =  np.cos(theta2)
T2[:, 2, 2] = 1.0
T2[:, 3, 3] = 1.0 # The "Homogeneous Anchor"

# Include the Translation of Link 1 (1.0 meter along the X axis)
# This means the Elbow is physically 1 meter away from the Shoulder
T2[:, 0, 3] = 1.0 


# 3. THE "CHAIN OF COMMAND" (VECTORIZED MATRIX MULTIPLICATION)
print("[!] Chaining T1 @ T2 across all robots...")
start_time = time.time()

# In standard Python this would be a 1,000,000-step For-loop.
# In Python NumPy, we calculate all 1 million robot arms at the exact same time.
# T_hand represents the exact coordinate frame of the wrist/hand.
T_hand = T1 @ T2

# To find where the hand is in 3D space, we look at the translation column
# which is the first 3 rows of the 4th column (index 3).
hand_x = T_hand[:, 0, 3]
hand_y = T_hand[:, 1, 3]
hand_z = T_hand[:, 2, 3]

end_time = time.time()


# 4. RESULTS
benchmark_ms = (end_time - start_time) * 1000
print("========================================================")
print(f" SUCCESS: Chain solved for {num_robots:,} robots in {benchmark_ms:.2f} ms.")

# Print the final (X,Y,Z) coordinates of Robot #0 just to prove it works
print(f" Sample Output (Robot 0): X={hand_x[0]:.2f}, Y={hand_y[0]:.2f}, Z={hand_z[0]:.2f}")
print("========================================================\n")
