import numpy as np
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: BATCH KINEMATICS ENGINE (DAY 1) ")
print("========================================================\n")

# 1. SETUP THE ENVIRONMENT
# We are creating 1,000,000 spatial points (Tensors) in memory.
# In a real robot, this could represent 1 million simulation states.
num_points = 1_000_000

# np.random.rand creates a massive grid of numbers.
# Shape: (1000000 rows, 3 columns). Each row is an [X, Y, Z] coordinate.
print(f"[+] Generating {num_points:,} 3D points in memory...")
points = np.random.rand(num_points, 3) 
print("    Done.\n")


# 2. DEFINE THE SE(3) TRANSFORMATION
# We want to mathematically rotate all 1 million points by 90 degrees around the Z-axis.
angle = np.radians(90)

# This is the 3x3 Rotation part of an SE(3) matrix.
# In Three.js this is hidden inside object.rotation. Here, we build it raw.
rotation_matrix_z = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle),  np.cos(angle), 0],
    [0,              0,             1]
])

print("[+] Z-Axis Rotation Matrix Constructed:")
print(np.round(rotation_matrix_z, 2))
print("\n")


# 3. BENCHMARK #1: THE BEGINNER WAY (Python For-Loop)
print("[!] 1. Initiating Standard Python For-Loop calculation...")
start_time_loop = time.time()

# We force Python to calculate each point one by one (this is slow!)
rotated_points_loop = np.zeros_like(points)
for i in range(num_points):
    rotated_points_loop[i] = rotation_matrix_z @ points[i]
    
end_time_loop = time.time()
loop_benchmark_ms = (end_time_loop - start_time_loop) * 1000
print(f"    Done: Took {loop_benchmark_ms:.2f} milliseconds.\n")


# 4. BENCHMARK #2: THE "MERCENARY" WAY (Vectorized Tensor Batching)
print("[!] 2. Initiating Vectorized Tensor Multiplication (C++ BLAS)...")
start_time_vec = time.time()

# We pass the entire block of 1 million points directly in one operation using .T
rotated_points_vec = points @ rotation_matrix_z.T

end_time_vec = time.time()
vec_benchmark_ms = (end_time_vec - start_time_vec) * 1000
print(f"    Done: Took {vec_benchmark_ms:.2f} milliseconds.\n")


# 5. THE RESULTS
speedup = loop_benchmark_ms / vec_benchmark_ms

print("========================================================")
print(f" RESULTS: 1,000,000 Spatial Points Processed")
print(f" For-Loop Speed:   {loop_benchmark_ms:.2f} ms")
print(f" Vectorized Speed: {vec_benchmark_ms:.2f} ms")
print(f" -> Tensors are {speedup:.1f}X faster than Loops.")
print("========================================================\n")
