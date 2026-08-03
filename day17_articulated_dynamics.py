"""Day 17 — Articulated body dynamics.

Step a million 6-joint arms through the equation of motion
M(q) * q_ddot = tau - C - g: build a simplified diagonal mass matrix from
cumulative link inertia, solve forward dynamics for joint accelerations, and
integrate with semi-implicit Euler. A miniature of Featherstone's ABA, the
O(N) algorithm that lets Isaac Lab step thousands of articulated robots per
GPU tick.
"""

import torch
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: ARTICULATED BODY ALGORITHM (DAY 17) ")
print("========================================================\n")

def compute_mass_matrix(joint_masses, joint_lengths, num_joints=6):
    """
    Approximate the Joint-Space Mass Matrix (M) for an articulated robot arm.

    In real physics (Featherstone's ABA), this is computed recursively up the
    kinematic chain. Here we build a simplified diagonal approximation to show
    how inertia grows with distance from the base joint.

    Args:
        joint_masses  (Tensor): Mass of each link (Shape: N, num_joints)
        joint_lengths (Tensor): Length of each link in meters (Shape: N, num_joints)
        num_joints    (int):    Number of joints in the chain

    Returns:
        M (Tensor): Diagonal mass matrix entries per robot (Shape: N, num_joints)
    """
    # Inertia of each link modeled as a thin rod: I = (1/3) * m * L^2
    # Joints further from the base accumulate inertia from all links above them
    link_inertia = (1.0 / 3.0) * joint_masses * (joint_lengths ** 2)

    # Cumulative sum: joint 3 feels inertia of links 1+2+3
    M = torch.cumsum(link_inertia, dim=1)
    return M


def forward_dynamics(M, tau, C, g_torque):
    """
    Forward Dynamics: Given torques, what is the joint acceleration?

    The Equation of Motion for a robot arm:
        M(q) * q_ddot = tau - C(q, q_dot) - g(q)

    Solving for q_ddot (joint accelerations):
        q_ddot = M_inv * (tau - C - g)

    Args:
        M        (Tensor): Mass matrix diagonal (Shape: N, num_joints)
        tau      (Tensor): Motor torques applied at each joint (Shape: N, num_joints)
        C        (Tensor): Coriolis & centrifugal forces (Shape: N, num_joints)
        g_torque (Tensor): Gravity compensation torques (Shape: N, num_joints)

    Returns:
        q_ddot (Tensor): Joint angular accelerations in rad/s^2 (Shape: N, num_joints)
    """
    net_torque = tau - C - g_torque

    # Avoid division by zero — M should always be positive definite
    M_safe = torch.clamp(M, min=1e-6)

    q_ddot = net_torque / M_safe
    return q_ddot


def integrate_joints(q, q_dot, q_ddot, dt=0.01):
    """
    Semi-implicit Euler Integration — one physics timestep.

    Updates velocity first, then position using the new velocity.
    This is more stable than explicit Euler for oscillatory systems.

    Args:
        q      (Tensor): Joint angles in radians (Shape: N, num_joints)
        q_dot  (Tensor): Joint velocities in rad/s (Shape: N, num_joints)
        q_ddot (Tensor): Joint accelerations in rad/s^2 (Shape: N, num_joints)
        dt     (float):  Timestep in seconds

    Returns:
        q_new, q_dot_new (Tensor, Tensor): Updated state
    """
    q_dot_new = q_dot + q_ddot * dt
    q_new     = q + q_dot_new * dt
    return q_new, q_dot_new


# ─────────────────────────────────────────────
#  BENCHMARK: 1,000,000 Robot Arms, 6 Joints
# ─────────────────────────────────────────────
num_robots = 1_000_000
num_joints = 6

print(f"[+] Spawning {num_robots:,} articulated robot arms ({num_joints} joints each)...")

# Random link properties (kg, meters)
joint_masses  = torch.empty(num_robots, num_joints).uniform_(0.5, 3.0)
joint_lengths = torch.empty(num_robots, num_joints).uniform_(0.2, 0.5)

# Motor torques (Nm) — what the PPO policy outputs
tau = torch.empty(num_robots, num_joints).uniform_(-50.0, 50.0)

# Coriolis + gravity approximations (simplified)
C        = torch.empty(num_robots, num_joints).uniform_(-5.0, 5.0)
g_torque = torch.empty(num_robots, num_joints).uniform_(0.0, 15.0)

# Initial joint state
q     = torch.zeros(num_robots, num_joints)
q_dot = torch.zeros(num_robots, num_joints)

print(f"[!] Running Articulated Body Forward Dynamics (ABA) across all agents...")
start_time = time.time()

# Step 1: Build the mass matrix
M = compute_mass_matrix(joint_masses, joint_lengths, num_joints)

# Step 2: Solve for joint accelerations
q_ddot = forward_dynamics(M, tau, C, g_torque)

# Step 3: Integrate one physics timestep (dt = 10ms)
q_new, q_dot_new = integrate_joints(q, q_dot, q_ddot, dt=0.01)

end_time = time.time()

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
benchmark_ms = (end_time - start_time) * 1000

print("========================================================")
print(f" SUCCESS: Stepped {num_robots:,} arms through 1 physics tick in {benchmark_ms:.3f} ms.")
print(f"\n Articulated Dynamics Report (Robot[0], Joint[0..5]):")
print(f"  Applied Torques (tau):      {tau[0].tolist()}")
print(f"  Mass Matrix Diagonal (M):   {[f'{v:.3f}' for v in M[0].tolist()]}")
print(f"  Joint Accelerations q_ddot: {[f'{v:.3f}' for v in q_ddot[0].tolist()]} rad/s^2")
print(f"  New Joint Angles q:         {[f'{v:.4f}' for v in q_new[0].tolist()]} rad")
print(f"  New Joint Velocities q_dot: {[f'{v:.4f}' for v in q_dot_new[0].tolist()]} rad/s")
print("========================================================\n")

# --- THE ARCHITECT'S NOTE ---
# Isaac Lab uses Featherstone's full Articulated Body Algorithm (ABA) in O(N) time
# per chain (not O(N^3) like naive matrix inversion). This is why a 37-DOF humanoid
# can be simulated in parallel across 4096 environments at 100,000 Hz on a single GPU.
