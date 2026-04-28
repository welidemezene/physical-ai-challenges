import torch
import time

print("\n========================================================")
print(" PHYSICAL AI DYNAMICS: DIFFERENTIABLE PHYSICS (DAY 4) ")
print("========================================================\n")

# 1. SETUP THE SCENARIO
h_initial = 5.0  # The ball is dropped from 5 meters
target_h  = 1.2  # The AI wants the ball to bounce exactly to 1.2 meters

# We initialize "bounciness" at a random, incorrect guess (0.1).
# `requires_grad=True` is the "Glass Box" switch. It tells PyTorch: 
# "Track every math operation I do with this variable so we can reverse-engineer it."
bounciness = torch.tensor([0.1], requires_grad=True)

# We use an Optimizer (Gradient Descent) to update the bounciness based on the slope.
# Learning Rate (lr) determines how big of a step the AI takes.
optimizer = torch.optim.SGD([bounciness], lr=0.01)

print(f"[+] Objective: Drop ball from {h_initial}m. Bounce to exactly {target_h}m.")
print(f"[+] Initial naive bounciness guess: {bounciness.item():.4f}\n")

# 2. THE TRAINING LOOP (Backpropagation Through Reality)
start_time = time.time()
epochs = 50

for epoch in range(epochs):
    # --- A. FORWARD PASS (The Physics Engine) ---
    # The physical equation for bounce height: h_final = h_initial * (bounciness^2)
    h_final = h_initial * (bounciness ** 2)
    
    # --- B. THE LOSS (How wrong were we?) ---
    # Mean Squared Error (MSE) Loss
    loss = (h_final - target_h) ** 2
    
    # --- C. THE MAGIC: BACKPROPAGATION ---
    # This single command calculates the partial derivative of the physics equation.
    # "How much should I change bounciness to make loss = 0?"
    loss.backward()
    
    # --- D. THE UPDATE ---
    # The optimizer applies the gradient to change the bounciness value.
    optimizer.step()
    
    # Reset the gradient for the next loop
    optimizer.zero_grad()
    
    # Print progress every 5 loops
    if epoch % 5 == 0 or loss.item() < 0.0001:
        print(f" Loop {epoch:02d} | Bounciness: {bounciness.item():.4f} | Bounce Height: {h_final.item():.4f}m | Loss: {loss.item():.6f}")
        
        # Stop early if we are perfectly hitting the target
        if loss.item() < 0.00001:
            print(f"\n[!] Target Hit Perfectly at Loop {epoch}.")
            break

end_time = time.time()

# 3. RESULTS
print("========================================================")
print(f" SUCCESS: Physics Differentiated in {(end_time - start_time)*1000:.2f} ms.")
print(f" FINAL BOUNCINESS REQUIRED: {bounciness.item():.4f}")
print("========================================================\n")
