# Day 13 Mastery: Temporal Intelligence & Observation Stacking

Your robot has a perfect physics engine (Day 4) and a brilliant Actor-Critic brain (Day 10). But up until today, your robot had the memory of a goldfish.

If a robot only looks at the "Current Frame," it suffers from **MDP (Markov Decision Process) Failure**. It lives in an Eternal Now.

---

### Mastery Concept 1: The Goldfish Memory Problem
Imagine throwing a baseball at a robot. 
In Frame 1, the robot takes a picture. It sees a baseball 10 meters away.
Because it has no memory, the robot cannot tell if the baseball is sitting still on the ground, rolling away, or flying directly at its face at 100 km/h. 
**If you only have one frame of data, you cannot calculate velocity or momentum.** 

In the real world, sensors fail. Sometimes a coffee leaf blocks the camera for a split second. If the robot has no memory, it instantly panics and forgets where the coffee cherry is. We call this a **POMDP (Partially Observable Markov Decision Process)**. The robot can only see a "part" of reality.

### Mastery Concept 2: Observation Stacking (The Solution)
To give the robot temporal intelligence without slowing down the GPU, we use **Observation Stacking** (also called Frame Stacking).

Instead of feeding the Brain an array of 14 sensors, we feed the Brain an array of `14 sensors * 5 frames = 70 numbers`.
We show the brain:
1.  Where the robot is *Now*.
2.  Where it was 1 step ago.
3.  Where it was 2 steps ago.
4.  Where it was 3 steps ago.
5.  Where it was 4 steps ago.

By looking at these 5 frames simultaneously, the Neural Network's internal math implicitly "invents" the concept of Velocity and Acceleration. It realizes: *"Wait, in frame 4 the ball was far, in frame 1 the ball was close. Therefore, the ball is moving toward me fast!"*

**Why not use an RNN (Recurrent Neural Network)?**
RNNs or LSTMs have "internal memory states," but they are sequential and slow. When training 1,000,000 parallel robots on an RTX 4090, throughput is everything. Observation Stacking is a pure Matrix Flattening operation (`.view(num_robots, -1)`). It runs at near `O(1)` speed on the GPU.

### Mastery Concept 3: The Timestep Trap (Frame Skipping)
There is a massive trap here.
If your physics simulator runs at 1,000 Hz (1,000 frames per second), the time between Frame 1 and Frame 2 is `0.001 seconds`. 

If you stack 5 frames at `0.001s` apart, you are only giving the robot a memory of `0.005 seconds`. 
That is so fast that the robot barely moved between frames. The difference between the frames is smaller than the **Sensor Noise** (Day 12). The robot will just see vibrating "jitter" instead of a clear path of motion.

**The Fix:** We use **Frame Skipping**. Instead of saving every single frame into the buffer, we save every 10th frame. 
Now, the 5 frames cover `0.05 seconds` of history. The robot can clearly see exactly where its arm was moving, and the noise is averaged out.

---

### Your Test for Day 13
If an NVIDIA Engineer asks you: *"Why do we flatten 5 frames of history into the MLP input instead of just calculating and feeding the velocity directly?"*

**Your Answer:**
> *"In a perfect simulator, we can cheat and feed exact ground-truth velocity. But in the real world (Sim-to-Real), calculating velocity from noisy sensors via finite differencing amplifies the noise dramatically. By feeding a stacked history buffer directly into the MLP, the network learns to implicitly extract robust temporal features and filter the noise better than hard-coded derivatives."*
