# Day 5 Mastery Checklist: OpenUSD

Before you move to Day 6, you must completely abandon the way Three.js thinks about 3D models. OpenUSD is not a "3D File Format." It is a Relational Database for Physical Reality. 

Memorize these 3 core concepts:

### Mastery Rule 1: The "Prim" (Everything is a Node)
In Three.js, you have a `Mesh`. 
In USD, you have a **Prim** (Primitive). A Prim is basically just an empty folder in a Tree Data Structure. 
A Prim does nothing until you attach a **Schema** to it. 
*   If you attach a `Geom` schema, it becomes a shape you can see.
*   If you attach a `Physics` schema, it gets mass and gravity.
*   If you attach an `Audio` schema, it makes noise.
**Test:** You must know that a Robot in USD is not a single file. It is a massive tree of Prims, all holding different mathematical data schemas.

### Mastery Rule 2: Composition Arcs (The $150k Concept)
This is why Apple, NVIDIA, and Pixar use USD. 
If an artist makes a coffee cup in a standard 3D program, and you need the cup to be red instead of blue for your AI simulation, you have to open the artist's file, change the color, and save it. You just destroyed the artist's original work.

In USD, you use a **Composition Arc** (an Override). 
You create a completely new, empty file. You "point" to the artist's blue cup, and you write a line of code: `Override: Make it Red`. 
When the computer loads the scene, it loads the blue cup, reads your override, and makes it red in real-time. **The original file was never touched.** This is how thousands of engineers can work on the same simulation simultaneously.

### Mastery Rule 3: The Stage
You cannot "open" a single USD file and see the whole simulation. 
Instead, the PyTorch engine boots up a **Stage**. 
The Stage acts as a giant funnel. It pulls in the Robot USD, the Environment USD, the Physics Override USD, and the Lighting USD. It smashes them all together into a single "Resolved" reality.

---

**Test Yourself:**
If an engineer asks you: *"Why do we use OpenUSD in Isaac Lab instead of GLTF models?"*

**The only answer you need to give is:**
> *"Because GLTF is a dead, flat format for browsers. OpenUSD is a hierarchical database that supports Non-Destructive Layering (Composition Arcs). It allows AI engineers to override physics, swap robot parts using VariantSets, and layer simulations without ever permanently altering the core 3D asset files."*
