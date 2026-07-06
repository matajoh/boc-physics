# Plan for creating a boc physics engine

## Stage 1: Clean XPBD/Jacobi serial solver

Clean up all the crap the AI wrote. 

1. [ ] Pare it back to just the code needed for a XPBD/Jacobi solver
2. [ ] Timing test - AI can usefully help here
3. [ ] Clean it up so it doesn't suck
4. [ ] Timing test - AI can usefully help here
5. [ ] Refactor to use boc.Matrix representations (e.g., no more Mr. RigidBody everywhere)
6. [ ] Timing test - AI can usefully help here

Gate: should get to this point without losing too much, or ideally any, perf.
The goal of this stage is to build a pure serial, arena-based allocation system

## Stage 2: Slabs

Divide the world into equal slabs, but compute serially.

1. [ ] Rearchitect serial code to be slab based.
2. [ ] Implement ghost rows etc.
3. [ ] Validate correctness - AI can usefully help here
4. [ ] Measure slowdown per slab - AI can usefully help here

## Stage 3: BOC

Schedule slabs as behaviors

1. [ ] Create slab-based behaviors
2. [ ] Measure naive schedule  - AI can usefully help here
3. [ ] Coloring - AI can usefully help here
4. [ ] Measure scheduling improvement - AI can usefully help here
