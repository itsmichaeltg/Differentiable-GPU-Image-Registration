# Differentiable GPU Image Registration (CS 179 Starter)

Starter scaffold for a Caltech CS 179 project on differentiable GPU image registration.

## What's included
- CMake scaffold for a two-person module split plus integration target
- Starter APIs and data structures for affine transforms, warping, loss, and alignment
- Explicit TODO stubs for student-owned modules
- Minimal CLI starter (`dgir_cli`) that documents next steps
- Unit tests that verify starter behavior and enforce TODO boundaries

## Build
```bash
cmake -S . -B build
cmake --build build
```

Optional CUDA starter build:
```bash
cmake -S . -B build -DDGIR_ENABLE_CUDA=ON
cmake --build build
```

## Run tests
```bash
cd build && ctest --output-on-failure
```

## Student TODOs
1. **Module 1: Warping Engine (Person A)**
   - Implement affine transformation kernels.
   - Implement texture memory binding.
   - Implement bilinear interpolation logic.
2. **Module 2: Alignment & Loss (Person B)**
   - Implement MSE/Cross-correlation loss kernels.
   - Implement parallel reduction for global loss.
   - Implement gradient-descent parameter updates.
3. **Integration & GUI (Joint)**
   - Add image I/O (e.g., stb_image).
   - Add performance benchmarking.
   - Add visualization/video output of iterative alignment.
