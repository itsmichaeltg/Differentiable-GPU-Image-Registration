# CS179 Project Proposal: Differentiable GPU Image Registration

## 1. Summary

This project implements a GPU-accelerated spatial transformation engine capable of aligning two images through iterative optimization. By leveraging CUDA texture memory for interpolation and parallel reductions for error calculation, the system automatically finds transformation parameters (rotation, scale, and translation) to register a source image to a target.

## 2. Background Information

Image registration is the process of transforming different sets of data into one coordinate system. The forward pass maps pixels `(x, y)` to new coordinates `(x', y')` using an affine matrix `M`.

Because transformed coordinates rarely land exactly on integer pixel locations, bilinear interpolation is required to calculate the resulting color. On the GPU, this is hardware-accelerated with CUDA texture objects. The optimization pass calculates a loss function such as mean squared error across millions of pixels and updates the affine parameters with gradient descent.

### Questions to Address

- Previous implementations: OpenCV provides CPU-based warp functions, but this implementation focuses on a differentiable pipeline where the loss-to-transform loop can stay on the GPU.
- Technical challenges: Use texture memory for sub-pixel accuracy and high-speed parallel reductions for global loss.
- Problems to solve: Bridge fixed image warping and iterative optimization by implementing a differentiable backward pass that calculates image gradients relative to transformation parameters. Use warp-shuffle reductions for loss calculation and a multi-resolution pyramid to reduce local-minimum failures.
- Deliverables: A CUDA warping library, a CLI tool for image alignment, a CPU baseline for speed comparison, and frame output that can be encoded into a video showing gradual alignment.

## 3. Work Distribution

| Feature | Developer | Components |
| --- | --- | --- |
| Module 1: Warping Engine | Person A | Affine transformation kernels, texture memory binding, and bilinear interpolation logic. |
| Module 2: Alignment and Loss | Person B | MSE/cross-correlation loss kernels, parallel reduction, and gradient descent parameter updates. |
| Integration and GUI | Joint | Image I/O, performance benchmarking, and visualization of the alignment process. |

## 4. Week-by-Week Timeline

### Week 1 (May 11 - May 17)

- Person A: Set up CUDA texture objects and write basic translation/rotation kernels.
- Person B: Implement a CPU-based mean squared error baseline and project structure.

### Week 2 (May 18 - May 24)

- Person A: Implement full affine support and boundary conditions.
- Person B: Implement a parallel reduction kernel to calculate MSE on the GPU.

### Week 3 (May 25 - May 31)

- CPU demo due May 27.
- Integrate the modules into a loop: warp, calculate loss, update parameters.
- Test with synthetic shifts, rotations, and inverse recovery cases.

### Week 4 (June 1 - Final Deadline)

- Profile with Nsight.
- Compare texture-cache behavior against global-memory baselines.
- Finalize performance analysis and documentation.

## Current Status

Week 1 is implemented:

- CMake project structure modeled after the CS179 labs.
- CPU image container, affine transforms, bilinear sampling, affine warp, and MSE baseline.
- CUDA texture-object affine warp wrapper for grayscale images.
- Tests for affine math, image edge cases, CPU warp/loss behavior, and optional GPU warp parity.

Week 2 is implemented:

- Full affine CPU/GPU warp paths share the same centered transform convention.
- Zero, clamp, and wrap boundary modes are covered by CPU tests and optional CUDA parity tests.
- GPU MSE uses a warp-shuffle block reduction.
- CPU and GPU normalized cross-correlation are available for registration diagnostics.

Week 3 is implemented:

- CPU differentiable loss/gradient and Adam-style parameter updates.
- Multi-resolution pyramid alignment for translation, rotation, and scale parameters.
- Synthetic registration tests that verify loss reduction and translation recovery.
- PGM/PPM image I/O and a `register_images` CLI demo tool.

Build and test:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

CPU demo:

```bash
./build/register_images source.pgm target.pgm --output aligned.pgm --iterations 200 --pyramid 3
```
