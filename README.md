# Differentiable GPU Image Registration (CS 179 Starter)

Starter scaffold for a Caltech CS 179 project on differentiable GPU image registration.

## What's included
- CPU baseline affine warp with bilinear sampling (`warp_affine_cpu`)
- CPU MSE loss and numeric gradient baseline
- CPU optimization loop (`align_images_cpu`)
- CUDA module placeholder (`warp_affine_cuda`) with TODOs for students
- Minimal CLI demo (`dgir_cli`)
- Unit tests for the starter baseline

## Build
```bash
cmake -S /home/runner/work/Differentiable-GPU-Image-Registration/Differentiable-GPU-Image-Registration -B /home/runner/work/Differentiable-GPU-Image-Registration/Differentiable-GPU-Image-Registration/build
cmake --build /home/runner/work/Differentiable-GPU-Image-Registration/Differentiable-GPU-Image-Registration/build
```

Optional CUDA starter build:
```bash
cmake -S /home/runner/work/Differentiable-GPU-Image-Registration/Differentiable-GPU-Image-Registration -B /home/runner/work/Differentiable-GPU-Image-Registration/Differentiable-GPU-Image-Registration/build -DDGIR_ENABLE_CUDA=ON
cmake --build /home/runner/work/Differentiable-GPU-Image-Registration/Differentiable-GPU-Image-Registration/build
```

## Run tests
```bash
cd /home/runner/work/Differentiable-GPU-Image-Registration/Differentiable-GPU-Image-Registration/build && ctest --output-on-failure
```

## Student TODOs
1. Replace numeric gradient with analytic/differentiable backward pass on GPU.
2. Implement CUDA texture-object-based affine warp in `src/warp_cuda.cu`.
3. Add GPU parallel reduction for loss computation.
4. Add image I/O + video output pipeline for iteration visualization.
5. Add multiresolution pyramid to improve convergence.
