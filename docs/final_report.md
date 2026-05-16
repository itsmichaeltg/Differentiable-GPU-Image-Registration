# Differentiable GPU Image Registration - Final Report

## Architecture

The project is organized like the CS179 lab handouts:

- `include/registration`: public headers for image storage, affine math, CUDA entry points, image I/O, and optimization.
- `src`: CPU and CUDA implementations.
- `tests`: small executable tests registered with CTest.
- `tools`: command-line demos and benchmark/profiling drivers.
- `scripts`: repeatable build, test, profiling, and video helper scripts.

The affine convention maps destination pixels back into source coordinates. This inverse-mapping design avoids holes in the output image and matches CUDA texture sampling naturally.

## Warping

The CPU warp path uses explicit bilinear interpolation with three boundary modes:

- `zero`: samples outside the image return zero.
- `clamp`: samples outside the image use the nearest valid pixel.
- `wrap`: samples wrap around image edges.

The CUDA texture path binds a grayscale source image to a `cudaTextureObject_t` and uses hardware linear filtering. A second CUDA path implements the same bilinear interpolation from global memory; this is the baseline for texture-cache profiling.

## Loss And Optimization

The CPU path implements:

- Mean squared error.
- Normalized cross-correlation.
- Differentiable MSE gradients with respect to rotation, scale, `tx`, and `ty`.
- Adam-style updates.
- A coarse-to-fine pyramid that scales translations between levels.

The CUDA path implements:

- MSE reduction with warp-shuffle block reductions.
- Normalized cross-correlation reduction.
- Texture-backed differentiable loss/gradient for grayscale images.
- Optional CUDA optimizer backend through `register_images --backend cuda`.

## CLI Demo

Generate a synthetic pair:

```bash
./build/make_synthetic_pair demo 160 128
```

Register the images and save alignment frames:

```bash
./build/register_images demo/source.pgm demo/target.pgm \
  --output demo/aligned.pgm \
  --mode similarity \
  --iterations 200 \
  --pyramid 3 \
  --frames demo/frames \
  --frame-stride 5
```

Convert frames to a video if `ffmpeg` is installed:

```bash
./scripts/make_video.sh demo/frames demo/alignment.mp4
```

## Profiling Plan

On a machine with an NVIDIA driver, run:

```bash
./scripts/profile.sh 1024 20
```

The benchmark reports CPU warp, CPU gradient, CUDA texture warp, CUDA global-memory warp, CUDA MSE reduction, and CUDA texture loss/gradient timings. Nsight Compute output is written under `profiling/`.

Important Nsight metrics:

- Texture path: texture cache hit rate, texture pipe utilization, global load transactions.
- Global path: global load efficiency and memory throughput.
- Reduction kernels: warp execution efficiency, achieved occupancy, shared-memory usage.

This sandbox has CUDA tooling but no active NVIDIA driver, so runtime CUDA tests skip cleanly here. The code still compiles with `nvcc`, and all CPU-visible tests pass under CTest.

## Testing

The test suite covers:

- Affine composition, inverse, centered transform construction, singular matrices, non-finite inputs.
- Image indexing, bilinear interpolation, boundary modes, channel handling, odd-size downsampling.
- CPU warp, MSE, normalized cross-correlation, mismatched shapes, constant-image NCC errors.
- PGM/PPM roundtrips, comment parsing, invalid image writes, missing files.
- Synthetic registration recovery with a coarse-to-fine CPU optimizer.
- Optional CUDA warp, reduction, global-memory baseline, and loss-gradient parity when a GPU is available.
