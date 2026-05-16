# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Caltech CS 179 starter scaffold for **Differentiable GPU Image Registration** — a GPU-accelerated spatial transformation engine that aligns a source image to a target by iteratively optimizing affine parameters (rotation, scale, translation) end-to-end on the device.

What makes this project distinct from existing tools (e.g. OpenCV's CPU warps): the **entire loss-to-transform loop stays on the GPU**. The backward pass differentiates image gradients with respect to transform parameters, so the warp → loss → parameter-update cycle never round-trips through the host.

Performance-critical design pillars (from the project proposal — keep these in mind when reviewing or extending GPU code):

- **CUDA texture objects** for hardware bilinear interpolation and sub-pixel sampling. The CUDA warp path in [src/warp_cuda.cu](src/warp_cuda.cu) is expected to bind the source as a texture object, not read from global memory.
- **Warp-shuffle parallel reductions** for computing MSE / cross-correlation loss across all pixels in one pass.
- **Multi-resolution pyramid** for the optimizer, to avoid stalling in local minima on large initial misalignments.
- **Boundary handling** (clamp / pad) is part of the warp contract, not an afterthought — the CPU `bilinear_sample_clamp` already commits to clamp-to-edge; the GPU path must match for CPU↔GPU parity tests.

Target deliverables: CUDA warping library, CLI alignment tool, CPU baseline for speed comparison, and a video/sequence output showing the source progressively aligning to the target over iterations.

### Repo state

The repo intentionally ships as a TODO-driven skeleton: a small set of helpers are implemented (`affine.cpp`, `make_checkerboard`), while the modules students are expected to write throw `std::logic_error("TODO(student): ...")`. Tests assert *both* the working helpers behave correctly *and* the student-owned functions still throw — implementing a TODO will break the corresponding `expect_todo` assertion in [tests/test_starter.cpp](tests/test_starter.cpp), and that test must be updated alongside the implementation.

## Build & test

```bash
cmake -S . -B build
cmake --build build
cd build && ctest --output-on-failure
```

CUDA module (opt-in; requires CUDA toolchain):

```bash
cmake -S . -B build -DDGIR_ENABLE_CUDA=ON
cmake --build build
```

Run a single test binary directly (only one binary today; `ctest -R <regex>` filters when more are added):

```bash
./build/dgir_tests
```

Run the CLI stub:

```bash
./build/dgir_cli
```

## Architecture

Two-person module split with an integration target, wired through [CMakeLists.txt](CMakeLists.txt):

- **`dgir_core`** (static lib) — CPU-side reference + scaffolding. Public headers live in [include/dgir/](include/dgir/); implementations in [src/](src/).
- **`dgir_cuda`** (optional, gated by `DGIR_ENABLE_CUDA`) — single `.cu` file holding the GPU warp kernel.
- **`dgir_cli`** — thin executable in [tools/](tools/) linking `dgir_core`.
- **`dgir_tests`** — CTest target built from [tests/test_starter.cpp](tests/test_starter.cpp).

Data flow across modules (each box is a header + matching `.cpp`):

```
TransformParams (affine.h)
   │ make_affine                     ─► Affine2x3
   ▼
warp_cpu.h / warp_cuda.h             ─► warped Image
   │ (samples Image via bilinear)
   ▼
loss.h  mse_loss / loss_and_numeric_gradient
   │                                 ─► LossAndGradient { loss, grad[4] }
   ▼
optimizer.h  gradient_descent_step   ─► updated TransformParams
   ▼
alignment.h  align_images_cpu        ─► AlignmentResult { params, losses[] }
```

Key conventions baked into the API surface:

- Gradient ordering is fixed: `[tx, ty, theta_rad, scale]`, matching the field order of `TransformParams`. Any new loss/optimizer code must preserve this.
- `warp_affine_cpu` takes a **source-to-target** affine; sampling code is expected to invert it internally via `invert_affine` to compute the source coordinate per output pixel.
- `bilinear_sample_clamp` uses clamp-to-edge boundary semantics — match this in any GPU port so CPU/GPU results are bit-comparable in tests.
- `Image` is row-major `std::vector<float>` with `at(x, y) == pixels[y * width + x]`.

## Module ownership (per [README.md](README.md))

- **Module 1 (Person A) — Warping engine**: [src/warp_cpu.cpp](src/warp_cpu.cpp), [src/warp_cuda.cu](src/warp_cuda.cu). Owns affine kernels, texture-memory binding, bilinear interpolation, and boundary modes (clamp / pad).
- **Module 2 (Person B) — Alignment & loss**: [src/loss.cpp](src/loss.cpp), [src/optimizer.cpp](src/optimizer.cpp), [src/alignment.cpp](src/alignment.cpp). Owns MSE / cross-correlation loss kernels, the parallel reduction, and gradient-descent parameter updates. (`make_checkerboard` in `alignment.cpp` is already implemented as a test fixture; the `align_images_cpu` driver loop is the actual TODO.)
- **Joint — Integration & GUI**: Image I/O (stb_image is the suggested dependency), Nsight-based profiling / benchmarking, visualization of the iterative alignment. None of this exists yet; add new sources to `dgir_core` in [CMakeLists.txt](CMakeLists.txt) when introducing them.

### Timeline anchors (from the proposal)

- **Week 1 (May 11–17)**: CUDA texture setup + basic translation/rotation kernel (A); CPU MSE baseline + project structure (B).
- **Week 2 (May 18–24)**: Full affine support + boundary handling (A); GPU parallel-reduction MSE (B).
- **Week 3 (May 25–31)** — **CPU demo due May 27**: integrate warp → loss → param-update loop; test by inverting synthetic shifts/rotations.
- **Week 4 (June 1 → final deadline)**: Nsight profiling (texture cache hit rates vs. global memory), performance analysis, documentation.

## When implementing a TODO

1. Replace the `throw std::logic_error(...)` body with the real implementation.
2. Open [tests/test_starter.cpp](tests/test_starter.cpp) and replace that function's `expect_todo(...)` line with a real behavioral assertion — leaving it will cause the test to fail because the function no longer throws.
3. The compile flags `-Wall -Wextra -Wpedantic` apply to `dgir_core` only; keep new code warning-clean.
