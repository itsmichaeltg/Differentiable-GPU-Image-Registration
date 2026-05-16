#pragma once

#include <functional>
#include <vector>

#include "registration/affine.hpp"
#include "registration/image.hpp"

namespace registration {

enum class RegistrationBackend {
    CPU,
    CUDA
};

struct LossGradient {
    float loss = 0.0f;
    TransformParams gradient{};
};

struct OptimizerOptions {
    int max_iterations = 150;
    int pyramid_levels = 1;
    float learning_rate_translation = 0.8f;
    float learning_rate_theta = 0.01f;
    float learning_rate_scale = 0.01f;
    float tolerance = 1.0e-7f;
    bool optimize_translation = true;
    bool optimize_rotation = true;
    bool optimize_scale = true;
    BoundaryMode boundary = BoundaryMode::Clamp;
    RegistrationBackend backend = RegistrationBackend::CPU;
    int callback_interval = 0;
};

struct IterationRecord {
    int pyramid_level = 0;
    int iteration = 0;
    TransformParams params{};
    float loss = 0.0f;
};

struct RegistrationResult {
    TransformParams params{};
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    int iterations = 0;
    std::vector<IterationRecord> history;
    Image aligned;
};

using IterationCallback = std::function<void(const IterationRecord&, const Image&)>;

LossGradient loss_gradient_cpu(
    const Image& source,
    const Image& target,
    const TransformParams& params,
    BoundaryMode boundary = BoundaryMode::Clamp);

RegistrationResult align_images(
    const Image& source,
    const Image& target,
    const TransformParams& initial,
    const OptimizerOptions& options,
    const IterationCallback& callback = nullptr);

}  // namespace registration
