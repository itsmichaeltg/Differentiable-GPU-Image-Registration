#include "registration/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "registration/cpu_registration.hpp"
#include "registration/gpu_registration.cuh"

namespace registration {

namespace {

struct AdamState {
    TransformParams first{};
    TransformParams second{};
    int step = 0;
};

void validate_optimizer_inputs(
    const Image& source,
    const Image& target,
    const TransformParams& initial,
    const OptimizerOptions& options) {
    validate_same_shape(source, target);
    validate_transform_params(initial);
    if (options.max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }
    if (options.pyramid_levels <= 0) {
        throw std::invalid_argument("pyramid_levels must be positive");
    }
    if (options.learning_rate_translation < 0.0f ||
        options.learning_rate_theta < 0.0f ||
        options.learning_rate_scale < 0.0f) {
        throw std::invalid_argument("learning rates must be non-negative");
    }
}

LossGradient evaluate_loss_gradient(
    const Image& source,
    const Image& target,
    const TransformParams& params,
    const OptimizerOptions& options) {
    if (options.backend == RegistrationBackend::CUDA) {
        return loss_gradient_cuda(source, target, params, options.boundary);
    }
    return loss_gradient_cpu(source, target, params, options.boundary);
}

Image warp_for_backend(
    const Image& source,
    int output_width,
    int output_height,
    const TransformParams& params,
    const OptimizerOptions& options) {
    if (options.backend == RegistrationBackend::CUDA) {
        return warp_affine_cuda(source, output_width, output_height, params, options.boundary);
    }
    return warp_affine_cpu(source, output_width, output_height, params, options.boundary);
}

std::vector<Image> build_pyramid(const Image& image, int requested_levels) {
    std::vector<Image> pyramid;
    pyramid.push_back(image);
    for (int level = 1; level < requested_levels; ++level) {
        const Image& previous = pyramid.back();
        if (previous.width == 1 && previous.height == 1) {
            break;
        }
        pyramid.push_back(downsample_half(previous));
    }
    return pyramid;
}

float sample_dx(
    const Image& image,
    float x,
    float y,
    int channel,
    BoundaryMode boundary) {
    return 0.5f * (
        sample_bilinear(image, x + 1.0f, y, channel, boundary) -
        sample_bilinear(image, x - 1.0f, y, channel, boundary));
}

float sample_dy(
    const Image& image,
    float x,
    float y,
    int channel,
    BoundaryMode boundary) {
    return 0.5f * (
        sample_bilinear(image, x, y + 1.0f, channel, boundary) -
        sample_bilinear(image, x, y - 1.0f, channel, boundary));
}

void adam_update_value(
    float& value,
    float gradient,
    float learning_rate,
    float& first,
    float& second,
    int step) {
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float epsilon = 1.0e-8f;

    first = beta1 * first + (1.0f - beta1) * gradient;
    second = beta2 * second + (1.0f - beta2) * gradient * gradient;

    const float first_hat = first / (1.0f - std::pow(beta1, static_cast<float>(step)));
    const float second_hat = second / (1.0f - std::pow(beta2, static_cast<float>(step)));
    value -= learning_rate * first_hat / (std::sqrt(second_hat) + epsilon);
}

void apply_update(
    TransformParams& params,
    const TransformParams& gradient,
    const OptimizerOptions& options,
    AdamState& adam) {
    ++adam.step;
    if (options.optimize_rotation) {
        adam_update_value(
            params.theta,
            gradient.theta,
            options.learning_rate_theta,
            adam.first.theta,
            adam.second.theta,
            adam.step);
    }
    if (options.optimize_scale) {
        adam_update_value(
            params.scale,
            gradient.scale,
            options.learning_rate_scale,
            adam.first.scale,
            adam.second.scale,
            adam.step);
        params.scale = std::clamp(params.scale, 0.10f, 10.0f);
    }
    if (options.optimize_translation) {
        adam_update_value(
            params.tx,
            gradient.tx,
            options.learning_rate_translation,
            adam.first.tx,
            adam.second.tx,
            adam.step);
        adam_update_value(
            params.ty,
            gradient.ty,
            options.learning_rate_translation,
            adam.first.ty,
            adam.second.ty,
            adam.step);
    }
}

TransformParams scale_translation(const TransformParams& params, float ratio_x, float ratio_y) {
    TransformParams scaled = params;
    scaled.tx *= ratio_x;
    scaled.ty *= ratio_y;
    return scaled;
}

}  // namespace

LossGradient loss_gradient_cpu(
    const Image& source,
    const Image& target,
    const TransformParams& params,
    BoundaryMode boundary) {
    validate_same_shape(source, target);
    validate_transform_params(params);

    const float source_cx = 0.5f * static_cast<float>(source.width - 1);
    const float source_cy = 0.5f * static_cast<float>(source.height - 1);
    const float dest_cx = 0.5f * static_cast<float>(target.width - 1);
    const float dest_cy = 0.5f * static_cast<float>(target.height - 1);
    const float cos_theta = std::cos(params.theta);
    const float sin_theta = std::sin(params.theta);
    const double inv_count = 1.0 / static_cast<double>(target.pixels.size());

    double loss = 0.0;
    double grad_theta = 0.0;
    double grad_scale = 0.0;
    double grad_tx = 0.0;
    double grad_ty = 0.0;

    for (int y = 0; y < target.height; ++y) {
        for (int x = 0; x < target.width; ++x) {
            const float u = static_cast<float>(x) - dest_cx;
            const float v = static_cast<float>(y) - dest_cy;
            const float rotated_x = cos_theta * u - sin_theta * v;
            const float rotated_y = sin_theta * u + cos_theta * v;
            const float source_x = source_cx + params.tx + params.scale * rotated_x;
            const float source_y = source_cy + params.ty + params.scale * rotated_y;

            const float dtheta_x = params.scale * (-sin_theta * u - cos_theta * v);
            const float dtheta_y = params.scale * (cos_theta * u - sin_theta * v);

            for (int c = 0; c < target.channels; ++c) {
                const float warped = sample_bilinear(source, source_x, source_y, c, boundary);
                const float error = warped - target.at(x, y, c);
                const float image_dx = sample_dx(source, source_x, source_y, c, boundary);
                const float image_dy = sample_dy(source, source_x, source_y, c, boundary);
                const double factor = 2.0 * static_cast<double>(error) * inv_count;

                loss += static_cast<double>(error) * static_cast<double>(error);
                grad_theta += factor * (image_dx * dtheta_x + image_dy * dtheta_y);
                grad_scale += factor * (image_dx * rotated_x + image_dy * rotated_y);
                grad_tx += factor * image_dx;
                grad_ty += factor * image_dy;
            }
        }
    }

    LossGradient output;
    output.loss = static_cast<float>(loss * inv_count);
    output.gradient.theta = static_cast<float>(grad_theta);
    output.gradient.scale = static_cast<float>(grad_scale);
    output.gradient.tx = static_cast<float>(grad_tx);
    output.gradient.ty = static_cast<float>(grad_ty);
    return output;
}

RegistrationResult align_images(
    const Image& source,
    const Image& target,
    const TransformParams& initial,
    const OptimizerOptions& options,
    const IterationCallback& callback) {
    validate_optimizer_inputs(source, target, initial, options);

    const std::vector<Image> source_pyramid = build_pyramid(source, options.pyramid_levels);
    const std::vector<Image> target_pyramid = build_pyramid(target, options.pyramid_levels);
    const int coarsest = static_cast<int>(source_pyramid.size()) - 1;

    TransformParams params = scale_translation(
        initial,
        static_cast<float>(source_pyramid[coarsest].width) / static_cast<float>(source.width),
        static_cast<float>(source_pyramid[coarsest].height) / static_cast<float>(source.height));

    RegistrationResult result;
    result.initial_loss = mse_cpu(
        warp_affine_cpu(source, target.width, target.height, initial, options.boundary),
        target);

    int total_iterations = 0;
    for (int level = coarsest; level >= 0; --level) {
        const Image& level_source = source_pyramid[level];
        const Image& level_target = target_pyramid[level];
        AdamState adam;
        float previous_loss = std::numeric_limits<float>::infinity();

        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            const LossGradient gradient =
                evaluate_loss_gradient(level_source, level_target, params, options);

            IterationRecord record;
            record.pyramid_level = level;
            record.iteration = iteration;
            record.params = params;
            record.loss = gradient.loss;
            result.history.push_back(record);

            if (callback && options.callback_interval > 0 &&
                (iteration % options.callback_interval == 0 || iteration == options.max_iterations - 1)) {
                callback(
                    record,
                    warp_for_backend(level_source, level_target.width, level_target.height, params, options));
            }

            ++total_iterations;
            if (std::fabs(previous_loss - gradient.loss) <= options.tolerance) {
                break;
            }
            previous_loss = gradient.loss;
            apply_update(params, gradient.gradient, options, adam);
        }

        if (level > 0) {
            params = scale_translation(
                params,
                static_cast<float>(source_pyramid[level - 1].width) / static_cast<float>(level_source.width),
                static_cast<float>(source_pyramid[level - 1].height) / static_cast<float>(level_source.height));
        }
    }

    result.params = params;
    result.aligned = warp_for_backend(source, target.width, target.height, params, options);
    result.final_loss = mse_cpu(result.aligned, target);
    result.iterations = total_iterations;
    return result;
}

}  // namespace registration
