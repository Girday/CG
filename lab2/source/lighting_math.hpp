#pragma once

#include "math3d.hpp"
#include <algorithm>
#include <numbers>

namespace lighting {
constexpr float radians = std::numbers::pi_v<float> / 180.0f;

inline math3d::Vec3 safeDirection(math3d::Vec3 v) {
    const float length2 = math3d::dot(v, v);
    if (length2 < 1e-10f) return {0.0f, 0.0f, -1.0f};
    return math3d::normalized(v);
}

// Камера: Z вверх; yaw поворачивает вокруг Z, pitch задаёт наклон.
inline math3d::Vec3 cameraForward(float yaw, float pitch) {
    return {std::cos(pitch * radians) * std::cos(yaw * radians),
            std::cos(pitch * radians) * std::sin(yaw * radians),
            std::sin(pitch * radians)};
}

inline float inverseSquare(float distance) {
    // Ограничение вблизи источника защищает от деления на ноль.
    return 1.0f / std::max(distance * distance, 0.01f);
}

inline float spotWeight(float cosine, float inner_degrees, float outer_degrees) {
    const float inner = std::cos(inner_degrees * radians);
    const float outer = std::cos(outer_degrees * radians);
    if (std::abs(inner - outer) < 1e-6f) return cosine >= outer ? 1.0f : 0.0f;
    const float t = std::clamp((cosine - outer) / (inner - outer), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
}
