#include "texture_math.hpp"
#include "torus.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition) {
    if (!condition) throw std::runtime_error("UV seam / dynamic UBO alignment check failed");
}

int main() {
    // Проверяем оба UV-шва: позиции совпадают, UV различаются ровно на период.
    for (uint32_t major : {3u, 7u, 48u}) for (uint32_t minor : {3u, 5u, 24u}) {
        const auto mesh = generateTorus(1, 0.35f, major, minor);
        const auto row = minor + 1;
        auto seam = [&](const TorusVertex& a, const TorusVertex& b, int axis) {
            for (int k = 0; k < 3; ++k) {
                require(std::abs(a.position[k] - b.position[k]) < 1e-5f);
                require(std::abs(a.normal[k] - b.normal[k]) < 1e-5f);
            }
            require(a.uv[axis] == 0 && b.uv[axis] == 1);
            require(a.uv[1 - axis] == b.uv[1 - axis]);
        };
        for (uint32_t j = 0; j <= minor; ++j) seam(mesh.vertices[j], mesh.vertices[major * row + j], 0);
        for (uint32_t i = 0; i <= major; ++i) seam(mesh.vertices[i * row], mesh.vertices[i * row + minor], 1);
        for (const auto& v : mesh.vertices)
            require(v.uv[0] >= 0 && v.uv[0] <= 1 && v.uv[1] >= 0 && v.uv[1] <= 1);
    }
    // 96 байт C++/GLSL не обязательно совпадают с шагом dynamic UBO.
    for (size_t alignment : {1u, 16u, 64u, 256u}) {
        const auto stride = texture_math::uniformStride(96, alignment);
        require(stride >= 96 && stride % alignment == 0);
        require(stride - 96 < alignment);
        for (size_t slot = 0; slot < 100; ++slot) {
            require((slot * stride) % alignment == 0);
            require(slot * stride + 96 <= 100 * stride);
        }
    }
    require(texture_math::uniformStride(96, 256) == 256);
    require(texture_math::uniformStride(96, 16) == 96);
    std::cout << "Both torus UV seams and dynamic UBO offsets: OK\n";
}
