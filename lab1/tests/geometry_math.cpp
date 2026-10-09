#include "math3d.hpp"
#include "torus.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {
bool close(float a, float b, float tolerance = 1e-5f) {
	return std::abs(a - b) < tolerance;
}

math3d::Vec3 position(const TorusVertex& vertex) {
	return {vertex.position[0], vertex.position[1], vertex.position[2]};
}
}

int main() {
	const auto mesh = generateTorus(1.0f, 0.35f, 48, 24);
	assert(mesh.vertices.size() == 49 * 25);
	assert(mesh.indices.size() == 48 * 24 * 6);
	for (const auto& vertex : mesh.vertices) {
		const math3d::Vec3 p = position(vertex);
		const float radial = std::sqrt(p.x * p.x + p.y * p.y);
		assert(close((radial - 1.0f) * (radial - 1.0f) + p.z * p.z, 0.35f * 0.35f));
		const auto& n = vertex.normal;
		assert(close(n[0] * n[0] + n[1] * n[1] + n[2] * n[2], 1.0f));
	}
	for (size_t i = 0; i < mesh.indices.size(); i += 3) {
		const uint32_t a = mesh.indices[i], b = mesh.indices[i + 1], c = mesh.indices[i + 2];
		assert(a < mesh.vertices.size() && b < mesh.vertices.size() && c < mesh.vertices.size());
		const auto face = math3d::cross(position(mesh.vertices[b]) - position(mesh.vertices[a]),
		                                position(mesh.vertices[c]) - position(mesh.vertices[a]));
		const auto& normal = mesh.vertices[a].normal;
		assert(math3d::dot(face, {normal[0], normal[1], normal[2]}) > 0.0f);
	}

	const auto model = math3d::multiply(math3d::translation({1.0f, 2.0f, 3.0f}),
	                                    math3d::scaling({2.0f, 3.0f, 4.0f}));
	assert(close(model[0] + model[12], 3.0f));
	assert(close(model[5] + model[13], 5.0f));
	assert(close(model[10] + model[14], 7.0f));

	const auto view = math3d::lookAt({0.0f, -5.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
	                                 {0.0f, 0.0f, 1.0f});
	assert(close(view[12], 0.0f) && close(view[13], 0.0f) && close(view[14], -5.0f));
	for (const auto& projection : {
		math3d::perspective(0.8f, 16.0f / 9.0f, 0.1f, 100.0f),
		math3d::orthographic(5.5f, 16.0f / 9.0f, 0.1f, 100.0f),
	}) {
		const auto depth = [&](float camera_z) {
			return (projection[10] * camera_z + projection[14]) /
			       (projection[11] * camera_z + projection[15]);
		};
		assert(close(depth(-0.1f), 0.0f));
		assert(close(depth(-100.0f), 1.0f));
	}
	std::cout << "Torus geometry and Vulkan projection matrices: OK\n";
}
