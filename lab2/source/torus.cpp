#include "torus.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

TorusMesh generateTorus(float major_radius, float minor_radius,
	                    uint32_t major_segments, uint32_t minor_segments) {
	if (major_radius <= minor_radius || minor_radius <= 0.0f ||
	    major_segments < 3 || minor_segments < 3) {
		throw std::invalid_argument("Invalid torus radii or segment counts");
	}

	TorusMesh mesh;
	mesh.vertices.reserve((major_segments + 1) * (minor_segments + 1));
	mesh.indices.reserve(major_segments * minor_segments * 6);

	const float full_turn = 2.0f * std::numbers::pi_v<float>;
	for (uint32_t i = 0; i <= major_segments; ++i) {
		const float u = full_turn * float(i) / float(major_segments);
		const float cos_u = std::cos(u);
		const float sin_u = std::sin(u);
		for (uint32_t j = 0; j <= minor_segments; ++j) {
			const float v = full_turn * float(j) / float(minor_segments);
			const float cos_v = std::cos(v);
			const float sin_v = std::sin(v);
			const float ring_radius = major_radius + minor_radius * cos_v;
			mesh.vertices.push_back({
				{ring_radius * cos_u, ring_radius * sin_u, minor_radius * sin_v},
				{cos_v * cos_u, cos_v * sin_u, sin_v},
				{0.5f + 0.5f * cos_u, 0.5f + 0.5f * sin_u, 0.5f + 0.5f * sin_v},
			});
		}
	}

	const uint32_t row = minor_segments + 1;
	for (uint32_t i = 0; i < major_segments; ++i) {
		for (uint32_t j = 0; j < minor_segments; ++j) {
			const uint32_t a = i * row + j;
			const uint32_t b = (i + 1) * row + j;
			const uint32_t c = b + 1;
			const uint32_t d = a + 1;
			mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
		}
	}
	return mesh;
}
