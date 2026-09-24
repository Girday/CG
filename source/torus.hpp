#pragma once

#include <cstdint>
#include <vector>

struct TorusVertex {
	float position[3];
	float normal[3];
	float color[3];
};

struct TorusMesh {
	std::vector<TorusVertex> vertices;
	std::vector<uint32_t> indices;
};

// major_radius задаёт расстояние от центра тора до центра трубки,
// minor_radius — радиус трубки; ось вращения направлена по Z.
TorusMesh generateTorus(float major_radius, float minor_radius,
	                    uint32_t major_segments, uint32_t minor_segments);
