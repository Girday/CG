#pragma once

#include <array>
#include <cmath>

namespace math3d {

struct Vec3 {
	float x, y, z;
};

using Mat4 = std::array<float, 16>; // Column-major, как в GLSL.

inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 normalized(Vec3 v) {
	const float length = std::sqrt(dot(v, v));
	return {v.x / length, v.y / length, v.z / length};
}

inline Mat4 identity() {
	return {1, 0, 0, 0,
	        0, 1, 0, 0,
	        0, 0, 1, 0,
	        0, 0, 0, 1};
}

inline Mat4 multiply(const Mat4& a, const Mat4& b) {
	Mat4 result{};
	for (int column = 0; column < 4; ++column)
		for (int row = 0; row < 4; ++row)
			for (int k = 0; k < 4; ++k)
				result[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
	return result;
}

inline Mat4 translation(Vec3 p) {
	Mat4 m = identity();
	m[12] = p.x; m[13] = p.y; m[14] = p.z;
	return m;
}

inline Mat4 scaling(Vec3 s) {
	Mat4 m = identity();
	m[0] = s.x; m[5] = s.y; m[10] = s.z;
	return m;
}

inline Mat4 rotationX(float angle) {
	Mat4 m = identity();
	const float c = std::cos(angle), s = std::sin(angle);
	m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
	return m;
}

inline Mat4 rotationY(float angle) {
	Mat4 m = identity();
	const float c = std::cos(angle), s = std::sin(angle);
	m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
	return m;
}

inline Mat4 rotationZ(float angle) {
	Mat4 m = identity();
	const float c = std::cos(angle), s = std::sin(angle);
	m[0] = c; m[1] = s; m[4] = -s; m[5] = c;
	return m;
}

inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
	const Vec3 forward = normalized(target - eye);
	const Vec3 right = normalized(cross(forward, up));
	const Vec3 camera_up = cross(right, forward);
	Mat4 m = identity();
	m[0] = right.x; m[4] = right.y; m[8] = right.z;
	m[1] = camera_up.x; m[5] = camera_up.y; m[9] = camera_up.z;
	m[2] = -forward.x; m[6] = -forward.y; m[10] = -forward.z;
	m[12] = -dot(right, eye);
	m[13] = -dot(camera_up, eye);
	m[14] = dot(forward, eye);
	return m;
}

// Правосторонняя проекция Vulkan: глубина 0..1, ось Y перевёрнута для viewport.
inline Mat4 perspective(float vertical_fov, float aspect, float near_plane, float far_plane) {
	const float f = 1.0f / std::tan(vertical_fov * 0.5f);
	Mat4 m{};
	m[0] = f / aspect;
	m[5] = -f;
	m[10] = far_plane / (near_plane - far_plane);
	m[11] = -1.0f;
	m[14] = far_plane * near_plane / (near_plane - far_plane);
	return m;
}

inline Mat4 orthographic(float vertical_height, float aspect, float near_plane, float far_plane) {
	Mat4 m = identity();
	m[0] = 2.0f / (vertical_height * aspect);
	m[5] = -2.0f / vertical_height;
	m[10] = 1.0f / (near_plane - far_plane);
	m[14] = near_plane / (near_plane - far_plane);
	return m;
}

} // namespace math3d
