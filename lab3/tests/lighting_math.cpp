#include "lighting_math.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

void check(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(EXIT_FAILURE); }
}
bool close(float a, float b) { return std::abs(a - b) < 1e-5f; }
int main() {
    using namespace lighting;
    check(close(inverseSquare(1), 1), "Falloff at distance 1");
    check(close(inverseSquare(2), 0.25f), "Falloff at distance 2");
    check(std::isfinite(inverseSquare(0)), "Finite falloff at source");
    check(close(spotWeight(1, 15, 30), 1), "Spot axis fully lit");
    check(close(spotWeight(std::cos(40*radians), 15, 30), 0), "Outside cone unlit");
    const float midpoint = (std::cos(15*radians) + std::cos(30*radians)) / 2;
    check(close(spotWeight(midpoint, 15, 30), 0.5f), "Soft cone midpoint");
    check(close(spotWeight(1, 30, 30), 1), "Hard cone inside");
    check(close(spotWeight(0, 30, 30), 0), "Hard cone outside");
    const auto f = cameraForward(90, 0);
    check(close(f.x, 0) && close(f.y, 1) && close(f.z, 0), "Camera yaw 90");
    const auto eye = math3d::Vec3{2, -4, 3};
    const auto forward = cameraForward(33, -25);
    const auto view = math3d::lookAt(eye, {eye.x+forward.x, eye.y+forward.y, eye.z+forward.z}, {0,0,1});
    for (int row = 0; row < 3; ++row) {
        const float camera_origin = view[row]*eye.x + view[4+row]*eye.y + view[8+row]*eye.z + view[12+row];
        check(close(camera_origin, 0), "View maps camera position to origin");
    }
    const auto safe = safeDirection({0,0,0});
    check(close(math3d::dot(safe,safe), 1), "Zero direction has finite unit fallback");
    std::cout << "Lighting falloff, spot edges, camera view: OK\n";
}
