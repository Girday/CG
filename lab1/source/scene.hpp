#pragma once

#include "graphics_internal.hpp"

namespace scene {
bool initialize();
void shutdown();
void update(double time);
void render(const graphics::internal::FrameData& frame);
}
