#include "application.hpp"
#include "scene.hpp"

namespace application {

bool initialize() {
	return scene::initialize();
}

void shutdown() {
	scene::shutdown();
}

void update(double time) {
	scene::update(time);
}

void render(const graphics::internal::FrameData& fd) {
	scene::render(fd);
}

} // namespace application
