#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"
layout(location = 0) in vec3 inPosition;
layout(push_constant) uniform MarkerData {
    mat4 model;
    vec4 color;
} marker;
layout(location = 0) out vec3 color;
void main() {
    gl_Position = scene.viewProjection * marker.model * vec4(inPosition, 1.0);
    color = marker.color.rgb;
}
