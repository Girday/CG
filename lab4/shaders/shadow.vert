#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"
#include "model.glsl"

layout(location = 0) in vec3 inPosition;
layout(push_constant) uniform ShadowPass {
    uint lightIndex;
} shadowPass;

void main() {
    gl_Position = scene.lightViewProjection[shadowPass.lightIndex]
                * object.model * vec4(inPosition, 1.0);
}
