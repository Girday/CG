#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(set = 0, binding = 0) uniform ObjectData {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 tint;
} object;

layout(location = 0) out vec3 worldNormal;
layout(location = 1) out vec3 vertexColor;

void main() {
    vec4 worldPosition = object.model * vec4(inPosition, 1.0);
    gl_Position = object.projection * object.view * worldPosition;
    worldNormal = normalize(mat3(transpose(inverse(object.model))) * inNormal);
    vertexColor = inColor;
}
