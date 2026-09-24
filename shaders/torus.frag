#version 450

layout(location = 0) in vec3 worldNormal;
layout(location = 1) in vec3 vertexColor;

layout(set = 0, binding = 0) uniform ObjectData {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 tint;
} object;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 lightDirection = normalize(vec3(0.3, -0.5, 1.0));
    float diffuse = max(dot(normalize(worldNormal), lightDirection), 0.0);
    vec3 color = vertexColor * object.tint.rgb * (0.35 + 0.65 * diffuse);
    outColor = vec4(color, 1.0);
}
