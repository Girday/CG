#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(std140, set = 1, binding = 0) uniform ModelData {
    mat4 model;
    vec4 albedo;
    vec4 specularShininess;
    vec4 options; // x: использовать процедурный цвет вершин
} object;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 vertexColor;

void main() {
    vec4 p = object.model * vec4(inPosition, 1.0);
    worldPosition = p.xyz;
    // Обратная транспонированная матрица сохраняет перпендикулярность нормали
    // поверхности даже при неравномерном масштабировании модели.
    worldNormal = transpose(inverse(mat3(object.model))) * inNormal;
    vertexColor = inColor;
    gl_Position = scene.viewProjection * p;
}
