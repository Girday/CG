#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

layout(std140, set = 1, binding = 0) uniform ModelData {
    mat4 model;
    vec4 albedo;
    vec4 specularShininess;
    vec4 options;
} object;

struct PointLight {
    vec4 positionIntensity;
    vec4 color;
};
struct SpotLight {
    vec4 positionIntensity;
    vec4 directionInner; // xyz: нормированное направление, w: cos(inner angle)
    vec4 colorOuter;     // rgb: цвет, w: cos(outer angle)
};
// Массивы без фиксированного максимального размера; длина задаётся буфером.
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight values[];
} points;
layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight values[];
} spots;

vec3 safeNormalize(vec3 value) {
    return value * inversesqrt(max(dot(value, value), 1e-10));
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
    float diffuse = max(dot(N, L), 0.0);
    if (diffuse <= 0.0) return vec3(0.0);
    vec3 H = safeNormalize(L + V);
    float specular = pow(max(dot(N, H), 0.0), object.specularShininess.w);
    return radiance * (albedo * diffuse + object.specularShininess.rgb * specular);
}

void main() {
    vec3 N = safeNormalize(worldNormal);
    vec3 V = safeNormalize(scene.cameraPosition.xyz - worldPosition);
    vec3 albedo = object.albedo.rgb * mix(vec3(1.0), vertexColor, object.options.x);
    vec3 color = scene.ambient.rgb * scene.ambient.w * albedo;
    color += blinnPhong(N, V, -scene.directionalDirection.xyz,
                       scene.directionalColor.rgb * scene.directionalDirection.w, albedo);
    for (int i = 0; i < min(scene.lightCounts.x, points.values.length()); ++i) {
        vec3 delta = points.values[i].positionIntensity.xyz - worldPosition;
        float distance2 = max(dot(delta, delta), 0.01);
        vec3 radiance = points.values[i].color.rgb * points.values[i].positionIntensity.w / distance2;
        color += blinnPhong(N, V, safeNormalize(delta), radiance, albedo);
    }
    for (int i = 0; i < min(scene.lightCounts.y, spots.values.length()); ++i) {
        vec3 delta = spots.values[i].positionIntensity.xyz - worldPosition;
        vec3 L = safeNormalize(delta);
        float cosine = dot(-L, spots.values[i].directionInner.xyz);
        float inner = spots.values[i].directionInner.w;
        float outer = spots.values[i].colorOuter.w;
        float weight = abs(inner - outer) < 1e-6
            ? step(outer, cosine) : smoothstep(outer, inner, cosine);
        float distance2 = max(dot(delta, delta), 0.01);
        vec3 radiance = spots.values[i].colorOuter.rgb * spots.values[i].positionIntensity.w * weight / distance2;
        color += blinnPhong(N, V, L, radiance, albedo);
    }
    outColor = vec4(color, 1.0);
}
