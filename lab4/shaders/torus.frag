#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"
#include "model.glsl"

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 1) uniform sampler2D albedoTexture;
layout(set = 1, binding = 2) uniform sampler2D specularTexture;
layout(set = 1, binding = 3) uniform sampler2D emissiveTexture;
layout(set = 0, binding = 3) uniform sampler2DShadow sunShadowMap;
layout(set = 0, binding = 4) uniform sampler2DShadow spotShadowMap;
layout(push_constant) uniform SamplingData {
    int effect;
    float time;
    float strength;
    float speed;
} sampling;

vec2 sampleUV() {
    if (sampling.effect == 0) return uv;
    float t = sampling.time * sampling.speed;
    // Периодическая деформация сохраняет непрерывность на UV-шве тора.
    return uv + sampling.strength * vec2(sin(6.2831853 * uv.y * 3.0 + t),
                                          cos(6.2831853 * uv.x * 2.0 - t));
}

vec3 sampleAlbedo(vec2 coordinate) {
    if (sampling.effect == 0) return texture(albedoTexture, coordinate).rgb;
    float t = sampling.time * sampling.speed;
    vec3 color;
    if (sampling.effect == 1) {
        vec2 shift = sampling.strength * 0.3 * vec2(cos(t), sin(t));
        // Три выборки одного изображения: цветовые каналы смещены по-разному.
        color = vec3(texture(albedoTexture, coordinate + shift).r,
                     texture(albedoTexture, coordinate).g,
                     texture(albedoTexture, coordinate - shift).b);
    } else {
        // Свертка 3x3. Шаг выборки измеряется в текселях, а не в размере окна.
        vec2 stepUV = (1.0 + sampling.strength * 50.0) / vec2(textureSize(albedoTexture, 0));
        color = vec3(0.0);
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
                color += texture(albedoTexture, coordinate + vec2(x, y) * stepUV).rgb / 9.0;
    }
    vec3 modulation = 0.85 + 0.15 * cos(t + vec3(0.0, 2.094, 4.189));
    return color * modulation;
}

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

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, vec3 specularMask) {
    float diffuse = max(dot(N, L), 0.0);
    if (diffuse <= 0.0) return vec3(0.0);
    vec3 H = safeNormalize(L + V);
    float specular = pow(max(dot(N, H), 0.0), object.specularShininess.w);
    return radiance * (albedo * diffuse + object.specularShininess.rgb * specularMask * specular);
}

float visibility(sampler2DShadow map, mat4 lightViewProjection, vec3 normal, vec3 toLight, float bias) {
    vec4 clip = lightViewProjection * vec4(worldPosition, 1.0);
    if (clip.w <= 0.0) return 1.0;
    vec3 ndc = clip.xyz / clip.w;
    // Только X/Y переводятся из [-1, 1] в [0, 1]. Глубина Vulkan уже в [0, 1].
    vec2 mapUV = ndc.xy * 0.5 + 0.5;
    if (ndc.z < 0.0 || ndc.z >= 1.0 || any(lessThan(mapUV, vec2(0.0))) ||
        any(greaterThanEqual(mapUV, vec2(1.0)))) return 1.0;

    float slope = 1.0 - max(dot(normal, toLight), 0.0);
    float referenceDepth = ndc.z - bias * (1.0 + 2.0 * slope);
    vec2 texel = 1.0 / vec2(textureSize(map, 0));
    float visible = 0.0;
    // Сэмплер сравнивает referenceDepth < storedDepth для каждого текселя.
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            visible += textureLod(map, vec3(mapUV + vec2(x, y) * texel, referenceDepth), 0.0);
    return visible / 9.0;
}

void main() {
    vec3 N = safeNormalize(worldNormal);
    vec3 V = safeNormalize(scene.cameraPosition.xyz - worldPosition);
    vec2 coordinate = sampleUV();
    // В w хранится вес текстуры. У пола рисунок приглушён, чтобы видеть края тени.
    vec3 albedo = object.albedo.rgb * mix(vec3(0.48), sampleAlbedo(coordinate), object.albedo.w);
    vec3 specularMask = texture(specularTexture, coordinate).rgb;
    vec3 color = scene.ambient.rgb * scene.ambient.w * albedo;
    vec3 toSun = -scene.directionalDirection.xyz;
    float sunVisibility = scene.shadowSettings.x > 0.0 && dot(N, toSun) > 0.0
        ? visibility(sunShadowMap, scene.lightViewProjection[0], N, toSun, scene.shadowSettings.y) : 1.0;
    color += sunVisibility * blinnPhong(N, V, toSun, scene.directionalColor.rgb * scene.directionalDirection.w,
                                       albedo, specularMask);
    for (int i = 0; i < min(scene.lightCounts.x, points.values.length()); ++i) {
        vec3 delta = points.values[i].positionIntensity.xyz - worldPosition;
        float distance2 = max(dot(delta, delta), 0.01);
        vec3 radiance = points.values[i].color.rgb * points.values[i].positionIntensity.w / distance2;
        color += blinnPhong(N, V, safeNormalize(delta), radiance, albedo, specularMask);
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
        if (i == scene.lightCounts.z && weight > 0.0 && dot(N, L) > 0.0)
            radiance *= visibility(spotShadowMap, scene.lightViewProjection[1], N, L, scene.shadowSettings.z);
        color += blinnPhong(N, V, L, radiance, albedo, specularMask);
    }
    // Emissive не умножается на свет, N.L, attenuation или ambient.
    color += texture(emissiveTexture, coordinate).rgb;
    outColor = vec4(color, 1.0);
}
