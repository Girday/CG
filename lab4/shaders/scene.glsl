// Общие данные: std140, каждый vec4 занимает 16 байт.
layout(std140, set = 0, binding = 0) uniform SceneData {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;             // rgb: цвет, w: интенсивность
    vec4 directionalDirection;// xyz: направление распространения, w: интенсивность
    vec4 directionalColor;
    ivec4 lightCounts;        // x: point, y: spot, z: shadow-casting spot index or -1
    mat4 lightViewProjection[2]; // sun, spot
    vec4 shadowSettings;      // x: sun enabled, y: sun bias, z: spot bias
} scene;
