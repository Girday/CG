// Один слот общего dynamic UBO, выбранный dynamic offset при привязке set=1.
layout(std140, set = 1, binding = 0) uniform ModelData {
    mat4 model;
    vec4 albedo;
    vec4 specularShininess;
} object;
