#include "scene.hpp"
#include "math3d.hpp"
#include "torus.hpp"
#include "lighting_math.hpp"
#include "gpu_resources.hpp"
#include "texture_math.hpp"
#include "shadow_maps.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include <imgui.h>

namespace scene {
namespace {

constexpr uint32_t object_count = 2;
constexpr float degrees_to_radians = std::numbers::pi_v<float> / 180.0f;

using gpu::Buffer;
using gpu::createBuffer;
using gpu::writeBuffer;
using gpu::destroyBuffer;

// Поля сгруппированы по 16 байт: одинаковые смещения в C++ и GLSL.
struct alignas(16) SceneData {
    math3d::Mat4 view_projection;
    std::array<float, 4> camera_position;
    std::array<float, 4> ambient;
    std::array<float, 4> directional_direction;
    std::array<float, 4> directional_color;
    std::array<int32_t, 4> light_counts;
    std::array<math3d::Mat4, 2> light_view_projection;
    std::array<float, 4> shadow_settings;
};
struct alignas(16) UniformData {
    math3d::Mat4 model;
    std::array<float, 4> albedo;
    std::array<float, 4> specular_shininess;
};
struct alignas(16) PointData {
    std::array<float, 4> position_intensity;
    std::array<float, 4> color;
};
struct alignas(16) SpotData {
    std::array<float, 4> position_intensity;
    std::array<float, 4> direction_inner;
    std::array<float, 4> color_outer;
};
struct alignas(16) SamplingData {
    int32_t effect;
    float time;
    float strength;
    float speed;
};
static_assert(sizeof(SceneData) == 288 && offsetof(SceneData, light_view_projection) == 144 &&
              offsetof(SceneData, shadow_settings) == 272);
static_assert(sizeof(UniformData) == 96 && offsetof(UniformData, specular_shininess) == 80);
static_assert(sizeof(PointData) == 32 && sizeof(SpotData) == 48 && sizeof(SamplingData) == 16);
static_assert(offsetof(SamplingData, time) == 4 && offsetof(SamplingData, strength) == 8 && offsetof(SamplingData, speed) == 12);

struct PointLight {
    float position[3]{0.0f, -2.0f, 2.0f};
    float intensity = 6.0f;
    float color[3]{1.0f, 0.45f, 0.16f};
};
struct SpotLight {
    float position[3]{-0.9f, -2.7f, 4.3f};
    float direction[3]{0.08f, 0.68f, -1.0f};
    float intensity = 36.0f;
    float color[3]{0.35f, 0.64f, 1.0f};
    float inner_angle = 24.0f;
    float outer_angle = 37.0f;
};

struct SceneObject {
	float position[3]{};
	float rotation[3]{}; // Градусы в интерфейсе.
	float scale[3]{0.85f, 0.85f, 0.85f};
	float tint[3]{1.0f, 1.0f, 1.0f};
	float specular[3]{0.8f, 0.8f, 0.8f};
    float shininess = 64.0f;
    int material = 0;
    int effect = 0;
    float effect_strength = 0.04f;
    float effect_speed = 1.0f;
    bool trajectory = false;
	bool playing = false;
	float phase = 0.0f;
	float speed = 0.8f;
	float radius = 0.35f;
	float height = 0.30f;
};

TorusMesh torus;
Buffer vertex_buffer;
Buffer index_buffer;
Buffer floor_vertices, floor_indices;
constexpr size_t floor_slot = object_count;
constexpr size_t first_marker_slot = floor_slot + 1;
constexpr uint32_t floor_index_count = 6;
std::array<SceneObject, object_count> objects;
VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
VkDescriptorSet scene_descriptor = VK_NULL_HANDLE;
Buffer scene_uniform, point_buffer, spot_buffer, model_uniform, cone_staging;
VkDeviceSize uniform_stride = 0;
float sampling_time = 0.0f;
bool animate_sampling = true;
struct Material {
    const char* name;
    const char* albedo_file;
    const char* specular_file;
    const char* emissive_file;
    gpu::Texture albedo, specular, emissive;
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
};
std::array<Material, 3> materials{{
    {"Porcelain / gold", "porcelain.png", "porcelain_specular.png", nullptr, {}, {}, {}},
    {"Circuit / emissive", "circuit.png", "circuit_specular.png", "circuit_emissive.png", {}, {}, {}},
    {"Woven / matte", "woven.png", nullptr, nullptr, {}, {}, {}},
}};
gpu::Texture black_texture;
VkSampler texture_sampler = VK_NULL_HANDLE;
Buffer cube_vertices, cube_indices, cone_vertices;
VkPipeline marker_pipeline = VK_NULL_HANDLE, cone_pipeline = VK_NULL_HANDLE;
std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;
std::vector<TorusVertex> cone_lines;
float camera_position[3]{0.0f, -7.0f, 4.8f};
float camera_yaw = 90.0f, camera_pitch = -29.7f;
float camera_speed = 3.0f;
float ambient_color[3]{0.70f, 0.80f, 1.0f};
float ambient_intensity = 0.17f;
float sun_direction[3]{0.6f, 0.32f, -1.0f};
float sun_color[3]{1.0f, 0.95f, 0.85f};
float sun_intensity = 0.95f;
std::array<shadows::DepthMap, 2> shadow_maps;
VkFormat shadow_format = VK_FORMAT_UNDEFINED;
VkSampler shadow_sampler = VK_NULL_HANDLE;
VkPipelineLayout shadow_pipeline_layout = VK_NULL_HANDLE;
VkPipeline shadow_pipeline = VK_NULL_HANDLE;
bool sun_shadows = true, spot_shadows = true;
float sun_ortho_width = 16.0f, sun_ortho_height = 16.0f;
float sun_bias = 0.0008f, spot_bias = 0.00025f;
int shadow_spot_index = 0;
bool show_light_markers = false, show_light_cones = false;
constexpr float panel_width = 360.0f;
float cone_length = 2.0f;
uint32_t cone_vertex_count = 0;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
bool orthographic_projection = false;
float field_of_view = 50.0f;
float ortho_height = 5.5f;
double previous_time = -1.0;

void destroyResources() {
	auto& context = graphics::internal::context;
    if (shadow_pipeline) vkDestroyPipeline(context.device, shadow_pipeline, nullptr);
    if (shadow_pipeline_layout) vkDestroyPipelineLayout(context.device, shadow_pipeline_layout, nullptr);
    shadow_pipeline = VK_NULL_HANDLE;
    shadow_pipeline_layout = VK_NULL_HANDLE;
	if (marker_pipeline) vkDestroyPipeline(context.device, marker_pipeline, nullptr);
    if (cone_pipeline) vkDestroyPipeline(context.device, cone_pipeline, nullptr);
    marker_pipeline = cone_pipeline = VK_NULL_HANDLE;
    if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
	if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	if (scene_layout) vkDestroyDescriptorSetLayout(context.device, scene_layout, nullptr);
	if (descriptor_layout) vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr);
	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	scene_layout = VK_NULL_HANDLE;
	descriptor_layout = VK_NULL_HANDLE;
    if (shadow_sampler) vkDestroySampler(context.device, shadow_sampler, nullptr);
    shadow_sampler = VK_NULL_HANDLE;
    for (auto& map : shadow_maps) shadows::destroy(map);

	destroyBuffer(floor_indices);
	destroyBuffer(floor_vertices);
	destroyBuffer(model_uniform);
    destroyBuffer(cone_staging);
    for (auto& material : materials) {
        gpu::destroyTexture(material.albedo);
        gpu::destroyTexture(material.specular);
        gpu::destroyTexture(material.emissive);
    }
    gpu::destroyTexture(black_texture);
    if (texture_sampler) vkDestroySampler(context.device, texture_sampler, nullptr);
    texture_sampler = VK_NULL_HANDLE;
	destroyBuffer(scene_uniform);
    destroyBuffer(point_buffer);
    destroyBuffer(spot_buffer);
    destroyBuffer(cube_vertices);
    destroyBuffer(cube_indices);
    destroyBuffer(cone_vertices);
    destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);
}

VkShaderModule loadShader(const char* path) {
	const auto shader_path = std::filesystem::path(CG_SHADER_DIRECTORY) / std::filesystem::path(path).filename();
	std::ifstream file(shader_path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Cannot open shader: " << path << '\n';
		return VK_NULL_HANDLE;
	}
	const std::streamsize byte_count = file.tellg();
	if (byte_count <= 0 || byte_count % 4 != 0) {
		std::cerr << "Invalid SPIR-V file: " << path << '\n';
		return VK_NULL_HANDLE;
	}
	std::vector<uint32_t> code(static_cast<size_t>(byte_count) / 4);
	file.seekg(0);
	if (!file.read(reinterpret_cast<char*>(code.data()), byte_count)) return VK_NULL_HANDLE;
	const VkShaderModuleCreateInfo create_info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = static_cast<size_t>(byte_count),
		.pCode = code.data(),
	};
	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &create_info, nullptr, &module) != VK_SUCCESS)
		std::cerr << "Cannot create shader module: " << path << '\n';
	return module;
}

bool createDescriptors() {
    auto& context = graphics::internal::context;
    const VkDescriptorSetLayoutBinding scene_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        // MoltenVK поддерживает сэмплеры сравнения в immutable bindings.
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, &shadow_sampler},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, &shadow_sampler},
    };
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = uint32_t(std::size(scene_bindings));
    layout_info.pBindings = scene_bindings;
    if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &scene_layout) != VK_SUCCESS) return false;
    const VkDescriptorSetLayoutBinding material_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    layout_info.bindingCount = 4;
    layout_info.pBindings = material_bindings;
    if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &descriptor_layout) != VK_SUCCESS) return false;
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, uint32_t(materials.size())},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, uint32_t(materials.size()) * 3 + uint32_t(shadow_maps.size())},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    const VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = uint32_t(materials.size()) + 1, .poolSizeCount = 4, .pPoolSizes = sizes,
    };
    if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) return false;
    const std::array<VkDescriptorSetLayout, 4> layouts{scene_layout, descriptor_layout, descriptor_layout, descriptor_layout};
    std::array<VkDescriptorSet, 4> sets{};
    const VkDescriptorSetAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool, .descriptorSetCount = uint32_t(layouts.size()), .pSetLayouts = layouts.data(),
    };
    if (vkAllocateDescriptorSets(context.device, &allocate_info, sets.data()) != VK_SUCCESS) return false;
    scene_descriptor = sets[0];
    for (uint32_t i = 0; i < shadow_maps.size(); ++i) {
        const VkDescriptorImageInfo image{VK_NULL_HANDLE, shadow_maps[i].view, shadows::sampled_layout};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = scene_descriptor;
        write.dstBinding = 3 + i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image;
        vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    }
    for (size_t i = 0; i < materials.size(); ++i) {
        auto& material = materials[i];
        material.descriptor = sets[i + 1];
        const VkDescriptorImageInfo images[]{
            {texture_sampler, material.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {texture_sampler, material.specular.view ? material.specular.view : black_texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {texture_sampler, material.emissive.view ? material.emissive.view : black_texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        for (uint32_t binding = 1; binding <= 3; ++binding) {
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = material.descriptor;
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &images[binding - 1];
            vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
        }
    }
    return true;
}

void bindBuffer(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                const Buffer& buffer, VkDeviceSize range) {
    const VkDescriptorBufferInfo info{buffer.handle, 0, range};
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set, .dstBinding = binding, .descriptorCount = 1,
        .descriptorType = type, .pBufferInfo = &info,
    };
    vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

// Вызывается только после ожидания fence в prepare(): старый буфер уже не читает GPU.
bool ensureCapacity(Buffer& buffer, VkDeviceSize required, VkBufferUsageFlags usage, bool device_local = false) {
    if (buffer.capacity >= required) return true;
    Buffer replacement;
    const auto size = std::max(required, std::max<VkDeviceSize>(buffer.capacity * 2, 64));
    if (!createBuffer(replacement, size, usage, device_local)) { destroyBuffer(replacement); return false; }
    destroyBuffer(buffer);
    buffer = replacement;
    return true;
}

enum class PipelineKind { Surface, Marker, Lines, Shadow };

bool createPipeline(VkPipeline& output, PipelineKind kind, const char* vertex_path, const char* fragment_path) {
	auto& context = graphics::internal::context;
    const bool depth_only = kind == PipelineKind::Shadow;
    const VkShaderModule vertex_shader = loadShader(vertex_path);
    const VkShaderModule fragment_shader = depth_only ? VK_NULL_HANDLE : loadShader(fragment_path);
	if (!vertex_shader || (!depth_only && !fragment_shader)) {
		if (vertex_shader) vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		if (fragment_shader) vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}
	const VkPipelineShaderStageCreateInfo stages[] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		 .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex_shader, .pName = "main"},
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		 .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment_shader, .pName = "main"},
	};
	const VkVertexInputBindingDescription binding{0, sizeof(TorusVertex), VK_VERTEX_INPUT_RATE_VERTEX};
	const VkVertexInputAttributeDescription attributes[] = {
		{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(TorusVertex, position)},
		{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(TorusVertex, normal)},
		{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(TorusVertex, uv)},
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = kind == PipelineKind::Surface ? 3u : 1u,
		.pVertexAttributeDescriptions = attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = kind == PipelineKind::Lines ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	const VkPipelineViewportStateCreateInfo viewport_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo rasterization{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisample{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineDepthStencilStateCreateInfo depth_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = depth_only ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_LESS_OR_EQUAL,
	};
	const VkPipelineColorBlendAttachmentState blend_attachment{
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo blend_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};
	const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	const VkPipelineDynamicStateCreateInfo dynamic_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};
    const VkPipelineRenderingCreateInfo rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .depthAttachmentFormat = shadow_format,
    };
	const VkGraphicsPipelineCreateInfo pipeline_info{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = depth_only ? &rendering : nullptr,
		.stageCount = depth_only ? 1u : 2u,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_state,
		.pColorBlendState = depth_only ? nullptr : &blend_state,
		.pDynamicState = &dynamic_state,
		.layout = depth_only ? shadow_pipeline_layout : pipeline_layout,
		.renderPass = depth_only ? VK_NULL_HANDLE : context.render_pass,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                &pipeline_info, nullptr, &output);
	if (fragment_shader) vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	return result == VK_SUCCESS;
}

math3d::Mat4 objectModel(const SceneObject& object) {
	math3d::Vec3 position{object.position[0], object.position[1], object.position[2]};
	float animated_turn = 0.0f;
	if (object.trajectory) {
		position.x += object.radius * std::cos(object.phase);
		position.y += object.radius * std::sin(object.phase);
		position.z += object.height * std::sin(2.0f * object.phase);
		animated_turn = object.phase;
	}
	using namespace math3d;
	const Mat4 rotation = multiply(rotationZ(object.rotation[2] * degrees_to_radians + animated_turn),
	                       multiply(rotationY(object.rotation[1] * degrees_to_radians),
	                                rotationX(object.rotation[0] * degrees_to_radians)));
	return multiply(translation(position),
	                multiply(rotation, scaling({object.scale[0], object.scale[1], object.scale[2]})));
}

float sceneViewportX() {
    const float display_width = ImGui::GetIO().DisplaySize.x;
    const float width = float(graphics::internal::context.swapchain_extent.width);
    return width * std::min(panel_width / std::max(display_width, 1.0f), 0.65f);
}

void buildConeLines() {
    using namespace math3d;
    cone_lines.clear();
    auto line = [](Vec3 a, Vec3 b) {
        cone_lines.push_back({{a.x, a.y, a.z}, {}, {}});
        cone_lines.push_back({{b.x, b.y, b.z}, {}, {}});
    };
    for (const auto& light : spot_lights) {
        const Vec3 origin{light.position[0], light.position[1], light.position[2]};
        const Vec3 d = lighting::safeDirection({light.direction[0], light.direction[1], light.direction[2]});
        const Vec3 helper = std::abs(d.z) < 0.9f ? Vec3{0, 0, 1} : Vec3{0, 1, 0};
        const Vec3 u = normalized(cross(d, helper)), v = cross(d, u);
        const Vec3 center{origin.x + d.x * cone_length, origin.y + d.y * cone_length, origin.z + d.z * cone_length};
        line(origin, center);
        // Внешний конус и внутреннее кольцо показывают оба угла прожектора.
        for (float angle : {light.inner_angle, light.outer_angle}) {
            const float radius = cone_length * std::tan(angle * degrees_to_radians);
            auto ringPoint = [&](int segment) {
                const float t = 2.0f * std::numbers::pi_v<float> * float(segment) / 32.0f;
                return Vec3{center.x + radius * (u.x * std::cos(t) + v.x * std::sin(t)),
                            center.y + radius * (u.y * std::cos(t) + v.y * std::sin(t)),
                            center.z + radius * (u.z * std::cos(t) + v.z * std::sin(t))};
            };
            for (int segment = 0; segment < 32; ++segment) {
                line(ringPoint(segment), ringPoint(segment + 1));
                if (segment % 8 == 0) line(origin, ringPoint(segment));
            }
        }
    }
    cone_vertex_count = static_cast<uint32_t>(cone_lines.size());
}

math3d::Mat4 lightView(math3d::Vec3 position, math3d::Vec3 direction) {
    // При почти вертикальном направлении Z не подходит в качестве up для LookAt.
    const math3d::Vec3 up = std::abs(direction.z) > 0.98f ? math3d::Vec3{0, 1, 0} : math3d::Vec3{0, 0, 1};
    return math3d::lookAt(position,
                          {position.x + direction.x, position.y + direction.y, position.z + direction.z}, up);
}

int activeShadowSpot() {
    const bool valid = shadow_spot_index >= 0 && static_cast<size_t>(shadow_spot_index) < spot_lights.size();
    return spot_shadows && valid ? shadow_spot_index : -1;
}

bool updateUniforms() {
    const auto extent = graphics::internal::context.swapchain_extent;
    if (extent.height == 0) return false;
    const float aspect = std::max(1.0f, float(extent.width) - sceneViewportX()) / float(extent.height);
    const auto forward = lighting::cameraForward(camera_yaw, camera_pitch);
    const math3d::Vec3 eye{camera_position[0], camera_position[1], camera_position[2]};
    const auto view = math3d::lookAt(eye, {eye.x + forward.x, eye.y + forward.y, eye.z + forward.z}, {0, 0, 1});
    const auto projection = orthographic_projection
        ? math3d::orthographic(ortho_height, aspect, 0.1f, 100.0f)
        : math3d::perspective(field_of_view * degrees_to_radians, aspect, 0.1f, 100.0f);
    const auto sun = lighting::safeDirection({sun_direction[0], sun_direction[1], sun_direction[2]});
    const auto sun_view = lightView({-15.0f * sun.x, -15.0f * sun.y, 0.8f - 15.0f * sun.z}, sun);
    const auto sun_projection = math3d::orthographic(sun_ortho_height, sun_ortho_width / sun_ortho_height,
                                                     0.1f, 32.0f);
    const int shadow_spot = activeShadowSpot();
    auto spot_view_projection = math3d::identity();
    if (shadow_spot >= 0) {
        const auto& spot = spot_lights[shadow_spot];
        const auto direction = lighting::safeDirection({spot.direction[0], spot.direction[1], spot.direction[2]});
        const auto spot_view = lightView({spot.position[0], spot.position[1], spot.position[2]}, direction);
        const float field_angle = 2.0f * std::max(spot.outer_angle, 0.5f) * degrees_to_radians;
        spot_view_projection = math3d::multiply(math3d::perspective(field_angle, 1.0f, 0.2f, 40.0f), spot_view);
    }
    const SceneData scene_data{
        math3d::multiply(projection, view),
        {eye.x, eye.y, eye.z, 1.0f},
        {ambient_color[0], ambient_color[1], ambient_color[2], ambient_intensity},
        {sun.x, sun.y, sun.z, sun_intensity},
        {sun_color[0], sun_color[1], sun_color[2], 1.0f},
        {static_cast<int32_t>(point_lights.size()), static_cast<int32_t>(spot_lights.size()), shadow_spot, 0},
        {math3d::multiply(sun_projection, sun_view), spot_view_projection},
        {sun_shadows ? 1.0f : 0.0f, sun_bias, spot_bias, 0.0f},
    };
    if (!writeBuffer(scene_uniform, &scene_data, sizeof(scene_data))) return false;
    std::vector<PointData> points;
    for (const auto& l : point_lights)
        points.push_back({{l.position[0], l.position[1], l.position[2], l.intensity}, {l.color[0], l.color[1], l.color[2], 0}});
    std::vector<SpotData> spots;
    for (const auto& l : spot_lights) {
        const auto d = lighting::safeDirection({l.direction[0], l.direction[1], l.direction[2]});
        spots.push_back({{l.position[0], l.position[1], l.position[2], l.intensity},
                         {d.x, d.y, d.z, std::cos(l.inner_angle * degrees_to_radians)},
                         {l.color[0], l.color[1], l.color[2], std::cos(l.outer_angle * degrees_to_radians)}});
    }
    const VkDeviceSize point_bytes = points.size() * sizeof(PointData), spot_bytes = spots.size() * sizeof(SpotData);
    if (!ensureCapacity(point_buffer, std::max<VkDeviceSize>(point_bytes, sizeof(PointData)), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        !ensureCapacity(spot_buffer, std::max<VkDeviceSize>(spot_bytes, sizeof(SpotData)), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
    if (!writeBuffer(point_buffer, points.data(), point_bytes) || !writeBuffer(spot_buffer, spots.data(), spot_bytes)) return false;
    bindBuffer(scene_descriptor, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_uniform, sizeof(SceneData));
    bindBuffer(scene_descriptor, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, point_buffer, point_buffer.capacity);
    bindBuffer(scene_descriptor, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spot_buffer, spot_buffer.capacity);
    const size_t slots = first_marker_slot + point_lights.size() + spot_lights.size() + 1;
    if (slots > std::numeric_limits<uint32_t>::max() / uniform_stride ||
        !ensureCapacity(model_uniform, slots * uniform_stride, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;
    for (const auto& material : materials)
        bindBuffer(material.descriptor, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, model_uniform, sizeof(UniformData));
    size_t slot = 0;
    for (const auto& object : objects) {
        const UniformData data{objectModel(object),
            {object.tint[0], object.tint[1], object.tint[2], 1.0f},
            {object.specular[0], object.specular[1], object.specular[2], object.shininess}};
        if (!writeBuffer(model_uniform, &data, sizeof(data), uniform_stride * slot++)) return false;
    }
    const UniformData floor{math3d::identity(), {0.63f, 0.69f, 0.77f, 0.18f}, {0.0f, 0.0f, 0.0f, 32.0f}};
    if (!writeBuffer(model_uniform, &floor, sizeof(floor), uniform_stride * slot++)) return false;
    auto writeMarker = [&](const float* position, const float* color) {
        const UniformData data{
            math3d::multiply(math3d::translation({position[0], position[1], position[2]}), math3d::scaling({0.075f, 0.075f, 0.075f})),
            {color[0], color[1], color[2], 1.0f}, {}};
        return writeBuffer(model_uniform, &data, sizeof(data), uniform_stride * slot++);
    };
    for (const auto& light : point_lights) if (!writeMarker(light.position, light.color)) return false;
    for (const auto& light : spot_lights) if (!writeMarker(light.position, light.color)) return false;
    const UniformData cone_data{math3d::identity(), {1, 1, 1, 1}, {}};
    if (!writeBuffer(model_uniform, &cone_data, sizeof(cone_data), uniform_stride * slot)) return false;
    cone_vertex_count = 0;
    if (show_light_markers && show_light_cones) {
        buildConeLines();
        const VkDeviceSize bytes = cone_lines.size() * sizeof(TorusVertex);
        if (bytes > 0 && (!ensureCapacity(cone_vertices, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true) ||
                          !ensureCapacity(cone_staging, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
                          !writeBuffer(cone_staging, cone_lines.data(), bytes))) return false;
    }
    return true;
}

bool createMarkerGeometry() {
    std::array<TorusVertex, 8> vertices{};
    for (int i = 0; i < 8; ++i) {
        vertices[i].position[0] = (i & 1) ? 1.0f : -1.0f;
        vertices[i].position[1] = (i & 2) ? 1.0f : -1.0f;
        vertices[i].position[2] = (i & 4) ? 1.0f : -1.0f;
    }
    const uint32_t indices[] = {0,2,3,0,3,1,4,5,7,4,7,6,0,1,5,0,5,4,2,6,7,2,7,3,0,4,6,0,6,2,1,3,7,1,7,5};
    return gpu::uploadGeometry(cube_vertices, vertices.data(), sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) &&
           gpu::uploadGeometry(cube_indices, indices, sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

bool createFloorGeometry() {
    constexpr TorusVertex vertices[]{
        {{-10.0f, -10.0f, 0.0f}, {0, 0, 1}, {0, 0}},
        {{ 10.0f, -10.0f, 0.0f}, {0, 0, 1}, {4, 0}},
        {{ 10.0f,  10.0f, 0.0f}, {0, 0, 1}, {4, 4}},
        {{-10.0f,  10.0f, 0.0f}, {0, 0, 1}, {0, 4}},
    };
    constexpr uint32_t indices[floor_index_count]{0, 1, 2, 0, 2, 3};
    return gpu::uploadGeometry(floor_vertices, vertices, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) &&
           gpu::uploadGeometry(floor_indices, indices, sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

} // namespace

bool initialize() {
	orthographic_projection = std::getenv("CG_START_ORTHOGRAPHIC") != nullptr;
	torus = generateTorus(1.0f, 0.35f, 48, 24);
	objects = {};
	objects[0].position[0] = -1.45f;
	objects[0].position[2] = 1.03f;
	objects[0].rotation[0] = 54.0f;
    objects[1].position[0] = 1.45f;
    objects[1].position[2] = 1.18f;
    objects[1].rotation[0] = 66.0f;
    objects[1].rotation[1] = 18.0f;
    objects[1].material = 1;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
    uniform_stride = texture_math::uniformStride(sizeof(UniformData), properties.limits.minUniformBufferOffsetAlignment);
    std::cout << "Model UBO: structure=" << sizeof(UniformData) << ", stride=" << uniform_stride << '\n';
    const auto texture_dir = std::getenv("CG_TEXTURE_DIR") ? std::filesystem::path(std::getenv("CG_TEXTURE_DIR"))
                                                       : std::filesystem::path(CG_TEXTURE_DIRECTORY);
    if (!gpu::loadTexture(black_texture, texture_dir / "black.png", false)) { destroyResources(); return false; }
    for (auto& material : materials) {
        if (!gpu::loadTexture(material.albedo, texture_dir / material.albedo_file, true) ||
            (material.specular_file && !gpu::loadTexture(material.specular, texture_dir / material.specular_file, false)) ||
            (material.emissive_file && !gpu::loadTexture(material.emissive, texture_dir / material.emissive_file, true))) {
            destroyResources(); return false;
        }
    }
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.maxLod = 0.0f;
    if (vkCreateSampler(graphics::internal::context.device, &sampler, nullptr, &texture_sampler) != VK_SUCCESS) {
        destroyResources(); return false;
    }
    point_lights = {PointLight{}};
    spot_lights = {SpotLight{}};
    shadow_format = shadows::selectFormat();
    if (shadow_format == VK_FORMAT_UNDEFINED || !shadows::createSampler(shadow_sampler) ||
        !shadows::create(shadow_maps[0], shadow_format) || !shadows::create(shadow_maps[1], shadow_format)) {
        std::cerr << "Failed to create sampled depth maps\n";
        destroyResources();
        return false;
    }
    if (!createBuffer(scene_uniform, sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
        !createMarkerGeometry() || !createFloorGeometry()) {
        destroyResources(); return false;
    }
	const VkDeviceSize vertex_bytes = torus.vertices.size() * sizeof(TorusVertex);
	const VkDeviceSize index_bytes = torus.indices.size() * sizeof(uint32_t);
    if (!gpu::uploadGeometry(vertex_buffer, torus.vertices.data(), vertex_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
        !gpu::uploadGeometry(index_buffer, torus.indices.data(), index_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) {
        destroyResources(); return false;
    }
	if (!createDescriptors()) { destroyResources(); return false; }
    const VkDescriptorSetLayout layouts[]{scene_layout, descriptor_layout};
    const VkPushConstantRange push_range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SamplingData)};
    const VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2, .pSetLayouts = layouts,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range,
    };
    const VkPushConstantRange shadow_push_range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(uint32_t)};
    auto shadow_layout_info = layout_info;
    shadow_layout_info.pPushConstantRanges = &shadow_push_range;
    if (vkCreatePipelineLayout(graphics::internal::context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS ||
        vkCreatePipelineLayout(graphics::internal::context.device, &shadow_layout_info, nullptr, &shadow_pipeline_layout) != VK_SUCCESS ||
        !createPipeline(pipeline, PipelineKind::Surface, "torus.vert.spv", "torus.frag.spv") ||
        !createPipeline(marker_pipeline, PipelineKind::Marker, "marker.vert.spv", "marker.frag.spv") ||
        !createPipeline(cone_pipeline, PipelineKind::Lines, "marker.vert.spv", "marker.frag.spv") ||
        !createPipeline(shadow_pipeline, PipelineKind::Shadow, "shadow.vert.spv", nullptr)) {
		std::cerr << "Failed to create scene or shadow pipeline\n";
		destroyResources();
		return false;
	}
	return true;
}

void shutdown() {
	vkDeviceWaitIdle(graphics::internal::context.device);
	destroyResources();
}

void update(double time) {
	const float delta = previous_time < 0.0 ? 0.0f :
		std::clamp(float(time - previous_time), 0.0f, 0.1f);
	previous_time = time;
    if (animate_sampling) sampling_time += delta;
	for (auto& object : objects)
		if (object.trajectory && object.playing) object.phase += object.speed * delta;

    auto& io = ImGui::GetIO();
    // Ввод камеры активен только справа от панели и с зажатой правой кнопкой мыши.
    if (!io.WantCaptureMouse && io.MousePos.x > panel_width && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        camera_yaw -= io.MouseDelta.x * 0.20f;
        camera_pitch = std::clamp(camera_pitch - io.MouseDelta.y * 0.20f, -89.0f, 89.0f);
        if (!io.WantCaptureKeyboard) {
            const auto f = lighting::cameraForward(camera_yaw, camera_pitch);
            const auto right = lighting::safeDirection(math3d::cross(f, {0, 0, 1}));
            const float forward = float(ImGui::IsKeyDown(ImGuiKey_W)) - float(ImGui::IsKeyDown(ImGuiKey_S));
            const float side = float(ImGui::IsKeyDown(ImGuiKey_D)) - float(ImGui::IsKeyDown(ImGuiKey_A));
            const float up = float(ImGui::IsKeyDown(ImGuiKey_E)) - float(ImGui::IsKeyDown(ImGuiKey_Q));
            camera_position[0] += camera_speed * delta * (f.x * forward + right.x * side);
            camera_position[1] += camera_speed * delta * (f.y * forward + right.y * side);
            camera_position[2] += camera_speed * delta * (f.z * forward + up);
        }
    }
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(std::min(panel_width, io.DisplaySize.x * 0.65f), io.DisplaySize.y));
    ImGui::Begin("Lab 4 - Shadow mapping", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.48f);
    ImGui::Text("Variant 7: Torus | %.0f FPS", io.Framerate);
    if (ImGui::BeginTabBar("Controls")) {
        if (ImGui::BeginTabItem("Shadows")) {
            ImGui::Checkbox("Sun shadows", &sun_shadows);
            ImGui::DragFloat3("Sun ray direction", sun_direction, 0.02f, -1.0f, 1.0f);
            ImGui::SliderFloat("Ortho width", &sun_ortho_width, 1.0f, 30.0f, "%.1f");
            ImGui::SliderFloat("Ortho height", &sun_ortho_height, 1.0f, 30.0f, "%.1f");
            ImGui::SliderFloat("Sun depth bias", &sun_bias, 0.0f, 0.01f, "%.5f");
            ImGui::Separator();
            ImGui::Checkbox("Spotlight shadows", &spot_shadows);
            if (spot_lights.empty()) {
                ImGui::TextDisabled("Add a light in the Spot tab.");
            } else {
                const std::string selected = "Spot " + std::to_string(shadow_spot_index + 1);
                if (ImGui::BeginCombo("Shadow caster", selected.c_str())) {
                    for (size_t i = 0; i < spot_lights.size(); ++i) {
                        const std::string label = "Spot " + std::to_string(i + 1);
                        if (ImGui::Selectable(label.c_str(), int(i) == shadow_spot_index)) shadow_spot_index = int(i);
                    }
                    ImGui::EndCombo();
                }
            }
            ImGui::SliderFloat("Spot depth bias", &spot_bias, 0.0f, 0.01f, "%.5f");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Textures")) {
            ImGui::Checkbox("Animate texture effects", &animate_sampling);
            if (ImGui::Button("Reset effect time")) sampling_time = 0;
            if (ImGui::Button("Lighting off")) {
                ambient_intensity = sun_intensity = 0;
                for (auto& l : point_lights) l.intensity = 0;
                for (auto& l : spot_lights) l.intensity = 0;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset lights")) {
                ambient_intensity = 0.17f; sun_intensity = 0.95f;
                point_lights = {PointLight{}}; spot_lights = {SpotLight{}};
                shadow_spot_index = 0;
            }
            for (uint32_t i = 0; i < object_count; ++i) {
                ImGui::PushID(int(i));
                auto& o = objects[i];
                ImGui::SeparatorText(i == 0 ? "Torus 1" : "Torus 2");
                if (ImGui::BeginCombo("Material", materials[o.material].name)) {
                    for (int m = 0; m < int(materials.size()); ++m) {
                        if (ImGui::Selectable(materials[m].name, o.material == m)) o.material = m;
                        if (o.material == m) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::Combo("Sampling", &o.effect, "Original\0Ripple / RGB echo\0Wobble / box blur\0");
                ImGui::SliderFloat("Strength", &o.effect_strength, 0.0f, 0.15f, "%.3f");
                ImGui::SliderFloat("Effect speed", &o.effect_speed, 0.0f, 3.0f);
                ImGui::TextDisabled("Specular: %s | Emissive: %s", materials[o.material].specular_file ? "map" : "black 1x1",
                                    materials[o.material].emissive_file ? "map" : "black 1x1");
                ImGui::PopID();
            }
            ImGui::TextWrapped("Emissive is added after lighting. Switch lighting off to inspect its map. Woven has no specular or emissive map.");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Scene")) {
            if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat3("Position", camera_position, 0.05f, -30.0f, 30.0f);
                ImGui::SliderFloat("Yaw", &camera_yaw, -180.0f, 180.0f);
                ImGui::SliderFloat("Pitch", &camera_pitch, -89.0f, 89.0f);
                if (ImGui::Button("Reset camera")) {
                    camera_position[0] = 0; camera_position[1] = -7; camera_position[2] = 4.8f;
                    camera_yaw = 90; camera_pitch = -29.7f;
                }
                ImGui::Checkbox("Orthographic", &orthographic_projection);
                ImGui::SliderFloat("Field of view", &field_of_view, 15.0f, 110.0f);
                if (orthographic_projection) ImGui::SliderFloat("Ortho height", &ortho_height, 2.0f, 15.0f);
                ImGui::SliderFloat("Move speed", &camera_speed, 0.2f, 12.0f);
                ImGui::TextWrapped("Hold RMB over the scene: mouse look, WASD move, Q/E down/up.");
            }
            if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::ColorEdit3("Ambient color", ambient_color);
                ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 1.0f);
            }
            if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat3("Ray direction", sun_direction, 0.02f, -1.0f, 1.0f);
                ImGui::ColorEdit3("Sun color", sun_color);
                ImGui::SliderFloat("Sun intensity", &sun_intensity, 0.0f, 3.0f);
                ImGui::TextWrapped("Direction points FROM the light towards the scene.");
            }
            ImGui::Checkbox("Show light markers", &show_light_markers);
            ImGui::Checkbox("Show spotlight cones", &show_light_cones);
            ImGui::SliderFloat("Cone display length", &cone_length, 0.2f, 5.0f);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Models")) {
            for (uint32_t i = 0; i < object_count; ++i) {
                ImGui::PushID(static_cast<int>(i));
                auto& object = objects[i];
                if (ImGui::CollapsingHeader(i == 0 ? "Torus 1" : "Torus 2", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragFloat3("Position", object.position, 0.02f, -6.0f, 6.0f);
                    ImGui::SliderFloat3("Rotation (deg)", object.rotation, -180.0f, 180.0f);
                    ImGui::SliderFloat3("Scale", object.scale, 0.1f, 2.0f);
                    ImGui::ColorEdit3("Texture tint", object.tint);
                    ImGui::ColorEdit3("Specular color", object.specular);
                    ImGui::SliderFloat("Shininess", &object.shininess, 1.0f, 256.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
                    ImGui::Checkbox("Trajectory", &object.trajectory);
                    if (object.trajectory) {
                        ImGui::Checkbox("Playing", &object.playing);
                        if (ImGui::Button("Reset phase")) object.phase = 0;
                        ImGui::SliderFloat("Speed", &object.speed, 0.0f, 3.0f);
                        ImGui::SliderFloat("Radius", &object.radius, 0.0f, 1.0f);
                        ImGui::SliderFloat("Height", &object.height, 0.0f, 1.0f);
                    }
                }
                ImGui::PopID();
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Point")) {
            if (ImGui::Button("Add point light")) point_lights.emplace_back();
            ImGui::SameLine();
            if (ImGui::Button("Clear all")) point_lights.clear();
            ImGui::TextWrapped("Attenuation: 1 / distance squared.");
            for (size_t i = 0; i < point_lights.size();) {
                ImGui::PushID(static_cast<int>(i));
                auto& l = point_lights[i];
                const std::string title = "Point " + std::to_string(i + 1);
                bool remove = false;
                if (ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragFloat3("Position", l.position, 0.04f, -20.0f, 20.0f);
                    ImGui::ColorEdit3("Color", l.color);
                    ImGui::SliderFloat("Intensity", &l.intensity, 0.0f, 50.0f);
                    remove = ImGui::Button("Remove");
                }
                ImGui::PopID();
                if (remove) point_lights.erase(point_lights.begin() + static_cast<std::ptrdiff_t>(i)); else ++i;
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Spot")) {
            if (ImGui::Button("Add spotlight")) spot_lights.emplace_back();
            ImGui::SameLine();
            if (ImGui::Button("Clear all")) { spot_lights.clear(); shadow_spot_index = 0; }
            for (size_t i = 0; i < spot_lights.size();) {
                ImGui::PushID(static_cast<int>(i));
                auto& l = spot_lights[i];
                const std::string title = "Spot " + std::to_string(i + 1);
                bool remove = false;
                if (ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragFloat3("Position", l.position, 0.04f, -20.0f, 20.0f);
                    ImGui::DragFloat3("Direction", l.direction, 0.02f, -1.0f, 1.0f);
                    ImGui::ColorEdit3("Color", l.color);
                    ImGui::SliderFloat("Intensity", &l.intensity, 0.0f, 80.0f);
                    ImGui::SliderFloat("Inner half-angle", &l.inner_angle, 0.0f, l.outer_angle);
                    ImGui::SliderFloat("Outer half-angle", &l.outer_angle, l.inner_angle, 80.0f);
                    ImGui::TextWrapped("Equal angles: hard edge. Larger outer angle: soft transition.");
                    remove = ImGui::Button("Remove");
                }
                ImGui::PopID();
                if (remove) {
                    spot_lights.erase(spot_lights.begin() + static_cast<std::ptrdiff_t>(i));
                    if (int(i) < shadow_spot_index) --shadow_spot_index;
                    shadow_spot_index = std::clamp(shadow_spot_index, 0, std::max(0, int(spot_lights.size()) - 1));
                } else ++i;
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::PopItemWidth();
    ImGui::End();
    // Воспроизводимая проверка роста SSBO, пустых массивов и равных углов.
    if (std::getenv("CG_SMOKE_LIGHTS")) {
        static int frame = 0;
        ++frame;
        if (frame == 5) { point_lights.clear(); spot_lights.clear(); }
        if (frame == 10) { point_lights.resize(17); spot_lights.resize(12); show_light_markers = show_light_cones = true; }
        if (frame == 20) { point_lights.resize(65); spot_lights.resize(25); }
        if (frame == 35) { point_lights.clear(); spot_lights.clear(); }
        if (frame == 45) { point_lights.emplace_back(); spot_lights.emplace_back(); spot_lights[0].inner_angle = spot_lights[0].outer_angle; }
        if (frame == 55) { spot_lights[0].direction[0] = spot_lights[0].direction[1] = spot_lights[0].direction[2] = 0; }
        if (frame == 60) { point_lights = {PointLight{}}; spot_lights = {SpotLight{}}; orthographic_projection = true; }
        if (frame == 70) { orthographic_projection = false; camera_yaw += 5; }
    }
    if (std::getenv("CG_SMOKE_TEXTURES")) {
        static int frame = 0;
        ++frame;
        if (frame == 5) {
            for (auto& object : objects) { object.material = 0; object.effect = 0; }
            std::cout << "Texture smoke: shared material, different dynamic offsets\n";
        }
        if (frame == 20) {
            for (auto& object : objects) object.material = 1;
            objects[0].effect = 1; objects[1].effect = 2;
            std::cout << "Texture smoke: ripple/RGB and wobble/box blur\n";
        }
        if (frame == 40) {
            for (auto& object : objects) object.material = 2;
            objects[0].effect = 0;
            std::cout << "Texture smoke: black specular/emissive fallback\n";
        }
        if (frame == 80) {
            ambient_intensity = sun_intensity = 0;
            for (auto& light : point_lights) light.intensity = 0;
            for (auto& light : spot_lights) light.intensity = 0;
            objects[0].material = 0; objects[1].material = 1;
            objects[0].effect = objects[1].effect = 0;
            std::cout << "Texture smoke: emissive with all lights off\n";
        }
    }
    if (std::getenv("CG_SMOKE_SHADOWS")) {
        static int frame = 0;
        ++frame;
        if (frame == 10) { spot_shadows = false; }
        if (frame == 25) { sun_shadows = false; spot_shadows = true; }
        if (frame == 40) { sun_shadows = true; spot_shadows = true; }
        if (frame == 50) { spot_lights.clear(); shadow_spot_index = 0; }
        if (frame == 65) { spot_lights.resize(2); shadow_spot_index = 1; }
        if (frame == 80) { sun_direction[0] = sun_direction[1] = 0.0f; sun_direction[2] = -1.0f; }
        if (frame == 90) { sun_ortho_width = 6.0f; sun_ortho_height = 5.0f; sun_bias = 0.0f; }
        if (frame == 105) { objects[0].trajectory = objects[0].playing = true; }
        if (frame == 140) std::cout << "Shadow smoke: sun, spot selection, empty lights, vertical sun, bias and movement\n";
    }
}

void render(const graphics::internal::FrameData& frame) {
	const auto& context = graphics::internal::context;
	const VkCommandBuffer command = frame.command_buffer;
	// prepare() дождался fence прошлого кадра; UBO можно обновить без гонки с GPU.
	if (!updateUniforms()) { std::cerr << "Failed to update uniform buffers\n"; std::abort(); }
	if (vkResetCommandBuffer(frame.command_buffer, 0) != VK_SUCCESS) return;
	const VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	if (vkBeginCommandBuffer(frame.command_buffer, &begin_info) != VK_SUCCESS) return;
	if (cone_vertex_count)
        gpu::copyGeometry(frame.command_buffer, cone_staging, cone_vertices, cone_vertex_count * sizeof(TorusVertex));

    auto bindModel = [&](VkPipelineLayout layout, size_t slot, VkDescriptorSet descriptor) {
        const uint32_t offset = uint32_t(slot * uniform_stride);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &descriptor, 1, &offset);
    };
    auto drawSurfaces = [&](VkPipelineLayout layout, bool use_materials) {
        auto draw = [&](size_t slot, int material, uint32_t index_count, const SamplingData& sampling) {
            bindModel(layout, slot, materials[material].descriptor);
            if (use_materials)
                vkCmdPushConstants(command, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(sampling), &sampling);
            vkCmdDrawIndexed(command, index_count, 1, 0, 0, 0);
        };
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command, 0, 1, &vertex_buffer.handle, &offset);
        vkCmdBindIndexBuffer(command, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
        for (size_t i = 0; i < objects.size(); ++i) {
            const auto& object = objects[i];
            draw(i, object.material, uint32_t(torus.indices.size()),
                 {object.effect, sampling_time, object.effect_strength, object.effect_speed});
        }
        vkCmdBindVertexBuffers(command, 0, 1, &floor_vertices.handle, &offset);
        vkCmdBindIndexBuffer(command, floor_indices.handle, 0, VK_INDEX_TYPE_UINT32);
        draw(floor_slot, 2, floor_index_count, {}); // Woven, без деформации UV.
    };

    const VkRect2D shadow_scissor{{0, 0}, shadows::extent};
    const VkViewport shadow_viewport{0, 0, float(shadows::extent.width), float(shadows::extent.height), 0, 1};
    for (uint32_t light = 0; light < shadow_maps.size(); ++light) {
        auto& map = shadow_maps[light];
        shadows::begin(command, map);
        if (light == 0 || activeShadowSpot() >= 0) {
            vkCmdSetViewport(command, 0, 1, &shadow_viewport);
            vkCmdSetScissor(command, 0, 1, &shadow_scissor);
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout,
                                    0, 1, &scene_descriptor, 0, nullptr);
            vkCmdPushConstants(command, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(light), &light);
            drawSurfaces(shadow_pipeline_layout, false);
        }
        shadows::end(command, map);
    }

    const VkClearValue clear_values[] = {
		{.color = {{0.012f, 0.020f, 0.038f, 1.0f}}},
		{.depthStencil = {1.0f, 0}},
	};
	const VkRenderPassBeginInfo render_pass_info{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = frame.framebuffer,
		.renderArea = {{0, 0}, context.swapchain_extent},
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(frame.command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
	const float viewport_x = sceneViewportX();
	const VkViewport viewport{viewport_x, 0.0f, float(context.swapchain_extent.width) - viewport_x,
	                          float(context.swapchain_extent.height), 0.0f, 1.0f};
	const VkRect2D scissor{{0, 0}, context.swapchain_extent};
	vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout, 0, 1, &scene_descriptor, 0, nullptr);
    drawSurfaces(pipeline_layout, true);
    if (show_light_markers) {
        vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, marker_pipeline);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &cube_vertices.handle, &offset);
        vkCmdBindIndexBuffer(frame.command_buffer, cube_indices.handle, 0, VK_INDEX_TYPE_UINT32);
        size_t slot = first_marker_slot;
        auto marker = [&]() {
            bindModel(pipeline_layout, slot++, materials[0].descriptor);
            vkCmdDrawIndexed(frame.command_buffer, 36, 1, 0, 0, 0);
        };
        for (size_t i = 0; i < point_lights.size() + spot_lights.size(); ++i) marker();
        if (cone_vertex_count > 0) {
            vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cone_pipeline);
            vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &cone_vertices.handle, &offset);
            bindModel(pipeline_layout, slot, materials[0].descriptor);
            vkCmdDraw(frame.command_buffer, cone_vertex_count, 1, 0, 0);
        }
    }
    vkCmdEndRenderPass(frame.command_buffer);
	if (vkEndCommandBuffer(frame.command_buffer) != VK_SUCCESS)
		std::cerr << "Failed to record draw commands\n";
}

} // namespace scene
