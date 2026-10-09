#include "scene.hpp"
#include "math3d.hpp"
#include "torus.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numbers>
#include <vector>

#include <imgui.h>

namespace scene {
namespace {

constexpr uint32_t object_count = 2;
constexpr float degrees_to_radians = std::numbers::pi_v<float> / 180.0f;

struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

struct alignas(16) UniformData {
	math3d::Mat4 model;
	math3d::Mat4 view;
	math3d::Mat4 projection;
	std::array<float, 4> tint;
};
static_assert(sizeof(UniformData) == 208);

struct SceneObject {
	float position[3]{};
	float rotation[3]{}; // Градусы в интерфейсе.
	float scale[3]{0.65f, 0.65f, 0.65f};
	float tint[3]{1.0f, 1.0f, 1.0f};
	bool trajectory = false;
	bool playing = false;
	float phase = 0.0f;
	float speed = 0.8f;
	float radius = 0.35f;
	float height = 0.30f;
	Buffer uniform;
	VkDescriptorSet descriptor = VK_NULL_HANDLE;
};

TorusMesh torus;
Buffer vertex_buffer;
Buffer index_buffer;
std::array<SceneObject, object_count> objects;
VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
bool orthographic_projection = false;
bool show_imgui_demo = false;
float field_of_view = 50.0f;
float ortho_height = 5.5f;
double previous_time = -1.0;

bool createBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage) {
	auto& context = graphics::internal::context;
	const VkBufferCreateInfo create_info{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info{
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo result_info{};
	if (vmaCreateBuffer(context.allocator, &create_info, &allocation_info,
	                    &buffer.handle, &buffer.allocation, &result_info) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}
	buffer.mapped = result_info.pMappedData;
	return buffer.mapped != nullptr;
}

bool writeBuffer(const Buffer& buffer, const void* source, VkDeviceSize size) {
	std::memcpy(buffer.mapped, source, static_cast<size_t>(size));
	return vmaFlushAllocation(graphics::internal::context.allocator,
	                          buffer.allocation, 0, size) == VK_SUCCESS;
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.handle != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle, buffer.allocation);
		buffer = {};
	}
}

void destroyResources() {
	auto& context = graphics::internal::context;
	if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
	if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	if (descriptor_layout) vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr);
	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	descriptor_layout = VK_NULL_HANDLE;
	for (auto& object : objects) destroyBuffer(object.uniform);
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);
}

VkShaderModule loadShader(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
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
	const VkDescriptorSetLayoutBinding uniform_binding{
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo layout_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &uniform_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &descriptor_layout) != VK_SUCCESS)
		return false;
	const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, object_count};
	const VkDescriptorPoolCreateInfo pool_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = object_count,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS)
		return false;
	const std::array<VkDescriptorSetLayout, object_count> layouts{descriptor_layout, descriptor_layout};
	std::array<VkDescriptorSet, object_count> sets{};
	const VkDescriptorSetAllocateInfo allocate_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = object_count,
		.pSetLayouts = layouts.data(),
	};
	if (vkAllocateDescriptorSets(context.device, &allocate_info, sets.data()) != VK_SUCCESS)
		return false;
	for (uint32_t i = 0; i < object_count; ++i) {
		objects[i].descriptor = sets[i];
		const VkDescriptorBufferInfo buffer_info{objects[i].uniform.handle, 0, sizeof(UniformData)};
		const VkWriteDescriptorSet write{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
		};
		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}
	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;
	const VkPipelineLayoutCreateInfo layout_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_layout,
	};
	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
		return false;
	const VkShaderModule vertex_shader = loadShader("shaders/torus.vert.spv");
	const VkShaderModule fragment_shader = loadShader("shaders/torus.frag.spv");
	if (!vertex_shader || !fragment_shader) {
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
		{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(TorusVertex, color)},
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 3,
		.pVertexAttributeDescriptions = attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
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
	const VkGraphicsPipelineCreateInfo pipeline_info{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_state,
		.pColorBlendState = &blend_state,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                &pipeline_info, nullptr, &pipeline);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
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

bool updateUniforms() {
	const auto extent = graphics::internal::context.swapchain_extent;
	if (extent.height == 0) return false;
	const float aspect = float(extent.width) / float(extent.height);
	// На более узком окне оставляем место для панели управления.
	const float focus_x = -1.40f - 2.3f * std::max(0.0f, 16.0f / 9.0f - aspect);
	const auto view = math3d::lookAt({focus_x, -5.8f, 3.4f}, {focus_x, 0.0f, 0.0f},
	                                 {0.0f, 0.0f, 1.0f});
	const auto projection = orthographic_projection
		? math3d::orthographic(ortho_height, aspect, 0.1f, 100.0f)
		: math3d::perspective(field_of_view * degrees_to_radians, aspect, 0.1f, 100.0f);
	for (auto& object : objects) {
		const UniformData data{
			.model = objectModel(object),
			.view = view,
			.projection = projection,
			.tint = {object.tint[0], object.tint[1], object.tint[2], 1.0f},
		};
		if (!writeBuffer(object.uniform, &data, sizeof(data))) return false;
	}
	return true;
}

} // namespace

bool initialize() {
	orthographic_projection = std::getenv("CG_START_ORTHOGRAPHIC") != nullptr;
	torus = generateTorus(1.0f, 0.35f, 48, 24);
	objects = {};
	objects[0].position[0] = -1.45f;
	objects[0].rotation[0] = 28.0f;
	objects[0].tint[0] = 1.0f;
	objects[0].tint[1] = 0.75f;
	objects[0].tint[2] = 0.45f;
	objects[1].position[0] = 1.45f;
	objects[1].rotation[0] = -22.0f;
	objects[1].tint[0] = 0.50f;
	objects[1].tint[1] = 0.85f;
	objects[1].tint[2] = 1.0f;
	objects[1].trajectory = true;
	objects[1].playing = true;
	const VkDeviceSize vertex_bytes = torus.vertices.size() * sizeof(TorusVertex);
	const VkDeviceSize index_bytes = torus.indices.size() * sizeof(uint32_t);
	if (!createBuffer(vertex_buffer, vertex_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
	    !writeBuffer(vertex_buffer, torus.vertices.data(), vertex_bytes) ||
	    !createBuffer(index_buffer, index_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT) ||
	    !writeBuffer(index_buffer, torus.indices.data(), index_bytes)) {
		destroyResources();
		return false;
	}
	for (auto& object : objects) {
		if (!createBuffer(object.uniform, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
			destroyResources();
			return false;
		}
	}
	if (!createDescriptors() || !createPipeline()) {
		std::cerr << "Failed to create torus descriptors or graphics pipeline\n";
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
	for (auto& object : objects)
		if (object.trajectory && object.playing) object.phase += object.speed * delta;

	ImGui::SetNextWindowSize(ImVec2(390, 610), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 1 - Torus");
	ImGui::Text("Variant 7: %zu vertices, %zu triangles, %u objects",
	            torus.vertices.size(), torus.indices.size() / 3, object_count);
	int projection_choice = orthographic_projection ? 1 : 0;
	if (ImGui::Combo("Projection", &projection_choice, "Perspective\0Orthographic\0"))
		orthographic_projection = projection_choice == 1;
	if (orthographic_projection)
		ImGui::SliderFloat("Ortho height", &ortho_height, 3.0f, 12.0f);
	else
		ImGui::SliderFloat("Field of view", &field_of_view, 25.0f, 85.0f);
	ImGui::Separator();
	for (uint32_t i = 0; i < object_count; ++i) {
		ImGui::PushID(static_cast<int>(i));
		auto& object = objects[i];
		const char* title = i == 0 ? "Torus 1" : "Torus 2";
		if (ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Position", object.position, 0.02f, -4.0f, 4.0f);
			ImGui::SliderFloat3("Rotation (deg)", object.rotation, -180.0f, 180.0f);
			ImGui::SliderFloat3("Scale", object.scale, 0.2f, 1.5f);
			ImGui::ColorEdit3("Color multiplier", object.tint);
			ImGui::Checkbox("Trajectory", &object.trajectory);
			if (object.trajectory) {
				if (ImGui::Button(object.playing ? "Pause" : "Play")) object.playing = !object.playing;
				ImGui::SameLine();
				if (ImGui::Button("Reset")) object.phase = 0.0f;
				ImGui::SliderFloat("Speed", &object.speed, 0.0f, 3.0f);
				ImGui::SliderFloat("Radius", &object.radius, 0.0f, 1.0f);
				ImGui::SliderFloat("Height", &object.height, 0.0f, 1.0f);
			}
		}
		ImGui::PopID();
	}
	ImGui::Separator();
	ImGui::Checkbox("Show ImGui demo", &show_imgui_demo);
	ImGui::End();
	if (show_imgui_demo) ImGui::ShowDemoWindow(&show_imgui_demo);
}

void render(const graphics::internal::FrameData& frame) {
	const auto& context = graphics::internal::context;
	// prepare() дождался fence прошлого кадра; UBO можно обновить без гонки с GPU.
	if (!updateUniforms()) std::cerr << "Failed to update uniform buffers\n";
	if (vkResetCommandBuffer(frame.command_buffer, 0) != VK_SUCCESS) return;
	const VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	if (vkBeginCommandBuffer(frame.command_buffer, &begin_info) != VK_SUCCESS) return;
	const VkClearValue clear_values[] = {
		{.color = {{0.07f, 0.10f, 0.16f, 1.0f}}},
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
	const VkViewport viewport{0.0f, 0.0f, float(context.swapchain_extent.width),
	                          float(context.swapchain_extent.height), 0.0f, 1.0f};
	const VkRect2D scissor{{0, 0}, context.swapchain_extent};
	vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	const VkBuffer buffers[] = {vertex_buffer.handle};
	const VkDeviceSize offsets[] = {0};
	vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, buffers, offsets);
	vkCmdBindIndexBuffer(frame.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
	for (const auto& object : objects) {
		vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        pipeline_layout, 0, 1, &object.descriptor, 0, nullptr);
		vkCmdDrawIndexed(frame.command_buffer, static_cast<uint32_t>(torus.indices.size()), 1, 0, 0, 0);
	}
	vkCmdEndRenderPass(frame.command_buffer);
	if (vkEndCommandBuffer(frame.command_buffer) != VK_SUCCESS)
		std::cerr << "Failed to record draw commands\n";
}

} // namespace scene
