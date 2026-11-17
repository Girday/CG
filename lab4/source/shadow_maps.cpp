#include "shadow_maps.hpp"

namespace shadows {
namespace {

using graphics::internal::context;
constexpr VkImageLayout attachment_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
constexpr VkPipelineStageFlags depth_stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
constexpr VkImageSubresourceRange image_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

} // namespace

VkFormat selectFormat() {
    constexpr VkFormat candidates[]{VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM};
    constexpr VkFormatFeatureFlags2 required = VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT |
                                               VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
                                               VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
    for (const auto format : candidates) {
        VkFormatProperties3 features{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &features};
        vkGetPhysicalDeviceFormatProperties2(context.physical_device, format, &properties);
        if ((features.optimalTilingFeatures & required) == required) return format;
    }
    return VK_FORMAT_UNDEFINED;
}

bool create(DepthMap& map, VkFormat format) {
    const VkImageCreateInfo image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo memory{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(context.allocator, &image_info, &memory, &map.image,
                       &map.allocation, nullptr) != VK_SUCCESS) return false;
    const VkImageViewCreateInfo view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = map.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = image_range,
    };
    return vkCreateImageView(context.device, &view_info, nullptr, &map.view) == VK_SUCCESS;
}

void destroy(DepthMap& map) {
    if (map.view) vkDestroyImageView(context.device, map.view, nullptr);
    if (map.image) vmaDestroyImage(context.allocator, map.image, map.allocation);
    map = {};
}

bool createSampler(VkSampler& sampler) {
    // PCF использует отдельные ближайшие тексели, поэтому линейная фильтрация
    // формата глубины не требуется. За пределами карты глубина равна 1.
    const VkSamplerCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .compareEnable = VK_TRUE,
        .compareOp = VK_COMPARE_OP_LESS,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    };
    return vkCreateSampler(context.device, &info, nullptr, &sampler) == VK_SUCCESS;
}

void begin(VkCommandBuffer command, DepthMap& map) {
    // Первый кадр начинает с UNDEFINED. Со второго ждём чтения карты
    // фрагментным шейдером, прежде чем снова записывать глубину.
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .oldLayout = map.initialized ? sampled_layout : VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = attachment_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = map.image,
        .subresourceRange = image_range,
    };
    const VkPipelineStageFlags source_stage = map.initialized
        ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    vkCmdPipelineBarrier(command, source_stage, depth_stages, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    const VkRenderingAttachmentInfo depth{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = map.view,
        .imageLayout = attachment_layout,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.depthStencil = {1.0f, 0}},
    };
    const VkRenderingInfo rendering{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .pDepthAttachment = &depth,
    };
    vkCmdBeginRendering(command, &rendering);
}

void end(VkCommandBuffer command, DepthMap& map) {
    vkCmdEndRendering(command);
    // Запись depth attachment должна завершиться до выборки в основном проходе.
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = attachment_layout,
        .newLayout = sampled_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = map.image,
        .subresourceRange = image_range,
    };
    vkCmdPipelineBarrier(command, depth_stages, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
    map.initialized = true;
}

} // namespace shadows
