#pragma once

#include "graphics_internal.hpp"

namespace shadows {

struct DepthMap {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    bool initialized = false;
};

inline constexpr VkExtent2D extent{2048, 2048};
inline constexpr VkImageLayout sampled_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

VkFormat selectFormat();
bool create(DepthMap& map, VkFormat format);
void destroy(DepthMap& map);
bool createSampler(VkSampler& sampler);
void begin(VkCommandBuffer command, DepthMap& map);
void end(VkCommandBuffer command, DepthMap& map);

} // namespace shadows
