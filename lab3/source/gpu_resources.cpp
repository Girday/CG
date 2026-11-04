#include "gpu_resources.hpp"

#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "../third_party/stb_image.h"

namespace gpu {
namespace {
// Временные ресурсы живут до fence: завершение записи CPU не означает завершение GPU.
bool upload(const std::function<void(VkCommandBuffer)>& record) {
    const auto& c = graphics::internal::context;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = c.graphics_queue_index;
    if (vkCreateCommandPool(c.device, &pool_info, nullptr, &pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    bool ok = vkAllocateCommandBuffers(c.device, &allocate, &command) == VK_SUCCESS &&
              vkCreateFence(c.device, &fence_info, nullptr, &fence) == VK_SUCCESS;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (ok) ok = vkBeginCommandBuffer(command, &begin) == VK_SUCCESS;
    if (ok) {
        record(command);
        ok = vkEndCommandBuffer(command) == VK_SUCCESS;
    }
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (ok) {
        ok = vkQueueSubmit(c.graphics_queue, 1, &submit, fence) == VK_SUCCESS;
        if (ok) ok = vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    }
    if (fence) vkDestroyFence(c.device, fence, nullptr);
    vkDestroyCommandPool(c.device, pool, nullptr);
    return ok;
}
}

bool createBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage, bool device_local) {
    const auto& c = graphics::internal::context;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocation{};
    allocation.usage = device_local ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO;
    if (device_local) allocation.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    else allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo result{};
    if (vmaCreateBuffer(c.allocator, &info, &allocation, &buffer.handle, &buffer.allocation, &result) != VK_SUCCESS)
        return false;
    buffer.mapped = result.pMappedData;
    buffer.capacity = size;
    return device_local || buffer.mapped != nullptr;
}

bool writeBuffer(const Buffer& buffer, const void* source, VkDeviceSize size, VkDeviceSize offset) {
    if (!buffer.mapped || offset > buffer.capacity || size > buffer.capacity - offset) return false;
    if (!size) return true;
    std::memcpy(static_cast<char*>(buffer.mapped) + offset, source, static_cast<size_t>(size));
    return vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, offset, size) == VK_SUCCESS;
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.handle) vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle, buffer.allocation);
    buffer = {};
}

void copyGeometry(VkCommandBuffer command, const Buffer& staging, const Buffer& target, VkDeviceSize size) {
    const VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(command, staging.handle, target.handle, 1, &region);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = target.handle;
    barrier.size = size;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
}

bool uploadGeometry(Buffer& buffer, const void* source, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer staging;
    bool ok = createBuffer(staging, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) && writeBuffer(staging, source, size) &&
              createBuffer(buffer, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    if (ok) ok = upload([&](VkCommandBuffer command) { copyGeometry(command, staging, buffer, size); });
    destroyBuffer(staging);
    return ok;
}

void destroyTexture(Texture& texture) {
    const auto& c = graphics::internal::context;
    if (texture.view) vkDestroyImageView(c.device, texture.view, nullptr);
    if (texture.image) vmaDestroyImage(c.allocator, texture.image, texture.allocation);
    texture = {};
}

bool loadTexture(Texture& texture, const std::filesystem::path& path, bool srgb) {
    int width = 0, height = 0, channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha), stbi_image_free);
    if (!pixels) {
        std::cerr << "Cannot load texture " << path << ": " << stbi_failure_reason() << '\n';
        return false;
    }
    const auto& c = graphics::internal::context;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(c.physical_device, &properties);
    if (width <= 0 || height <= 0 || uint32_t(width) > properties.limits.maxImageDimension2D ||
        uint32_t(height) > properties.limits.maxImageDimension2D) return false;
    const VkDeviceSize bytes = VkDeviceSize(width) * VkDeviceSize(height) * 4;
    Buffer staging;
    bool ok = createBuffer(staging, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) && writeBuffer(staging, pixels.get(), bytes);
    const VkFormat format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = format;
    image.extent = {uint32_t(width), uint32_t(height), 1};
    image.mipLevels = image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocation{};
    allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocation.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (ok) ok = vmaCreateImage(c.allocator, &image, &allocation, &texture.image, &texture.allocation, nullptr) == VK_SUCCESS;
    if (ok) ok = upload([&](VkCommandBuffer command) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = texture.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = image.extent;
        vkCmdCopyBufferToImage(command, staging.handle, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    });
    destroyBuffer(staging);
    if (ok) {
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = texture.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ok = vkCreateImageView(c.device, &view, nullptr, &texture.view) == VK_SUCCESS;
    }
    if (!ok) destroyTexture(texture);
    return ok;
}
}
