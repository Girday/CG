#pragma once

#include "graphics_internal.hpp"
#include <filesystem>

namespace gpu {
struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize capacity = 0;
};
struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

bool createBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage, bool device_local = false);
bool writeBuffer(const Buffer& buffer, const void* source, VkDeviceSize size, VkDeviceSize offset = 0);
void destroyBuffer(Buffer& buffer);
// Копирование и барьер TRANSFER_WRITE -> VERTEX_ATTRIBUTE/INDEX_READ.
void copyGeometry(VkCommandBuffer command, const Buffer& staging, const Buffer& target, VkDeviceSize size);
bool uploadGeometry(Buffer& buffer, const void* source, VkDeviceSize size, VkBufferUsageFlags usage);
bool loadTexture(Texture& texture, const std::filesystem::path& path, bool srgb);
void destroyTexture(Texture& texture);
}
