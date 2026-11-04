#pragma once
#include <cstddef>

namespace texture_math {
// Дескриптор адресует одну структуру, а динамический offset выбирает её слот.
constexpr std::size_t uniformStride(std::size_t size, std::size_t alignment) {
    return alignment ? ((size + alignment - 1) / alignment) * alignment : size;
}
}
