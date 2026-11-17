cmake_minimum_required(VERSION 3.20)
project(cg-lab4 LANGUAGES C CXX)

# The Vulkan shell and material assets are shared with the previous lab.
set(lab3_dir "${CMAKE_CURRENT_SOURCE_DIR}/../lab3")
if(NOT TARGET cg-lab3)
    add_subdirectory("${lab3_dir}" "${CMAKE_CURRENT_BINARY_DIR}/lab3" EXCLUDE_FROM_ALL)
endif()
include(FetchContent)
FetchContent_GetProperties(imgui SOURCE_DIR imgui_dir)
find_package(Vulkan REQUIRED)
get_filename_component(vulkan_bin "${Vulkan_GLSLC_EXECUTABLE}" DIRECTORY)
find_program(GLSLC_EXECUTABLE glslc HINTS "${vulkan_bin}" REQUIRED)

set(shader_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
add_executable(cg-lab4
    source/scene.cpp source/shadow_maps.cpp
    "${lab3_dir}/source/main.cpp" "${lab3_dir}/source/application.cpp"
    "${lab3_dir}/source/graphics_internal.cpp" "${lab3_dir}/source/torus.cpp"
    "${lab3_dir}/source/gpu_resources.cpp"
    "${imgui_dir}/backends/imgui_impl_glfw.cpp" "${imgui_dir}/backends/imgui_impl_vulkan.cpp"
    "${imgui_dir}/imgui.cpp" "${imgui_dir}/imgui_draw.cpp"
    "${imgui_dir}/imgui_tables.cpp" "${imgui_dir}/imgui_widgets.cpp"
)
set_target_properties(cg-lab4 PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED TRUE)
target_include_directories(cg-lab4 PRIVATE "${lab3_dir}/source" "${imgui_dir}" "${imgui_dir}/backends")
target_compile_definitions(cg-lab4 PRIVATE CG_DYNAMIC_RENDERING
    CG_SHADER_DIRECTORY="${shader_dir}" CG_TEXTURE_DIRECTORY="${lab3_dir}/textures")
target_link_libraries(cg-lab4 PRIVATE glfw Vulkan::Vulkan vk-bootstrap::vk-bootstrap GPUOpen::VulkanMemoryAllocator)
if(MSVC)
    target_compile_options(cg-lab4 PRIVATE /W4 /wd4201)
    target_link_options(cg-lab4 PRIVATE /subsystem:windows /entry:mainCRTStartup)
else()
    target_compile_options(cg-lab4 PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
    if(WIN32)
        target_link_options(cg-lab4 PRIVATE -mwindows)
    endif()
endif()
if(WIN32)
    target_compile_definitions(cg-lab4 PRIVATE _USE_MATH_DEFINES)
endif()

set(shader_outputs)
foreach(shader torus.vert torus.frag marker.vert marker.frag shadow.vert)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}")
    if(shader MATCHES "^marker\\.")
        set(source "${lab3_dir}/shaders/${shader}")
    endif()
    set(output "${shader_dir}/${shader}.spv")
    add_custom_command(OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${shader_dir}"
        COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders" -I "${lab3_dir}/shaders" "${source}" -o "${output}"
        DEPENDS "${source}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/scene.glsl"
            "${lab3_dir}/shaders/scene.glsl" "${lab3_dir}/shaders/model.glsl"
        VERBATIM)
    list(APPEND shader_outputs "${output}")
endforeach()
add_custom_target(lab4_shaders DEPENDS ${shader_outputs})
add_dependencies(cg-lab4 lab4_shaders)
