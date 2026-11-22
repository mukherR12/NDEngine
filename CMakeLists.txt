
cmake_minimum_required(VERSION 3.20)

project(VulkanTest LANGUAGES CXX)

# ------------------------------------------------------------------------------
# C++ configuration
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# ------------------------------------------------------------------------------
# Slang compiler
# ------------------------------------------------------------------------------

find_program(SLANGC_EXECUTABLE slangc REQUIRED)

# ------------------------------------------------------------------------------
# Shader paths
# ------------------------------------------------------------------------------

set(SHADER_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang
)

set(SHADER_OUTPUT_DIR
        ${CMAKE_BINARY_DIR}/shaders
)

set(SHADER_OUTPUT
        ${SHADER_OUTPUT_DIR}/slang.spv
)

# ------------------------------------------------------------------------------
# Copy textures into build directory
# ------------------------------------------------------------------------------

file(COPY ${CMAKE_SOURCE_DIR}/textures
        DESTINATION ${CMAKE_BINARY_DIR})

# ------------------------------------------------------------------------------
# Compile Slang -> SPIR-V
# ------------------------------------------------------------------------------

add_custom_command(
        OUTPUT ${SHADER_OUTPUT}

        COMMAND ${CMAKE_COMMAND} -E make_directory
        ${SHADER_OUTPUT_DIR}

        COMMAND ${SLANGC_EXECUTABLE}
        ${SHADER_SOURCE}
        -target spirv
        -profile spirv_1_4
        -emit-spirv-directly
        -fvk-use-entrypoint-name
        -entry vertMain
        -entry fragMain
        -o ${SHADER_OUTPUT}

        DEPENDS ${SHADER_SOURCE}

        COMMENT "Compiling Slang shaders"
        VERBATIM
)

add_custom_target(shaders
        DEPENDS ${SHADER_OUTPUT}
)

# ------------------------------------------------------------------------------
# Main executable
# ------------------------------------------------------------------------------

add_executable(VulkanTest
        src/main.cpp
)

target_link_libraries(VulkanTest PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
)

# Ensure shaders build before executable
add_dependencies(VulkanTest shaders)

# ------------------------------------------------------------------------------
# Shader output directory
# ------------------------------------------------------------------------------

target_compile_definitions(VulkanTest PRIVATE
        SHADER_OUTPUT_DIR="${SHADER_OUTPUT_DIR}"
)

# ------------------------------------------------------------------------------
# GLFW runtime DLL
# ------------------------------------------------------------------------------

#if(WIN32)
#    add_custom_command(TARGET VulkanTest POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different
#            "C:/msys64/clangarm64/bin/glfw3.dll"
#            "$<TARGET_FILE_DIR:VulkanTest>"
#            COMMENT "Copying GLFW runtime DLL"
#    )
#endif()
