cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

add_compile_definitions(SAMPLE_NAME="${CMAKE_PROJECT_NAME}")

# ####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#
if(NOT BASE_DIRECTORY)
    find_path(BASE_DIRECTORY
        NAMES nvpro_core/cmake/setup.cmake
        PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../..
        REQUIRED
        DOC "Directory containing nvpro_core"
    )
endif()

if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
    include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
    message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

_add_project_definitions(${PROJNAME})

# ###############################################################################
# fetch OpenXR and NVTX
#
include(FetchContent)
FetchContent_Declare(
    OpenXR-SDK
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenXR-SDK.git
    GIT_TAG release-1.1.38
)
FetchContent_MakeAvailable(OpenXR-SDK)

FetchContent_Declare(
    nvtx
    GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
    GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(nvtx)

list(APPEND CMAKE_MODULE_PATH "$ENV{SUPERLUMINAL_API_DIR}")
find_package(Vulkan REQUIRED)
find_package(SuperluminalAPI)

# ###############################################################################
# shaders
#
set(
    SHADERS_SRC
    shaders/layeredMesh.slang
)

set(
    SHADERS_DEPENDENCIES
    shaders/perlin.h
)

if(NOT Vulkan_SLANGC_EXECUTABLE)
    get_filename_component(Vulkan_LIBRARY_DIR ${Vulkan_LIBRARY} DIRECTORY)
    find_program(Vulkan_SLANGC_EXECUTABLE
        NAMES slangc
        HINTS ${Vulkan_LIBRARY_DIR}/../Bin
    )
endif()

foreach(SHADER_FILE ${SHADERS_SRC})
    set(COMPILED_SHADER_FILE ${CMAKE_BINARY_DIR}/${SHADER_FILE}.spv)
    set(COMPILED_SHADER_INL_FILE ${CMAKE_BINARY_DIR}/include/${SHADER_FILE}.inl)
    add_custom_command(
        OUTPUT ${COMPILED_SHADER_FILE}
        COMMAND ${Vulkan_SLANGC_EXECUTABLE} ${CMAKE_SOURCE_DIR}/${SHADER_FILE} -matrix-layout-row-major -profile sm_6_6 -target spirv -o ${COMPILED_SHADER_FILE}
        DEPENDS ${SHADER_FILE} ${SHADERS_DEPENDENCIES}
        COMMENT "Compiling ${SHADER_FILE}"
    )
    add_custom_command(
        OUTPUT ${COMPILED_SHADER_INL_FILE}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bin2inl.py ${COMPILED_SHADER_FILE} ${COMPILED_SHADER_INL_FILE}
        DEPENDS ${COMPILED_SHADER_FILE} bin2inl.py
        COMMENT "Converting ${SHADER_FILE}"
    )
    list(APPEND SHADERS_DST ${COMPILED_SHADER_INL_FILE})
endforeach()

add_custom_target(
    xr_multi_gpu_shaders
    DEPENDS ${SHADERS_DST}
)

# ###############################################################################
# main executable
#
add_executable(
    xr_multi_gpu
    src/App.cpp
    src/Instance.cpp
    src/main.cpp
    src/Matrix.cpp
    src/Options.cpp
    src/Renderer.cpp
    src/RenderTarget.cpp
    src/Scene.cpp
    src/SimpleTimingProfiler.cpp
    src/StereoProjection.cpp
    src/TriangleMesh.cpp
    src/VulkanAppProfiler.cpp
    src/VulkanImageResource.cpp
    src/VulkanQueueFamily.cpp
    src/Window.cpp
    src/WindowUserInterface.cpp
    src/XrUserInterface.cpp
)

target_include_directories(
    xr_multi_gpu
    PUBLIC include
    PUBLIC ${CMAKE_BINARY_DIR}/include
    PUBLIC ${nvtx_SOURCE_DIR}/c/include
)

target_link_libraries(
    xr_multi_gpu
    PUBLIC Vulkan::Vulkan
    PUBLIC openxr_loader
)

add_dependencies(
    xr_multi_gpu
    xr_multi_gpu_shaders
)

if(SuperluminalAPI_FOUND)
    target_link_libraries(xr_multi_gpu PUBLIC SuperluminalAPI)
endif()

# ####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_finalize_target(${PROJNAME})
