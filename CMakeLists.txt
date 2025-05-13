cmake_minimum_required(VERSION 3.22...3.31 FATAL_ERROR)
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
_add_nvpro_core_lib()

# ###############################################################################
# fetch OpenXR
#
include(ExternalProject)
set(OPENXR_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/bin_x64/openxr)
ExternalProject_Add(
    OpenXR-SDK
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenXR-SDK.git
    GIT_TAG release-1.1.38
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${OPENXR_INSTALL_DIR}
)
# The OpenXR loader library ends in a `d` only if it's built in
# debug mode and on Windows.
if(WIN32)
    set(OPENXR_DEBUG_POSTFIX "d")
else()
    set(OPENXR_DEBUG_POSTFIX "")
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(Vulkan REQUIRED)

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
        COMMAND ${Vulkan_SLANGC_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/${SHADER_FILE} -matrix-layout-row-major -profile sm_6_6 -target spirv -o ${COMPILED_SHADER_FILE}
        DEPENDS ${SHADER_FILE} ${SHADERS_DEPENDENCIES}
        COMMENT "Compiling ${SHADER_FILE}"
    )
    add_custom_command(
        OUTPUT ${COMPILED_SHADER_INL_FILE}
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/bin2inl.py ${COMPILED_SHADER_FILE} ${COMPILED_SHADER_INL_FILE}
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
    xr_multi_gpu PUBLIC
    include
    ${CMAKE_BINARY_DIR}/include
    ${OPENXR_INSTALL_DIR}/include
)

target_link_libraries(
    xr_multi_gpu PUBLIC
    Vulkan::Vulkan
    nvtx
    # This generator expression adds a 'd' to the end of the library
    # to search for if we're on Windows and building in debug mode.
    ${OPENXR_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openxr_loader$<$<CONFIG:Debug>:${OPENXR_DEBUG_POSTFIX}>${CMAKE_STATIC_LIBRARY_SUFFIX}
)

add_dependencies(
    xr_multi_gpu
    OpenXR-SDK
    xr_multi_gpu_shaders
)

# ####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_finalize_target(${PROJNAME})
