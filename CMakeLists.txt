cmake_minimum_required(VERSION 3.8)
project(vkexamples)

include(FindPkgConfig)

pkg_check_modules(VULKAN vulkan)
pkg_check_modules(GLFW3 glfw3)

include_directories(
    ${VULKAN_INCLUDE_DIRS}
    ${GLFW3_INCLUDE_DIRS}
)

function(add_program BIN_NAME SRC_NAME)
    set(PROGRAM_SUBDIR "${CMAKE_BINARY_DIR}/bin/vk${BIN_NAME}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROGRAM_SUBDIR})
    add_executable(${BIN_NAME} ${SRC_NAME})
    target_link_libraries(${BIN_NAME} ${VULKAN_LIBRARIES} ${GLFW3_LIBRARIES} m)
    if(ARGV2)
        target_compile_definitions(${BIN_NAME} PRIVATE ${ARGV2})
    endif()
endfunction(add_program)

function(copy_shaders BIN_NAME)
    set(PROGRAM_SUBDIR "${CMAKE_BINARY_DIR}/bin/vk${BIN_NAME}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROGRAM_SUBDIR})
    file(COPY ${ARGN} DESTINATION ${PROGRAM_SUBDIR})
endfunction(copy_shaders)

add_subdirectory(vktriangle)
add_subdirectory(vktriangle_descriptor)
add_subdirectory(vktriangle_glfw)
add_subdirectory(vktriangle_subpass)
add_subdirectory(vktriangle_vertex)
