cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0146 OLD)
project(tensor_renderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_VIRTUALENV FIRST)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(CUDA REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
    ${Python3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

add_executable(tensor_renderer main.cpp)

target_link_libraries(tensor_renderer
    ${Python3_LIBRARIES}
    ${CUDA_LIBRARIES}
    glfw
    OpenGL
    GLEW
)

target_compile_definitions(tensor_renderer PRIVATE
    PYTHON_AVAILABLE
)

set_target_properties(tensor_renderer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
