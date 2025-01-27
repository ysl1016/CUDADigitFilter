cmake_minimum_required(VERSION 3.10)
project(mnist_filter CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

add_executable(mnist_filter
    src/main.cu
    src/mnist_loader.cu
    src/filters.cu
)

set_target_properties(mnist_filter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(mnist_filter ${OpenCV_LIBS})