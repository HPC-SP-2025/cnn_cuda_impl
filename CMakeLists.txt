cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 86)
project(MyCudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
  # or 61, 75, 86 etc. depending on your GPU

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SRC_FILES 
  ${CMAKE_SOURCE_DIR}/src/layers/relu.cu
  ${CMAKE_SOURCE_DIR}/src/nn/sequential.cpp
  ${CMAKE_SOURCE_DIR}/src/layers/softmax.cu
  ${CMAKE_SOURCE_DIR}/src/layers/cross_entropy_loss.cu
  ${CMAKE_SOURCE_DIR}/src/layers/linear.cu
)
list(APPEND SOURCES ${SRC_FILES})

list(APPEND SOURCES ${CMAKE_SOURCE_DIR}/main.cu)
list(APPEND SOURCES ${CMAKE_SOURCE_DIR}/model.cpp)

foreach(SOURCE_FILE ${SOURCES})
    message(STATUS "Detected source file: ${SOURCE_FILE}")
endforeach()

add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
)
