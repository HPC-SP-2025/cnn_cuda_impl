cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
  # or 61, 75, 86 etc. depending on your GPU

# Enable device flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3")

# Set compiler optimization and warnings for host code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Try to find OpenMP
find_package(OpenMP REQUIRED)

# Add OpenMP flags if found
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP!")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SRC_FILES 
  ${CMAKE_SOURCE_DIR}/src/layers/relu.cu
  ${CMAKE_SOURCE_DIR}/src/nn/sequential.cpp
  ${CMAKE_SOURCE_DIR}/src/layers/softmax.cu
  ${CMAKE_SOURCE_DIR}/src/layers/cross_entropy_loss.cu
  ${CMAKE_SOURCE_DIR}/src/layers/linear.cu
  ${CMAKE_SOURCE_DIR}/src/dataloader/dataloader.cpp
)
list(APPEND SOURCES ${SRC_FILES})

list(APPEND SOURCES ${CMAKE_SOURCE_DIR}/main.cpp)
list(APPEND SOURCES ${CMAKE_SOURCE_DIR}/model.cpp)

foreach(SOURCE_FILE ${SOURCES})
    message(STATUS "Detected source file: ${SOURCE_FILE}")
endforeach()

add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)