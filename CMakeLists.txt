cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB SOURCES "*.cu" "*.cpp" "*.c" "cuh" "*.h" "*.hpp")

if(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std c++11")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G;-g;-lineinfo;--ptxas-options=-v")
  else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;-g;-lineinfo;--ptxas-options=-v")
  endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

  if(OpenCV_FOUND)
    CUDA_INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    CUDA_ADD_EXECUTABLE(imageFilteringGpu ${SOURCES})
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(imageFilteringGpu ${OpenCV_LIBS})
  endif(OpenCV_FOUND)
endif(CUDA_FOUND)
