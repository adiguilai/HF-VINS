cmake_minimum_required(VERSION 3.15)
project(test)

add_library(Extractor Extractor.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11)
find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(Extractor ${CUDA_LIBRARIES})
endif()

# set your own libtorch path
set(TORCH_PATH /home/jesse/libtorch/share/cmake/Torch)
if(TORCH_PATH)
    message("TORCH_PATH set to: ${TORCH_PATH}")
    set(Torch_DIR ${TORCH_PATH})
else()
    message(FATAL_ERROR "Need to specify Torch path, e.g., pytorch/torch/share/cmake/Torch ")
endif()

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_DIRS})

target_link_libraries(Extractor ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(test test.cpp)
target_link_libraries(test Extractor)