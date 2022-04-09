cmake_minimum_required(VERSION 3.15)
project(test)

add_library(hloc hloc.cpp)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA)
if (CUDA_FOUND)
    target_link_libraries(hloc ${CUDA_LIBRARIES})
endif ()

# set your own libtorch path
set(TORCH_PATH /home/jesse/libtorch/share/cmake/Torch)
if (TORCH_PATH)
    message("TORCH_PATH set to: ${TORCH_PATH}")
    set(Torch_DIR ${TORCH_PATH})
else ()
    message(FATAL_ERROR "Need to specify Torch path, e.g., pytorch/torch/share/cmake/Torch ")
endif ()

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_DIRS})

target_link_libraries(hloc ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(test test.cpp)
target_link_libraries(test hloc)