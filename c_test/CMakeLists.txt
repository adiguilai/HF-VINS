cmake_minimum_required(VERSION 3.15)
project(test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_COMPILER  /usr/local/cuda-11.6/bin/nvcc)
find_package(CUDA)

# set your own libtorch path
set(TORCH_PATH /home/jesse/libtorch/share/cmake/Torch)
if (TORCH_PATH)
    message("TORCH_PATH set to: ${TORCH_PATH}")
    set(Torch_DIR ${TORCH_PATH})
else ()
    message(FATAL_ERROR "Need to specify Torch path, e.g., pytorch/torch/share/cmake/Torch ")
endif ()

find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(test
        test.cpp
        hloc/hloc.cpp
        hloc/hlocDatabase.h)
target_link_libraries(test ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS})