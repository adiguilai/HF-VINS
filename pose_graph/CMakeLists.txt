cmake_minimum_required(VERSION 2.8.3)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

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

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        nav_msgs
        camera_model
        cv_bridge
        roslib
        )

find_package(OpenCV REQUIRED)


find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR})

catkin_package()

add_executable(pose_graph
        src/pose_graph_node.cpp
        src/pose_graph.cpp
        src/keyframe.cpp
        src/utility/CameraPoseVisualization.cpp
        src/hloc/hloc.cpp
        )

target_link_libraries(pose_graph ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
# message("catkin_lib  ${catkin_LIBRARIES}")
