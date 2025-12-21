cmake_minimum_required(VERSION 2.8)
project(example CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "/home/wangda/anaconda3/envs/MBR-PCR/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")

find_package(pybind11 REQUIRED)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# add_subdirectory(pybind11)
pybind11_add_module(distributionSample distributionSample.cpp)

target_link_libraries(distributionSample PRIVATE pybind11::module ${PCL_LIBRARIES})
