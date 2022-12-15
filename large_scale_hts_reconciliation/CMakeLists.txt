# Referencing https://github.com/pybind/cmake_example

cmake_minimum_required(VERSION 3.4...3.18)
project(lhts)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(OpenMP)
find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(${MPI_INCLUDE_PATH})

add_subdirectory(pybind11)
pybind11_add_module(lhts src/main.cpp src/S.cpp src/G.cpp src/distribute_forecast.cpp src/Distributed.cpp src/reconcile.cpp src/metrics.cpp)

target_include_directories(lhts PRIVATE include)
target_link_libraries(lhts PRIVATE mpi)

if(OpenMP_CXX_FOUND)
  target_link_libraries(lhts PUBLIC OpenMP::OpenMP_CXX)
else()
  message( FATAL_ERROR "Your compiler does not support OpenMP" )
endif()

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(lhts
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
# add_compile_definitions(EIGEN_DONT_PARALLELIZE)