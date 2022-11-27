# Referencing https://github.com/pybind/cmake_example

cmake_minimum_required(VERSION 3.4...3.18)
project(lhts)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_subdirectory(pybind11)
pybind11_add_module(lhts src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(lhts
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})