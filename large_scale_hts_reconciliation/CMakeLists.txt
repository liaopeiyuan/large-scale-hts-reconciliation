# Referencing https://github.com/pybind/cmake_example

cmake_minimum_required(VERSION 3.4...3.18)
project(lhts)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_subdirectory(pybind11)
pybind11_add_module(lhts src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(lhts
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})