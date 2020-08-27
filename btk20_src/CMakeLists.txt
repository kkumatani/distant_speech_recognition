cmake_minimum_required(VERSION 2.8.12)
project(btk20)

# Command example:
# mkdir -p build; cd build; cmake -DCMAKE_INSTALL_PREFIX=$HOME/local .. && make install

# Setting installation directory
set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE STRING
                         "Installation directory for headers")
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING
                         "Installation directory for library")
set(TARGET_PYTHON_VERSION "2.7" CACHE STRING
                          "python version to be used")
set(PYTHON_SITE_PACKAGES "lib/python${TARGET_PYTHON_VERSION}/dist-packages/${PROJECT_NAME}")
message(STATUS "Setting installation directory...")
message(STATUS "   Include: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "   Lib:     ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
message(STATUS "   Python:  ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES}")

set(CMAKE_C_FLAGS "-Wall -fPIC")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS "-Wall -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
# print verbose information for memory debugging
# add_definitions(-DDEBUG_MEMORY_MANAGER)
add_definitions(-DENABLE_LEGACY_BTK_API)

# For swig
find_package(SWIG 3.0 REQUIRED)
include(${SWIG_USE_FILE})
set(CMAKE_SWIG_FLAGS "-I${CMAKE_SOURCE_DIR}" "-I${CMAKE_SOURCE_DIR}/include")

# For python binding
find_package(PythonLibs ${TARGET_PYTHON_VERSION} REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
message(STATUS "PYTHON_INCLUDE_PATH: ${PYTHON_INCLUDE_PATH}")
message(STATUS "PYTHON_LIBRARIES:    ${PYTHON_LIBRARIES}")

find_package(PythonInterp ${TARGET_PYTHON_VERSION} REQUIRED)

# For numpy
add_definitions(-DNPY_NO_DEPRECATED_API)
if(NOT PYTHON_NUMPY_INCLUDE_DIR)
       message(STATUS "Searchinig numpy...")
       include("FindNumPy.cmake")
       if (NOT NUMPY_FOUND)
          message(FATAL_ERROR "Could not find NUMPY!")
       endif ()
endif(NOT PYTHON_NUMPY_INCLUDE_DIR)

# For GSL
find_package(GSL REQUIRED)

# For libsndfile
include("FindSndFile.cmake")

# For CUDA
find_package(CUDA 9.0)
if(NOT CUDA_VERSION)
       message(STATUS "Skipping CUDA support")
else()
       add_definitions(-DUSE_GPU)
       message (STATUS "CUDA_ROOT_DIR     = ${CUDA_TOOLKIT_ROOT_DIR}")
       message (STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
       message (STATUS "CUDA_LIBRARIES    = ${CUDA_LIBRARIES}")
endif(NOT CUDA_VERSION)

# For pytorch
#find_package(Torch)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR})

#### BTK20 library

# Basic functions
include_directories(common)
add_subdirectory(common)
include_directories(utils)
add_subdirectory(utils)

# Vector, matrix and basic linear algebra
include_directories(stream)
add_subdirectory(stream)
include_directories(matrix)
add_subdirectory(matrix)

# Speech features
include_directories(feature)
add_subdirectory(feature)

# Basic LTI processing
include_directories(convolution)
add_subdirectory(convolution)

# Subband processing
include_directories(modulated)
add_subdirectory(modulated)

# Adaptive processing
include_directories(aec)
add_subdirectory(aec)
include_directories(dereverberation)
add_subdirectory(dereverberation)
include_directories(square_root)
add_subdirectory(square_root)
include_directories(lms)
add_subdirectory(lms)
include_directories(beamformer)
add_subdirectory(beamformer)
include_directories(postfilter)
add_subdirectory(postfilter)

# time delay estimation
include_directories(tde)
add_subdirectory(tde)

# source localization
include_directories(localization)
add_subdirectory(localization)

include_directories(sad)
add_subdirectory(sad)

include_directories(objective_measure)
add_subdirectory(objective_measure)

# python libraries
include_directories(lib)
add_subdirectory(lib)
