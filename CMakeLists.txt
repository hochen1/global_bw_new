cmake_minimum_required(VERSION 3.5)
project(clpeak)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)
include(GNUInstallDirs)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenCL)
FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS})
if(NOT OpenCL_FOUND OR NOT HPP_FOUND)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(BuildSdk)

  find_package(OpenCL REQUIRED)

  set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} "${SDK_ROOT}/include")
  FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS} REQUIRED)
endif()

if(NOT OpenCL_FOUND)
  message(FATAL_ERROR "Could not find OpenCL include/libs. Set OCL_ROOT to your OpenCL SDK")
else()
  message(STATUS "Selected OpenCL includes from ${OpenCL_INCLUDE_DIRS}")
  message(STATUS "Selected OpenCL lib ${OpenCL_LIBRARIES}")
endif()

# extra libs for static library
if(UNIX AND ${OpenCL_LIBRARIES} MATCHES "libOpenCL.a")
  set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} pthread dl)
elseif(WIN32 AND ${OpenCL_LIBRARIES} MATCHES "OpenCL.lib")
  set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} cfgmgr32.lib runtimeobject.lib)
endif()

set(CMAKE_CXX_STANDARD "11")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CLPEAK_SOURCE_FILES
    src/common.cpp
    src/entry.cpp
)

add_executable(clpeak ${CLPEAK_SOURCE_FILES})
target_link_libraries(clpeak ${OpenCL_LIBRARIES})
target_include_directories(clpeak PRIVATE ${OpenCL_INCLUDE_DIRS} "include" "src/kernel")
target_compile_definitions(clpeak PRIVATE CL_TARGET_OPENCL_VERSION=120)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(clpeak PRIVATE -fPIC -Wall -Wextra -Wno-unused-parameter)
endif()