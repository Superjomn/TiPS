cmake_minimum_required(VERSION 3.0)
project(swiftsnails2)

# for IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

option(WITH_TESTING         "Compile with Unittests"          ON)
option(WITH_CUDA            "Compile with CUDA support"       OFF)
option(WITH_DEBUG           "Compile with debug information"  OFF)

if (WITH_TESTING)
  add_definitions(-DSWIFTS_WITH_TEST)
endif()
if (WITH_DEBUG)
  add_definitions(-DSWIFTS_WITH_DEBUG)
endif()

if (WITH_CUDA)
  message(STATUS "Enable CUDA")
  add_definitions(-DCINN_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(${CMAKE_SOURCE_DIR}/cinn/runtime/cuda)
  set(CUDA_SEPARABLE_COMPILATION ON)
endif()

include(cmake/core.cmake)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/thirds)

include(cmake/external/pybind11.cmake)
include(cmake/external/gflags.cmake)
include(cmake/external/glog.cmake)
include(cmake/external/gtest.cmake)
include(cmake/external/openmpi.cmake)
find_package(Threads REQUIRED)
## load in pkg-config support
find_package(PkgConfig)
## use pkg-config to get hints for 0mq locations
pkg_check_modules(PC_ZeroMQ QUIET zmq)

## use the hint from above to find where 'zmq.hpp' is located
find_path(ZeroMQ_INCLUDE_DIR
        NAMES zmq.hpp
        PATHS ${PC_ZeroMQ_INCLUDE_DIRS})

## use the hint from about to find the location of libzmq
find_library(ZeroMQ_LIBRARY
        NAMES zmq
        PATHS ${PC_ZeroMQ_LIBRARY_DIRS})

message(STATUS "Found ZMQ header: ${ZeroMQ_INCLUDE_DIR}")
message(STATUS "Found ZMQ library: ${ZeroMQ_LIBRARY}")
include_directories(${ZeroMQ_INCLUDE_DIR})


set(core_src CACHE INTERNAL "" FORCE)

add_subdirectory(swiftps)

cc_library(swiftscore SRCS ${core_src} DEPS glog)
target_link_libraries(swiftscore ${OpenMPI_LIBRARY} ${ZeroMQ_LIBRARY})
