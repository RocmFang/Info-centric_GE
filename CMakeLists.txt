cmake_minimum_required(VERSION 3.5)


project(FeLoG)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_EXPORT_COMPILE_COMMANDS on) # generate compile_commands.json

set(DGER_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(WITH_TESTS "Build unit test programs" OFF)
option(WITH_NCCL "Enable NCCL embedding synchronization" OFF)
option(WITH_MPI_CUDA "Enable CUDA-aware MPI embedding synchronization" OFF)

include(felog.cmake)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI REQUIRED)
if(MPI_FOUND)
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
    #set(MPI_EXECUTABLE_SUFFIX .openmpi)
    include_directories(${MPI_INCLUDE_PATH})
endif()

# if(WITH_TESTS)
#     enable_testing()
#     include_directories("${CMAKE_CURRENT_SOURCE_DIR}/ext/gtest/googletest/include")
#     add_subdirectory(ext/gtest)
#     link_directories(${CMAKE_BINARY_DIR}/lib)
#     set(GTEST_LIBRARIES "gtest" "gtest_main")
# endif()

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${MPI_CXX_COMPILE_FLAGS} -ccbin=g++ ")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")

if(WITH_NCCL)
    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS
            $ENV{NCCL_HOME}/include
            ${CUDA_TOOLKIT_ROOT_DIR}/include
            /usr/local/cuda/include
            /usr/include
    )
    find_library(NCCL_LIBRARY nccl
        HINTS
            $ENV{NCCL_HOME}/lib
            $ENV{NCCL_HOME}/lib64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            /usr/local/cuda/lib64
            /usr/lib/x86_64-linux-gnu
    )
    if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
        message(FATAL_ERROR "WITH_NCCL=ON requires nccl.h and libnccl. Set NCCL_HOME or install NCCL.")
    endif()
    include_directories(${NCCL_INCLUDE_DIR})
    add_definitions(-DWITH_NCCL)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DWITH_NCCL")
    message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
    message(STATUS "NCCL library: ${NCCL_LIBRARY}")
endif()

if(WITH_MPI_CUDA)
    add_definitions(-DWITH_MPI_CUDA)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DWITH_MPI_CUDA")
    message(STATUS "CUDA-aware MPI embedding synchronization enabled")
endif()

#use c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -O3 -D USE_MKL -mavx512f")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x ")
message("${CMAKE_CXX_FLAGS}")



set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DGER_RUNTIME_OUTPUT_DIRECTORY})

include_directories(include)
include_directories(ext)

add_subdirectory(src)
