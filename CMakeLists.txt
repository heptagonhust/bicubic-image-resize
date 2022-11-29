cmake_minimum_required(VERSION 2.8)
project(resize)
option(USE_OPENMP      "Set to ON to build use openmp"  ON) #打开OpenMP并行处理开关
option(USE_SSE         "Set to ON to build use SSE"  ON)    #打开SSE加速开关
add_compile_options(-O3)
add_compile_options(-funsafe-math-optimizations)
add_executable(resize main.cpp)
if (USE_OPENMP)
    find_package(OpenMP QUIET)   
    if (OPENMP_FOUND)
        message(STATUS "Use OpenMP")
        add_definitions(-DUSE_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()
# Use SSE                                                  #使用SSE
if (USE_SSE)
    add_definitions(-DUSE_SSE)
    message(STATUS "Use SSE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(Threads REQUIRED)
target_link_libraries(resize Threads::Threads)
find_package(OpenMP REQUIRED)
target_link_libraries(resize OpenMP::OpenMP_CXX)
add_library(sse4_1 INTERFACE IMPORTED)
target_compile_options(resize PRIVATE -msse4.2)
set_target_properties(${TARGET} PROPERTIES
COMPILE_FLAGS "-pthread"
LINK_FLAGS "-pthread")
# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})
# Link your application with OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})
