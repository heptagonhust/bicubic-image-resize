cmake_minimum_required(VERSION 2.8)
project(resize)

string(REGEX REPLACE "(-O[0123s])" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "(-O[0123s])" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
string(REGEX REPLACE "(-O[0123s])" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REGEX REPLACE "(-O[0123s])" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

add_compile_options(-O3 -std=c++17 -march=native -fopenmp)
add_link_options(-fopenmp)

add_executable(resize main.cpp)
