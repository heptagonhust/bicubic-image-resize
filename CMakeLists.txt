cmake_minimum_required(VERSION 2.8)
project(resize)
add_compile_options(-O3)
add_executable(resize main.cpp)


find_package(Threads REQUIRED)
target_link_libraries(resize PUBLIC Threads::Threads )