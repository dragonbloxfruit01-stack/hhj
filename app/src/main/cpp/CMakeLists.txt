cmake_minimum_required(VERSION 3.22.1)
project(localai800)

add_library(localai800 SHARED native-lib.cpp)

find_library(log-lib log)
target_link_libraries(localai800 ${log-lib})
