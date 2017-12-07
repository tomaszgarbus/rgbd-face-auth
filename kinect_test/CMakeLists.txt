cmake_minimum_required(VERSION 3.7)
project(kinect_test)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
add_executable(kinect_test ${SOURCE_FILES})
target_link_libraries(kinect_test freenect)
