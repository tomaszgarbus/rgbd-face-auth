cmake_minimum_required(VERSION 3.9)
project(kinect2_test)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${freenect2_DIR}/lib/cmake/freenect2)
find_package(freenect2 REQUIRED)
include_directories($ENV{HOME}/freenect2/include)

message(status "home dir: " $ENV{HOME})
message(status "libfreenect2 source dir: " ${freenect2_DIR})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${freenect2_LIBRARIES})
