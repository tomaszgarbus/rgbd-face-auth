cmake_minimum_required(VERSION 3.9)
project(opencv_test)
find_package(OpenCV REQUIRED)
find_package(freenect2 REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${freenect2_DIR}/lib/cmake/freenect2)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($ENV{HOME}/freenect2/include)

add_executable(from_file from_file.cpp)
target_link_libraries(from_file ${OpenCV_LIBS})

add_executable(from_kinect from_kinect.cpp)
target_link_libraries(from_kinect ${OpenCV_LIBS} ${freenect2_LIBRARIES})

file(COPY haarcascades/haarcascade_smile.xml haarcascades/haarcascade_frontalface_alt_tree.xml haarcascades/haarcascade_frontalface_alt.xml haarcascades/haarcascade_eye_tree_eyeglasses.xml DESTINATION haarcascades)