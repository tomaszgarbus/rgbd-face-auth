cmake_minimum_required(VERSION 3.5.1)
project(rgbd_face_auth)

find_package(OpenCV REQUIRED)
find_package(freenect2 REQUIRED)

set(SOURCE_FILES src/libkinect/libkinect.cpp src/libkinect/libkinect.hpp)
set(CMAKE_CXX_STANDARD 17)

add_executable(take_photo src/take_photo.cpp ${SOURCE_FILES})
target_link_libraries(take_photo freenect)
target_link_libraries(take_photo ${OpenCV_LIBS})
