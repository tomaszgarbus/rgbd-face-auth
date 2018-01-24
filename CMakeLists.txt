cmake_minimum_required(VERSION 3.5.1)
project(rgbd_face_auth)

find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(freenect2 REQUIRED)

set(BASIC_SOURCE_FILES src/basic_types.hpp src/picture.hpp)
set(LIBKINECT_SOURCE_FILES src/libkinect.hpp)
set(CMAKE_CXX_STANDARD 17)

include_directories($ENV{HOME}/freenect2/include)

add_executable(take_photo src/take_photo.cpp ${BASIC_SOURCE_FILES} ${LIBKINECT_SOURCE_FILES})
target_link_libraries(take_photo freenect)
target_link_libraries(take_photo ${OpenCV_LIBS})
target_link_libraries(take_photo ${freenect2_LIBRARIES})
target_link_libraries(take_photo ${ZLIB_LIBRARIES})