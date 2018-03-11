cmake_minimum_required(VERSION 3.5.1)
project(rgbd_face_auth)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

find_package(ZLIB REQUIRED)

find_package(freenect2 REQUIRED)
include_directories($ENV{HOME}/freenect2/include)

set(wxWidgets_CONFIGURATION mswu)
find_package(wxWidgets COMPONENTS core base REQUIRED)
include(${wxWidgets_USE_FILE})

set(BASIC_SOURCE_FILES src/basic_types.hpp src/picture.hpp)
set(LIBKINECT_SOURCE_FILES src/libkinect.hpp)

add_executable(take_photo src/take_photo.cpp ${BASIC_SOURCE_FILES} ${LIBKINECT_SOURCE_FILES})
target_link_libraries(take_photo freenect)
target_link_libraries(take_photo ${OpenCV_LIBS})
target_link_libraries(take_photo ${freenect2_LIBRARIES})
target_link_libraries(take_photo ${ZLIB_LIBRARIES})

add_executable(simple_display src/simple_display.cpp ${BASIC_SOURCE_FILES} ${LIBKINECT_SOURCE_FILES})
target_link_libraries(simple_display freenect)
target_link_libraries(simple_display ${OpenCV_LIBS})
target_link_libraries(simple_display ${freenect2_LIBRARIES})
target_link_libraries(simple_display ${ZLIB_LIBRARIES})
target_link_libraries(simple_display ${wxWidgets_LIBRARIES})
