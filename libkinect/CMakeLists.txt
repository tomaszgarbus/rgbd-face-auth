cmake_minimum_required(VERSION 3.8.2)
project(rgbd_face_auth)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

find_package(OpenCV REQUIRED)

find_package(ZLIB REQUIRED)

find_package(freenect2 REQUIRED)
include_directories($ENV{HOME}/freenect2/include)

set(wxWidgets_CONFIGURATION mswu)
find_package(wxWidgets COMPONENTS core base REQUIRED)
include(${wxWidgets_USE_FILE})

set(BASIC_SOURCE_FILES src/basic_types.hpp src/picture.hpp)
set(LIBKINECT_SOURCE_FILES src/libkinect.hpp)

add_executable(live_display src/live_display.cpp ${BASIC_SOURCE_FILES} ${LIBKINECT_SOURCE_FILES})
target_link_libraries(live_display freenect)
target_link_libraries(live_display ${OpenCV_LIBS})
target_link_libraries(live_display ${freenect2_LIBRARIES})
target_link_libraries(live_display ${ZLIB_LIBRARIES})
target_link_libraries(live_display ${wxWidgets_LIBRARIES})

add_executable(file_display src/file_display.cpp ${BASIC_SOURCE_FILES})
target_link_libraries(file_display ${OpenCV_LIBS})
target_link_libraries(file_display ${ZLIB_LIBRARIES})
target_link_libraries(file_display ${wxWidgets_LIBRARIES})

add_executable(thumbnailer src/thumbnailer.cpp ${BASIC_SOURCE_FILES})
target_link_libraries(thumbnailer ${OpenCV_LIBS})
target_link_libraries(thumbnailer ${ZLIB_LIBRARIES})
