
simple_display: src/rgbd_picture.h src/basic_types.h src/simple_display.cpp
	# DEBUG !!!
	g++ -std=c++14 -Wall -Wextra -pedantic -g3 src/simple_display.cpp `wx-config --cxxflags` `wx-config --libs` -o simple_display

test_rgbd_picture: src/rgbd_picture.h src/basic_types.h test_rgbd_picture.cpp
	g++ -std=c++14 -Wall -Wextra -pedantic test_rgbd_picture.cpp -o test_rgbd_picture
