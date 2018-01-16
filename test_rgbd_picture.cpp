#include "src/rgbd_picture.h"

#include <fstream>
#include <string>

int main(int argc, char *argv[]) {
    uint64_t min_depth = 0, max_depth = 10000;

    if(argc >= 2) {
        min_depth = std::stoll(argv[1]);
    }

    if(argc >= 3) {
        max_depth = std::stoll(argv[2]);
    }

    std::fstream test_file_1("kinect_test/photo_kinect1_depth.txt", std::fstream::in);
    rgbd_picture_t<1> first_test(test_file_1);

    first_test.update_bitmap(min_depth, max_depth);

    std::fstream output_file_1("picture_v1.ppm", std::fstream::out);
    first_test.print_ppm(output_file_1);


    std::fstream test_file_2("kinect2_test/photo_depth.txt", std::fstream::in);
    rgbd_picture_t<2> second_test(test_file_2);

    second_test.update_bitmap(min_depth, max_depth);

    std::fstream output_file_2("picture_v2.ppm", std::fstream::out);
    second_test.print_ppm(output_file_2);
}
