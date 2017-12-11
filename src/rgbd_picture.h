#ifndef RGBD_PICTURE_H
#define RGBD_PICTURE_H

#include <cmath>
#include <tuple>
#include <stdexcept>

#include "basic_types.h"

template<uint8_t KinectT>
class rgbd_picture_t {
    static_assert(KinectT == 1 || KinectT == 2, "This type of Kinect isn't supported yet.");

    using dye_t = uint8_t; // type holding value of one colour
    using pixel_t = std::array<dye_t, 3>;
    using depth_t = uint64_t;

    static size_t constexpr WIDTH = (KinectT == 2) ? 512 : 640, HEIGHT = (KinectT == 2) ? 424 : 488;

    matrix<depth_t, HEIGHT, WIDTH> picture_depth;

    pixel_t deptg_to_rgb(depth_t const depth, depth_t const min_depth, depth_t const max_depth) const {
        if(depth == 0) {

            return {0, 0, 0};
        }

        if(depth < min_depth) {

            return {255, 20, 10};
        }

        if(depth > max_depth) {

            return {10, 10, 255};
        }

        long double percentage = static_cast<long double>(depth - min_depth) / (max_depth - min_depth + 1);

        /* FIRST TYPE OF COLORS */
        dye_t const green = 255 * std::pow(percentage, 5);
        dye_t const red = 255 * std::pow( std::pow(1 - (percentage - 0.5), 2)/8, 0.6);
        dye_t const blue = 255 * (1 - 0.8 * std::pow(percentage, 0.05));
    

        /* SECOND TYPE OF COLORS 
         * dye_t const red = 20 + 250 * percentage;
         * dye_t const green = 230 - 180 * percentage;
         * dye_t const blue = 250 - 240 * percentage;
         */

        return {red, green, blue};
    }
 
public:

    template<typename InputStreamT>
    rgbd_picture_t(InputStreamT &in) {
        for(auto &row : picture_depth) {
            for(auto &depth : row) {
                in >> depth;
            }
        }
    }


    template<typename OutputStreamT>
    void print_ppm(OutputStreamT& out, depth_t const min_depth, depth_t const max_depth) const {
        char const endl = '\n';

        out << "P3" << endl;
        out << "#Some deep comment." << endl;
        out << WIDTH << ' ' << HEIGHT << endl;
        out << 255 << endl;

        for(auto const &row : picture_depth) {
            for(auto const &depth : row) {
                for(auto const dye : deptg_to_rgb(depth, min_depth, max_depth)) {
                    out << ' ' << std::to_string(dye) << ' ';
                }
                out << endl;
            }
        }
    }


};

#endif // RGBD_PICTURE_H
