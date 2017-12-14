#ifndef RGBD_PICTURE_H
#define RGBD_PICTURE_H

#include <cmath>
#include <tuple>
#include <fstream>
#include <stdexcept>

#include "basic_types.h"

template<uint8_t KinectT>
class rgbd_picture_t {
    static_assert(KinectT == 1 || KinectT == 2, "This type of Kinect isn't supported yet.");

public:
    static size_t constexpr width = (KinectT == 2) ? 512 : 640, height = (KinectT == 2) ? 424 : 488;

    using dye_t = uint8_t; // type holding value of one colour
    using pixel_t = std::array<dye_t, 3>;
    using depth_t = uint64_t;

private:
    matrix<depth_t, height, width> picture_depth;
    matrix<pixel_t, height, width> picture_map;

    pixel_t deptg_to_rgb(depth_t const depth, depth_t const min_depth, depth_t const max_depth) const;

public:
    rgbd_picture_t() = default;
    rgbd_picture_t(std::string const input_file_name); 

    void update_bitmap(depth_t const min_depth, depth_t const max_depth);
    dye_t *raw_bitmap();
    
    // Template functions -- definition

    template<typename InputStreamT>
    void read_depth(InputStreamT &in) {
        for(auto &row : picture_depth) {
            for(auto &depth : row) {
                in >> depth;
            }
        }
    }

    template<typename InputStreamT>
    rgbd_picture_t(InputStreamT &in) {
        read_depth(in);

        update_bitmap(0, 10000);
    }


    template<typename OutputStreamT>
    void print_ppm(OutputStreamT &out) const {
        char const endl = '\n';

        out << "P3" << endl;
        out << "#Some deep comment." << endl;
        out << width << ' ' << height << endl;
        out << 255 << endl;

        for(auto const &row : picture_map) {
            for(auto const &pixel : row) {
                for(auto const dye : pixel) {
                    out << ' ' << static_cast<uint32_t>(dye) << ' ';
                }
                out << endl;
            }
        }
    }
};

// Pirivate definition

template<uint8_t KinectT>
typename rgbd_picture_t<KinectT>::pixel_t rgbd_picture_t<KinectT>::deptg_to_rgb(depth_t const depth,
depth_t const min_depth, depth_t const max_depth) const {
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
 

// Public definitions

template<uint8_t KinectT>
rgbd_picture_t<KinectT>::rgbd_picture_t(std::string const input_file_name) {
    std::fstream in(input_file_name, std::fstream::in);
    read_depth(in);

    update_bitmap(0, 10000);
}

template<uint8_t KinectT>
void rgbd_picture_t<KinectT>::update_bitmap(depth_t const min_depth, depth_t const max_depth) {
        for(size_t i = 0; i < height; ++i) {
            for(size_t j = 0; j < width; ++j) {
                picture_map[i][j] = deptg_to_rgb(picture_depth[i][j], min_depth, max_depth);
            }
        }
}

template<uint8_t KinectT>
typename rgbd_picture_t<KinectT>::dye_t *rgbd_picture_t<KinectT>::raw_bitmap() {
    // TODO: Is this best solution? I don't think so...
    dye_t * const ret = reinterpret_cast<dye_t *>( malloc(height * width * 3) );
    dye_t const * const map = reinterpret_cast<dye_t const *>( picture_map.data() );

    for(size_t i = 0; i < 3 * width * height; ++i) {
        ret[i] = map[i];
    }
    
    return ret;
}

#endif // RGBD_PICTURE_H
