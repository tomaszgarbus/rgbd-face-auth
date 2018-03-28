#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "picture.hpp"

const float min_depth = 500.0;
const float max_depth = 1500.0;

int main(int argc, char **argv) {
   // argv[1] - input file
   // argv[2] - output file
   // argv[3] - thumbnail size
   Picture::DepthOrIrFrame frame(argv[1]);
   auto thumb_max_size = static_cast<size_t>(std::stoi(argv[3]));
   size_t thumb_width = thumb_max_size;
   size_t thumb_height = thumb_max_size;
   if (frame.pixels->width > frame.pixels->height) {
      thumb_height = thumb_max_size * frame.pixels->height / frame.pixels->width;
   } else {
      thumb_width = thumb_max_size * frame.pixels->width / frame.pixels->height;
   }
   frame.resize(thumb_width, thumb_height);
   auto int_pixels = new uint8_t[thumb_width * thumb_height];
   for (size_t i = 0; i < thumb_width * thumb_height; ++i) {
      float val = 255.0f * (frame.pixels->data()[i] - min_depth) / (max_depth - min_depth);
      val = std::min(val, 255.0f);
      val = std::max(val, 0.0f);
      int_pixels[i] = static_cast<uint8_t>(val);
   }
   cv::Mat current_image(cv::Size(static_cast<int>(thumb_width), static_cast<int>(thumb_height)), CV_8UC1, int_pixels);
   cv::Mat destination_image(cv::Size(static_cast<int>(thumb_width), static_cast<int>(thumb_height)), CV_8UC3);
   cv::applyColorMap(current_image, destination_image, cv::COLORMAP_RAINBOW);
   cv::imwrite(std::string(argv[2]) + ".png", destination_image);
   rename((std::string(argv[2]) + ".png").c_str(), argv[2]);
   return 0;
}