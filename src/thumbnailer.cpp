/*
   Novelty face authentication with liveness detection using depth and IR camera
   Copyright (C) 2017-2018
   Tomasz Garbus, Dominik Klemba, Jan Ludziejewski, Łukasz Raszkiewicz

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "picture.hpp"

const float min_depth = 500.0;
const float max_depth = 1500.0;
const float max_ir_v1 = 1024.0;
const float max_ir_v2 = 65535.0;

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
   float max_ir = 0.0;
   if (!frame.is_depth) {
      for (size_t i = 0; i < thumb_width * thumb_height; ++i) {
         max_ir = std::max(max_ir, frame.pixels->data()[i]);
      }
      if (max_ir <= 1024.0) {  // Kinect v1
         max_ir = max_ir_v1;
      } else {  // Kinect v2
         max_ir = std::max(max_ir, max_ir_v2);
      }
   }
   for (size_t i = 0; i < thumb_width * thumb_height; ++i) {
      float val;
      if (frame.is_depth) {
         val = 255.0f * (frame.pixels->data()[i] - min_depth) / (max_depth - min_depth);
      } else {
         val = 255.0f * frame.pixels->data()[i] / max_ir;
      }
      val = std::min(val, 255.0f);
      val = std::max(val, 0.0f);
      int_pixels[i] = static_cast<uint8_t>(val);
   }
   cv::Mat current_image(cv::Size(static_cast<int>(thumb_width), static_cast<int>(thumb_height)), CV_8UC1, int_pixels);
   if (frame.is_depth) {
      cv::Mat destination_image(cv::Size(static_cast<int>(thumb_width), static_cast<int>(thumb_height)), CV_8UC3);
      cv::applyColorMap(current_image, destination_image, cv::COLORMAP_RAINBOW);
      cv::imwrite(std::string(argv[2]) + ".png", destination_image);
   } else {
      cv::imwrite(std::string(argv[2]) + ".png", current_image);
   }
   rename((std::string(argv[2]) + ".png").c_str(), argv[2]);
   return 0;
}
