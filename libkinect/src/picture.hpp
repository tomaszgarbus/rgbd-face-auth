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

#ifndef PICTURE_HPP
#define PICTURE_HPP

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include <libfreenect2/libfreenect2.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <zlib.h>

#include "basic_types.hpp"

// Declarations

class Picture {
 public:
   class ColorFrame;
   class DepthOrIrFrame;

   Picture() = default;
   Picture(ColorFrame *color_frame, DepthOrIrFrame *depth_frame, DepthOrIrFrame *ir_frame);
   Picture(const Picture &src);
   ~Picture();

   void save_all_to_files(std::string const &base_filename) const;
   void resize_all(size_t width, size_t height);

   ColorFrame *color_frame = nullptr;
   DepthOrIrFrame *depth_frame = nullptr;
   DepthOrIrFrame *ir_frame = nullptr;
};

class Picture::ColorFrame {
 public:
   struct ColorPixel {
      uint8_t blue, green, red;
   };

   explicit ColorFrame(Matrix<ColorPixel> *pixels);
   explicit ColorFrame(std::string const &filename);
   ColorFrame(const ColorFrame &src);
   ~ColorFrame();

   void save_to_file(std::string const &filename) const;
   void resize(size_t width, size_t height);

   std::chrono::time_point<std::chrono::system_clock> time_received = std::chrono::system_clock::now();

   Matrix<ColorPixel> *pixels = nullptr;
};

class Picture::DepthOrIrFrame {
 public:
   DepthOrIrFrame(Matrix<float> *pixels, bool is_depth);
   explicit DepthOrIrFrame(std::string const &filename);
   DepthOrIrFrame(const DepthOrIrFrame &src);
   ~DepthOrIrFrame();

   void save_to_file(std::string const &filename) const;
   void resize(size_t width, size_t height);

   Matrix<float> *pixels = nullptr;
   bool is_depth;  // false means that it's an IR photo

   std::chrono::time_point<std::chrono::system_clock> time_received = std::chrono::system_clock::now();

   std::shared_ptr<libfreenect2::Frame> freenect2_frame = nullptr;
};

// Definitions - ColorFrame

Picture::ColorFrame::ColorFrame(Matrix<ColorPixel> *pixels) : pixels(pixels) {}

Picture::ColorFrame::ColorFrame(std::string const &filename) {
   cv::Mat image = cv::imread(filename);
   auto width = static_cast<size_t>(image.size().width), height = static_cast<size_t>(image.size().height);
   pixels = new Matrix<ColorPixel>(height, width);
   for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
         auto pixel = image.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
         (*pixels)[i][j] = ColorPixel{pixel[0], pixel[1], pixel[2]};
      }
   }
}

Picture::ColorFrame::ColorFrame(const Picture::ColorFrame &src)
      : pixels(new Matrix<Picture::ColorFrame::ColorPixel>(*src.pixels)), time_received(src.time_received) {}

Picture::ColorFrame::~ColorFrame() {
   delete pixels;
}

void Picture::ColorFrame::save_to_file(std::string const &filename) const {
   auto *pixels_copy = new Matrix<ColorPixel>(*pixels);
   std::thread t([filename, pixels_copy] {
      cv::Mat image(cv::Size(static_cast<int>(pixels_copy->width), static_cast<int>(pixels_copy->height)), CV_8UC3,
            (uint8_t *)(pixels_copy->data()));
      cv::imwrite(filename, image);
      delete pixels_copy;
   });
   t.detach();
}

void Picture::ColorFrame::resize(size_t const width, size_t const height) {
   cv::Mat current_image(cv::Size(static_cast<int>(pixels->width), static_cast<int>(pixels->height)), CV_8UC3,
         (uint8_t *)(pixels->data()));
   cv::Mat destination_image(cv::Size(static_cast<int>(width), static_cast<int>(height)), CV_8UC3);
   cv::resize(current_image, destination_image, destination_image.size());
   delete pixels;
   pixels = new Matrix<ColorPixel>(height, width);
   for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
         auto pixel = destination_image.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
         (*pixels)[i][j] = ColorPixel{pixel[0], pixel[1], pixel[2]};
      }
   }
}

// Definitions - DepthOrIrFrame

Picture::DepthOrIrFrame::DepthOrIrFrame(Matrix<float> *pixels, bool const is_depth)
      : pixels(pixels), is_depth(is_depth) {}

Picture::DepthOrIrFrame::DepthOrIrFrame(std::string const &filename) {
   std::ifstream file_stream(filename, std::ifstream::binary);
   if (file_stream) {
      char header[12];
      file_stream.read(header, 12);
      std::string magic(header, 4);
      if (magic == "PHDE") {
         is_depth = true;
      } else if (magic == "PHIR") {
         is_depth = false;
      } else {
         throw std::invalid_argument("Invalid magic in file " + filename);
      }
      size_t width = reinterpret_cast<uint32_t *>(header)[1], height = reinterpret_cast<uint32_t *>(header)[2];
      pixels = new Matrix<float>(height, width);
      file_stream.read(reinterpret_cast<char *>(pixels->data()), height * width * sizeof(float));
   } else {
      throw std::runtime_error("Error reading file " + filename);
   }
}

Picture::DepthOrIrFrame::DepthOrIrFrame(const Picture::DepthOrIrFrame &src)
      : pixels(new Matrix<float>(*src.pixels)), is_depth(src.is_depth), freenect2_frame(src.freenect2_frame),
        time_received(src.time_received) {}

Picture::DepthOrIrFrame::~DepthOrIrFrame() {
   delete pixels;
}

void Picture::DepthOrIrFrame::save_to_file(std::string const &filename) const {
   size_t pixels_size = pixels->height * pixels->width * sizeof(float);
   auto file_data = new char[12 + pixels_size];
   if (is_depth) {
      memcpy(file_data, "PHDE", 4);
   } else {
      memcpy(file_data, "PHIR", 4);
   }
   reinterpret_cast<uint32_t *>(file_data)[1] = static_cast<uint32_t>(pixels->width);
   reinterpret_cast<uint32_t *>(file_data)[2] = static_cast<uint32_t>(pixels->height);
   memcpy(file_data + 12, reinterpret_cast<char *>(pixels->data()), pixels_size);

   gzFile gz_file = gzopen((filename + ".gz").c_str(), "w");
   if (gz_file == Z_NULL) {
      throw std::runtime_error("gzopen() could not open file " + filename + ".gz");
   }
   if (gzwrite(gz_file, file_data, static_cast<unsigned int>(12 + pixels_size)) != 12 + pixels_size) {
      throw std::runtime_error("gzwrite() did not correctly write to file " + filename + ".gz");
   }
   if (gzclose(gz_file) != Z_OK) {
      throw std::runtime_error("gzclose() did not correctly close file " + filename + ".gz");
   }
   delete[] file_data;
}

void Picture::DepthOrIrFrame::resize(size_t width, size_t height) {
   cv::Mat current_image(cv::Size(static_cast<int>(pixels->width), static_cast<int>(pixels->height)), CV_32FC1,
         (uint8_t *)(pixels->data()));
   cv::Mat destination_image(cv::Size(static_cast<int>(width), static_cast<int>(height)), CV_32FC1);
   cv::resize(current_image, destination_image, destination_image.size());
   delete pixels;
   pixels = new Matrix<float>(height, width);
   for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
         auto pixel = destination_image.at<float>(static_cast<int>(i), static_cast<int>(j));
         (*pixels)[i][j] = pixel;
      }
   }
}

// Definitions - Picture

Picture::Picture(ColorFrame *color_frame, DepthOrIrFrame *depth_frame, DepthOrIrFrame *ir_frame)
      : color_frame(color_frame), depth_frame(depth_frame), ir_frame(ir_frame) {}

Picture::Picture(const Picture &src) {
   if (src.color_frame != nullptr) {
      color_frame = new ColorFrame(new Matrix<Picture::ColorFrame::ColorPixel>(*src.color_frame->pixels));
   }
   if (src.depth_frame != nullptr) {
      depth_frame = new DepthOrIrFrame(new Matrix<float>(*src.depth_frame->pixels), true);
   }
   if (src.ir_frame != nullptr) {
      ir_frame = new DepthOrIrFrame(new Matrix<float>(*src.ir_frame->pixels), false);
   }
}

Picture::~Picture() {
   delete color_frame;
   delete depth_frame;
   delete ir_frame;
}

void Picture::save_all_to_files(std::string const &base_filename) const {
   if (color_frame != nullptr) {
      color_frame->save_to_file(base_filename + ".png");
   }
   if (depth_frame != nullptr) {
      depth_frame->save_to_file(base_filename + ".depth");
   }
   if (ir_frame != nullptr) {
      ir_frame->save_to_file(base_filename + ".ir");
   }
}

void Picture::resize_all(size_t width, size_t height) {
   if (color_frame != nullptr) {
      color_frame->resize(width, height);
   }
   if (depth_frame != nullptr) {
      depth_frame->resize(width, height);
   }
   if (ir_frame != nullptr) {
      ir_frame->resize(width, height);
   }
}

#endif
