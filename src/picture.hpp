#ifndef PICTURE_HPP
#define PICTURE_HPP

#include <fstream>
#include <iostream>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "basic_types.hpp"

// Declarations

class Picture {
 public:
   class ColorFrame;
   class DepthOrIrFrame;

   Picture() = default;
   Picture(ColorFrame *color_frame, DepthOrIrFrame *depth_frame, DepthOrIrFrame *ir_frame);

   void save_all_to_files(std::string const &base_filename) const;
   void resize_all(size_t width, size_t height);

   ColorFrame *color_frame     = nullptr;
   DepthOrIrFrame *depth_frame = nullptr;
   DepthOrIrFrame *ir_frame    = nullptr;
};

class Picture::ColorFrame {
 public:
   struct ColorPixel {
      uint8_t blue, green, red;
   };

   explicit ColorFrame(Matrix<ColorPixel> *pixels);
   explicit ColorFrame(std::string const &filename);

   void save_to_file(std::string const &filename) const;
   void resize(size_t width, size_t height);

   Matrix<ColorPixel> *pixels = nullptr;
};

class Picture::DepthOrIrFrame {
 public:
   explicit DepthOrIrFrame(Matrix<float> *pixels, bool is_depth);
   explicit DepthOrIrFrame(std::string const &filename);

   void save_to_file(std::string const &filename) const;
   void resize(size_t width, size_t height);

   Matrix<float> *pixels = nullptr;
   bool is_depth;  // false means that it's an IR photo
};

// Definitions - ColorFrame

Picture::ColorFrame::ColorFrame(Matrix<ColorPixel> *pixels) : pixels(pixels) {}

Picture::ColorFrame::ColorFrame(std::string const &filename) {
   cv::Mat image = cv::imread(filename);
   auto width = static_cast<size_t>(image.size().width), height = static_cast<size_t>(image.size().height);
   pixels = new Matrix<ColorPixel>(height, width);
   for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
         auto pixel      = image.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
         (*pixels)[i][j] = ColorPixel{pixel[0], pixel[1], pixel[2]};
      }
   }
}

void Picture::ColorFrame::save_to_file(std::string const &filename) const {
   cv::Mat image(cv::Size(static_cast<int>(pixels->width), static_cast<int>(pixels->height)), CV_8UC3,
         (uint8_t *)(pixels->data()));
   cv::imwrite(filename, image);
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
         auto pixel      = destination_image.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
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

void Picture::DepthOrIrFrame::save_to_file(std::string const &filename) const {
   std::ofstream file_stream(filename, std::ofstream::binary);
   char header[12];
   if (is_depth) {
      memcpy(header, "PHDE", 4);
   } else {
      memcpy(header, "PHIR", 4);
   }
   reinterpret_cast<uint32_t *>(header)[1] = static_cast<uint32_t>(pixels->width);
   reinterpret_cast<uint32_t *>(header)[2] = static_cast<uint32_t>(pixels->height);
   file_stream.write(header, 12);
   file_stream.write(reinterpret_cast<char *>(pixels->data()), pixels->height * pixels->width * sizeof(float));
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
         auto pixel      = destination_image.at<float>(static_cast<int>(i), static_cast<int>(j));
         (*pixels)[i][j] = pixel;
      }
   }
}

// Definitions - Picture

Picture::Picture(ColorFrame *color_frame, DepthOrIrFrame *depth_frame, DepthOrIrFrame *ir_frame)
      : color_frame(color_frame), depth_frame(depth_frame), ir_frame(ir_frame) {}

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
