#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "libkinect.hpp"

char constexpr photos_directory[] = "../photos/";

bool depth_photo_taken, ir_photo_taken, color_photo_taken;

std::string get_current_time() {
   auto current_time           = std::chrono::system_clock::now();
   auto current_time_as_time_t = std::chrono::system_clock::to_time_t(current_time);

   char current_time_char[100];
   std::strftime(
      current_time_char, sizeof(current_time_char), "%Y-%m-%d-%H-%M-%S", std::gmtime(&current_time_as_time_t));
   std::string current_time_string = current_time_char;

   current_time_string += '-';
   std::string milliseconds = std::to_string(
      std::chrono::duration_cast<std::chrono::milliseconds>(current_time.time_since_epoch()).count() % 1000);
   current_time_string += std::string(3 - milliseconds.length(), '0') + milliseconds;

   return current_time_string;
}

class MyKinectDevice : public KinectDevice {
 public:
   explicit MyKinectDevice(int device_number) : KinectDevice(device_number) {}

   void frame_handler(Frame const &frame) const override {
      if (depth_photo_taken && color_photo_taken && ir_photo_taken) {
         exit(0);
      }
      std::string filename =
         photos_directory + get_current_time() + "-kinect" + std::to_string(get_kinect_version()) + "-";

      if (frame.type == FrameType::depth) {
         if (depth_photo_taken) {
            return;
         }
         std::ofstream depth_file(filename + "depth.txt");
         for (int i = 0; i < frame.height; ++i) {
            for (int j = 0; j < frame.width; ++j) {
               depth_file << frame.data.depth_or_ir_data[frame.width * i + j] << '\t';
            }
            depth_file << std::endl;
         }
         depth_file.close();
         depth_photo_taken = true;
         std::cout << "Depth photo taken.\n";
      } else if (frame.type == FrameType::ir) {
         if (ir_photo_taken) {
            return;
         }
         std::ofstream ir_file(filename + "ir.txt");
         for (int i = 0; i < frame.height; ++i) {
            for (int j = 0; j < frame.width; ++j) {
               ir_file << frame.data.depth_or_ir_data[frame.width * i + j] << '\t';
            }
            ir_file << std::endl;
         }
         ir_file.close();
         ir_photo_taken = true;
         std::cout << "IR photo taken.\n";
      } else if (frame.type == FrameType::color) {
         if (color_photo_taken) {
            return;
         }
         cv::Mat image(cv::Size(int(frame.width), int(frame.height)), CV_8UC3, frame.data.color_data);
         cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
         cv::imwrite(filename + "rgb.png", image);
         color_photo_taken = true;
         std::cout << "RGB photo taken.\n";
      }
   }
};

int main() {
   MyKinectDevice kienct_device(0);
   bool use_depth = true, use_rgb = true, use_ir = kienct_device.get_kinect_version() != 1;
   depth_photo_taken = !use_depth;
   color_photo_taken = !use_rgb;
   ir_photo_taken    = !use_ir;
   kienct_device.start_streams(use_depth, use_rgb, use_ir);
   return 0;
}