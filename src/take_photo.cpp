#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "basic_types.hpp"
#include "libkinect.hpp"

char constexpr photos_directory[] = "../photos/";

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

   void frame_handler(Picture const &picture) const override {
      std::string filename = photos_directory + get_current_time() + "-kinect" + std::to_string(which_kinect);

      auto start_time = std::chrono::system_clock::now();
      picture.save_all_to_files(filename);
      auto end_time = std::chrono::system_clock::now();

      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      if (picture.color_frame) {
         std::cout << "color: " << ms << " ms\n";
      }
      if (picture.depth_frame) {
         std::cout << "depth: " << ms << " ms\n";
      }
      if (picture.ir_frame) {
         std::cout << "ir:    " << ms << " ms\n";
      }
      std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << std::endl;
   }
};

int main() {
   MyKinectDevice kinect_device(0);
   bool use_depth = true, use_rgb = true, use_ir = kinect_device.which_kinect != 1;
   kinect_device.start_streams(use_depth, use_rgb, use_ir);
   while (true) {
   }
   return 0;
}