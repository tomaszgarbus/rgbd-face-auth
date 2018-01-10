#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "libkinect/libkinect.hpp"

bool depthPhotoTaken, irPhotoTaken, rgbPhotoTaken;

class MyKinectDevice : public KinectDevice {
 public:
  explicit MyKinectDevice(int deviceNumber) : KinectDevice(deviceNumber) {}
  void frameHandler(const Frame &frame) const override {
    if (depthPhotoTaken && rgbPhotoTaken && irPhotoTaken) {
      exit(0);
    }
    if (frame.type == FrameType::DEPTH) {
      if (depthPhotoTaken) {
        return;
      }
      std::ofstream depthFile("../photo_kinect1_depth.txt");
      for (int i = 0; i < frame.height; ++i) {
        for (int j = 0; j < frame.width; ++j) {
          depthFile << frame.data.depthOrIrData[frame.width * i + j] << '\t';
        }
        depthFile << std::endl;
      }
      depthFile.close();
      depthPhotoTaken = true;
      std::cout << "Depth photo taken.\n";
    } else if (frame.type == FrameType::IR) {
      if (irPhotoTaken) {
        return;
      }
      std::ofstream irFile("../photo_kinect1_ir.txt");
      for (int i = 0; i < frame.height; ++i) {
        for (int j = 0; j < frame.width; ++j) {
          irFile << frame.data.depthOrIrData[frame.width * i + j] << '\t';
        }
        irFile << std::endl;
      }
      irFile.close();
      irPhotoTaken = true;
      std::cout << "IR photo taken.\n";
    } else if (frame.type == FrameType::RGB) {
      if (rgbPhotoTaken) {
        return;
      }
      cv::Mat image(
          cv::Size(int(frame.width), int(frame.height)),
          CV_8UC3, frame.data.rgbData);
      cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
      cv::imwrite("../photo_kinect1_rgb.png", image);
      rgbPhotoTaken = true;
      std::cout << "RGB photo taken.\n";
    }
  }
};

int main() {
  MyKinectDevice kinectDevice(0);
  bool useDepth = true,
      useRgb = true,
      useIr = kinectDevice.getKinectVersion() != 1;
  depthPhotoTaken = !useDepth;
  rgbPhotoTaken = !useRgb;
  irPhotoTaken = !useIr;
  kinectDevice.startStreams(useDepth, useRgb, useIr);
  return 0;
}