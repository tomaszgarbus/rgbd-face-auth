#include <ctime>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "libkinect/libkinect.hpp"

std::string PHOTOS_DIRECTORY = "../photos/";

bool depthPhotoTaken, irPhotoTaken, rgbPhotoTaken;

std::string getCurrentTime() {
  auto currentTime = std::chrono::system_clock::now();
  auto currentTimeAsTimeT = std::chrono::system_clock::to_time_t(currentTime);

  char currentTimeChar[100];
  std::strftime(currentTimeChar, sizeof(currentTimeChar),
                "%Y-%m-%d-%H-%M-%S", std::gmtime(&currentTimeAsTimeT));
  std::string currentTimeStr = currentTimeChar;

  currentTimeStr += '-';
  std::string milliseconds = std::to_string(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          currentTime.time_since_epoch()).count() % 1000);
  currentTimeStr += std::string(3 - milliseconds.length(), '0') + milliseconds;

  return currentTimeStr;
}

class MyKinectDevice : public KinectDevice {
 public:
  explicit MyKinectDevice(int deviceNumber) : KinectDevice(deviceNumber) {}
  void frameHandler(const Frame &frame) const override {
    if (depthPhotoTaken && rgbPhotoTaken && irPhotoTaken) {
      exit(0);
    }
    std::string filename = PHOTOS_DIRECTORY + getCurrentTime()
        + "-kinect" + std::to_string(getKinectVersion()) + "-";

    if (frame.type == FrameType::DEPTH) {
      if (depthPhotoTaken) {
        return;
      }
      std::ofstream depthFile(filename + "depth.txt");
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
      std::ofstream irFile(filename + "ir.txt");
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
      cv::imwrite(filename + "rgb.png", image);
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