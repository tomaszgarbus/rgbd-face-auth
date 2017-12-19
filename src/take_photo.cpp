#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "libkinect/libkinect.hpp"

bool depthPhotoTaken, irPhotoTaken, rgbPhotoTaken;

class MyKinectDevice : public KinectDevice {
 public:
  explicit MyKinectDevice(int deviceNumber) : KinectDevice(deviceNumber) {}
  void frameHandler(Frame frame) override {
    if (depthPhotoTaken && rgbPhotoTaken && irPhotoTaken)
      exit(0);
    if (frame.type == FrameType::DEPTH) {
      if (depthPhotoTaken)
        return;
      auto data = static_cast<float *>(frame.data);
      FILE *depthFile = fopen("../photo_kinect1_depth.txt", "w");
      for (int i = 0; i < frame.height; ++i) {
        for (int j = 0; j < frame.width; ++j) {
          fprintf(depthFile, "%f\t", data[frame.width * i + j]);
        }
        fprintf(depthFile, "\n");
      }
      fclose(depthFile);
      depthPhotoTaken = true;
      printf("Depth photo taken.\n");
    } else if (frame.type == FrameType::IR) {
      if (irPhotoTaken)
        return;
      auto data = static_cast<float *>(frame.data);
      FILE *depthFile = fopen("../photo_kinect1_ir.txt", "w");
      for (int i = 0; i < frame.height; ++i) {
        for (int j = 0; j < frame.width; ++j) {
          fprintf(depthFile, "%f\t", data[frame.width * i + j]);
        }
        fprintf(depthFile, "\n");
      }
      fclose(depthFile);
      irPhotoTaken = true;
      printf("IR photo taken.\n");
    } else if (frame.type == FrameType::RGB) {
      if (rgbPhotoTaken)
        return;
      auto data = static_cast<uint8_t *>(frame.data);
      cv::Mat image(
          cv::Size(int(frame.width), int(frame.height)),
          CV_8UC3, videoBufferMine);
      cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
      cv::imwrite("../photo_kinect1_rgb.png", image);
      rgbPhotoTaken = true;
      printf("RGB photo taken.\n");
    }
  }
};

int main() {
  MyKinectDevice kinectDevice(0);
  bool useDepth = true, useRgb = true, useIr = false;
  depthPhotoTaken = !useDepth;
  rgbPhotoTaken = !useRgb;
  irPhotoTaken = !useIr;
  kinectDevice.startStreams(useDepth, useRgb, useIr);
  return 0;
}