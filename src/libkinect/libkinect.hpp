#ifndef LIBKINECT_H
#define LIBKINECT_H

#include <cstdlib>
#include <libfreenect/libfreenect.h>

enum class FrameType {DEPTH, IR, RGB};

class Frame {
 public:
  Frame(FrameType type, int width, int height, void *data)
      : type(type), width(width), height(height), data(data) {};
  ~Frame();
  FrameType type;
  int width, height;
  void *data;
};

class KinectDevice {
 public:
  KinectDevice();
  ~KinectDevice();
  void startStreams(bool depth, bool rgb, bool ir);
  virtual void frameHandler(Frame frame) = 0;

 protected:
  int whichKinect;  // 1 or 2
  freenect_context *freenectContext;
  freenect_device *freenectDevice;
  void *videoBufferFreenect, *videoBufferMine;

 private:
  static void kinect1DepthCallback(
      freenect_device *device, void *depthVoid, uint32_t timestamp);
  static void kinect1VideoCallback(
      freenect_device *device, void *buffer, uint32_t timestamp);
};

#endif