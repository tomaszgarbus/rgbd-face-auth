#ifndef LIBKINECT_H
#define LIBKINECT_H

#include <cstdlib>
#include <libfreenect/libfreenect.h>
#include <libfreenect2/libfreenect2.hpp>

enum class FrameType {DEPTH, IR, RGB};

class Frame {
 public:
  Frame(FrameType type, size_t width, size_t height, void *data)
      : type(type), width(width), height(height), data(data) {};
  ~Frame();
  FrameType type;
  size_t width, height;
  void *data;
};

class KinectDevice {
 public:
  explicit KinectDevice(int deviceNumber);
  ~KinectDevice();
  void startStreams(bool depth, bool rgb, bool ir);
  virtual void frameHandler(Frame frame) = 0;

 protected:
  int whichKinect = 0;  // 1 or 2
  // Kinect v1:
  freenect_context *freenectContext;
  freenect_device *freenectDevice;
  void *videoBufferFreenect, *videoBufferMine;
  // Kinect v2:
  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *freenect2Device = nullptr;
  libfreenect2::PacketPipeline *freenect2Pipeline = nullptr;

 private:
  static void kinect1DepthCallback(
      freenect_device *device, void *depthVoid, uint32_t timestamp);
  static void kinect1VideoCallback(
      freenect_device *device, void *buffer, uint32_t timestamp);
  class Kinect2IrAndDepthListener : public libfreenect2::FrameListener {
   public:
    bool onNewFrame(
        libfreenect2::Frame::Type type, libfreenect2::Frame *frame) override;
   private:
    KinectDevice *kinectDevice;
  };
  class Kinect2RgbListener : public libfreenect2::FrameListener {
   public:
    bool onNewFrame(
        libfreenect2::Frame::Type type, libfreenect2::Frame *frame) override;
   private:
    KinectDevice *kinectDevice;
  };
};

#endif