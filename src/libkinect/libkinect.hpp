#ifndef LIBKINECT_H
#define LIBKINECT_H

#include <cstdint>
#include <cstddef>
#include <libfreenect/libfreenect.h>
#include <libfreenect2/libfreenect2.hpp>

enum class FrameType {DEPTH, IR, RGB};

union FrameData {
 public:
  float *depthOrIrData;
  uint8_t *rgbData;

  FrameData() = default;
  explicit FrameData(float *depthOrIrData) : depthOrIrData(depthOrIrData) {};
  explicit FrameData(uint8_t *rgbData) : rgbData(rgbData) {};
};

class Frame {
 public:
  Frame(FrameType type, size_t width, size_t height, FrameData data)
      : type(type), width(width), height(height), data(data) {};
  Frame(FrameType type, size_t width, size_t height, float *depthOrIrData)
      : type(type), width(width), height(height), data(depthOrIrData) {};
  Frame(FrameType type, size_t width, size_t height, uint8_t *rgbData)
      : type(type), width(width), height(height), data(rgbData) {};
  FrameType type;
  size_t width, height;
  FrameData data;
};

class KinectDevice {
 public:
  explicit KinectDevice(int deviceNumber);
  ~KinectDevice();
  void startStreams(bool depth, bool rgb, bool ir);
  void stopStreams();
  int getKinectVersion() const;
  virtual void frameHandler(const Frame &frame) const = 0;

 protected:
  int whichKinect = 0;  // 1 or 2
  bool depthRunning = false, irRunning = false, rgbRunning = false;
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
    explicit Kinect2IrAndDepthListener(KinectDevice *kinectDevice);
    bool onNewFrame(
        libfreenect2::Frame::Type type, libfreenect2::Frame *frame) override;

   private:
    KinectDevice *kinectDevice;
  };

  class Kinect2RgbListener : public libfreenect2::FrameListener {
   public:
    explicit Kinect2RgbListener(KinectDevice *kinectDevice);
    bool onNewFrame(
        libfreenect2::Frame::Type type, libfreenect2::Frame *frame) override;

   private:
    KinectDevice *kinectDevice;
  };

  Kinect2RgbListener *kinect2RgbListener = nullptr;
  Kinect2IrAndDepthListener *kinect2IrAndDepthListener = nullptr;
};

#endif