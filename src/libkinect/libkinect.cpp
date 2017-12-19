#include "libkinect.hpp"

#include <iostream>
#include <cstring>
#include <cassert>
#include <libfreenect/libfreenect.h>

Frame::~Frame() {
  free(data);
}

KinectDevice::KinectDevice(int deviceNumber = 0) {
  if (freenect_init(&freenectContext, nullptr) != 0) {
    fprintf(stderr, "freenect_init() failed.\n");
    exit(1);
  }

  int kinect1Devices, kinect2Devices;
  kinect1Devices = freenect_num_devices(freenectContext);
  kinect2Devices = freenect2.enumerateDevices();

  if (deviceNumber < kinect1Devices) {
    if (freenect_open_device(
        freenectContext, &freenectDevice, deviceNumber) != 0) {
      fprintf(stderr, "freenect_open_device() failed.\n");
      freenect_shutdown(freenectContext);
      exit(1);
    }
    whichKinect = 1;
    freenect_set_user(freenectDevice, this);
    fprintf(stderr, "Using a Kinect v1 device.\n");
  } else if (deviceNumber < kinect1Devices + kinect2Devices) {
    freenect2Pipeline = new libfreenect2::CpuPacketPipeline();
    freenect2Device = freenect2.openDevice(
        deviceNumber + kinect1Devices,freenect2Pipeline);
    if (!freenect2Device) {
      fprintf(stderr, "Error opening a Kinect v2 device.\n");
      exit(1);
    }
    fprintf(stderr, "Using a Kinect v2 device.\n");
  } else {
    fprintf(stderr, "There are less than %d devices connected.\n",
            deviceNumber + 1);
    exit(1);
  }
}

KinectDevice::~KinectDevice() {
  if (whichKinect == 1) {
    freenect_close_device(freenectDevice);
    freenect_shutdown(freenectContext);
  } else if (whichKinect == 2) {
    freenect2Device->stop();
    freenect2Device->close();
  }
}

void KinectDevice::startStreams(bool depth, bool rgb, bool ir) {
  if (whichKinect == 1) {
    if (rgb && ir) {
      fprintf(stderr, "Kinect v1: can't stream RGB and IR at the same time.\n");
      return;
    }

    stopStreams();

    if (depth) {
      freenect_set_depth_mode(freenectDevice, freenect_find_depth_mode(
          FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED));
      freenect_set_depth_callback(freenectDevice, kinect1DepthCallback);
      freenect_start_depth(freenectDevice);
    }

    if (rgb || ir) {
      auto resolution = FREENECT_RESOLUTION_HIGH;
      if (depth && ir)
        resolution = FREENECT_RESOLUTION_MEDIUM;
      // TODO: check if the high resolution IR stream ever works properly
      auto videoMode = rgb ? FREENECT_VIDEO_RGB : FREENECT_VIDEO_IR_10BIT;
      freenect_frame_mode frameMode =
          freenect_find_video_mode(resolution, videoMode);
      freenect_set_video_mode(freenectDevice, frameMode);
      videoBufferMine = malloc(size_t(frameMode.bytes));
      videoBufferFreenect = malloc(size_t(frameMode.bytes));
      freenect_set_video_callback(freenectDevice, kinect1VideoCallback);
      freenect_set_video_buffer(freenectDevice, videoBufferFreenect);
      freenect_start_video(freenectDevice);
    }

    depthRunning = depth;
    rgbRunning = rgb;
    irRunning = ir;
    while (freenect_process_events(freenectContext) >= 0);
    // TODO: this loop should be moved to a thread
  } else if (whichKinect == 2) {
    if (int(depth) + int(ir) == 1) {
      fprintf(stderr, "Kinect v2 can't stream only one of (depth, IR).\n");
      return;
    }

    stopStreams();

    if (depth && ir) {
      kinect2IrAndDepthListener = new Kinect2IrAndDepthListener();
      freenect2Device->setIrAndDepthFrameListener(kinect2IrAndDepthListener);
    }
    if (rgb) {
      kinect2RgbListener = new Kinect2RgbListener();
      freenect2Device->setColorFrameListener(kinect2RgbListener);
    }

    freenect2Device->startStreams(rgb, depth && ir);
    depthRunning = depth;
    rgbRunning = rgb;
    irRunning = ir;
    while (true);
    // TODO: don't do this after moving the Kinect v1 loop to a thread
  }
}

void KinectDevice::stopStreams() {
  if (whichKinect == 1) {
    if (depthRunning) {
      freenect_stop_depth(freenectDevice);
    }
    if (rgbRunning || irRunning) {
      freenect_stop_video(freenectDevice);
    }
  } else if (whichKinect == 2) {
    if (depthRunning || rgbRunning || irRunning) {
      freenect2Device->stop();
    }
    delete kinect2IrAndDepthListener;
    delete kinect2RgbListener;
    kinect2IrAndDepthListener = nullptr;
    kinect2RgbListener = nullptr;
  }
  depthRunning = false;
  rgbRunning = false;
  irRunning = false;
}

void KinectDevice::kinect1DepthCallback(
    freenect_device *device, void *data, uint32_t timestamp) {

  auto kinectDevice = static_cast<KinectDevice *>(freenect_get_user(device));
  freenect_frame_mode frameMode = freenect_get_current_depth_mode(device);
  auto convertedData = static_cast<float *>(
      malloc(frameMode.width * frameMode.height * sizeof(float)));
  for (size_t i = 0; i < frameMode.height; ++i) {
    for (size_t j = 0; j < frameMode.width; ++j) {
      convertedData[frameMode.width * i + j] =
          float(static_cast<uint16_t *>(data)[frameMode.width * i + j]);
    }
  }
  kinectDevice->frameHandler(Frame(
      FrameType::DEPTH, size_t(frameMode.width),
      size_t(frameMode.height), convertedData));
}

void KinectDevice::kinect1VideoCallback(
    freenect_device *device, void *buffer, uint32_t timestamp) {

  auto kinectDevice = static_cast<KinectDevice *>(freenect_get_user(device));

  assert(buffer == kinectDevice->videoBufferFreenect);
  kinectDevice->videoBufferFreenect = kinectDevice->videoBufferMine;
  freenect_set_video_buffer(device, kinectDevice->videoBufferFreenect);
  kinectDevice->videoBufferMine = buffer;

  freenect_frame_mode frameMode = freenect_get_current_video_mode(device);
  FrameType frameType;
  void *convertedData;
  if (frameMode.video_format == FREENECT_VIDEO_RGB) {
    frameType = FrameType::RGB;
    size_t dataSize = frameMode.width * frameMode.height * 3 * sizeof(uint8_t);
    assert(dataSize == frameMode.bytes);
    convertedData = malloc(dataSize);
    memcpy(convertedData, buffer, dataSize);
  } else if (frameMode.video_format == FREENECT_VIDEO_IR_10BIT) {
    frameType = FrameType::IR;
    convertedData = malloc(
        frameMode.width * frameMode.height * 3 * sizeof(float));
    for (size_t i = 0; i < frameMode.height; ++i) {
      for (size_t j = 0; j < frameMode.width; ++j) {
        static_cast<float *>(convertedData)[frameMode.width * i + j] =
            float(static_cast<uint16_t *>(buffer)[frameMode.width * i + j]);
      }
    }
  } else {
    fprintf(stderr, "Invalid video format.\n");
    return;
  }
  kinectDevice->frameHandler(Frame(
      frameType, size_t(frameMode.width),
      size_t(frameMode.height), convertedData));
}

bool KinectDevice::Kinect2IrAndDepthListener::onNewFrame(
    libfreenect2::Frame::Type type, libfreenect2::Frame *frame) {

  size_t dataSize = frame->width * frame->height * sizeof(float);
  auto convertedData = static_cast<float *>(malloc(dataSize));
  memcpy(convertedData, frame->data, dataSize);
  FrameType frameType;
  if (type == libfreenect2::Frame::Type::Ir) {
    frameType = FrameType::IR;
  } else if (type == libfreenect2::Frame::Type::Depth) {
    frameType = FrameType::DEPTH;
  } else {
    fprintf(stderr, "Invalid video format.\n");
    return false;
  }
  kinectDevice->frameHandler(Frame(
      frameType, frame->width, frame->height, convertedData));
  return false;
}

bool KinectDevice::Kinect2RgbListener::onNewFrame(
    libfreenect2::Frame::Type type, libfreenect2::Frame *frame) {

  assert(type == libfreenect2::Frame::Type::Color);
  assert(frame->format == libfreenect2::Frame::BGRX);
  auto convertedData = static_cast<uint8_t *>(malloc(
      frame->width * frame->height * sizeof(uint8_t) * 3));
  auto data = static_cast<uint8_t *>(frame->data);
  for (size_t i = 0; i < frame->height; ++i) {
    for (size_t j = 0; j < frame->height; ++j) {
      // Convert BGRX to RGB.
      size_t pixelIndex = i * frame->width + j;
      convertedData[3 * pixelIndex] = data[4 * pixelIndex + 2];
      convertedData[3 * pixelIndex + 1] = data[4 * pixelIndex + 1];
      convertedData[3 * pixelIndex + 2] = data[4 * pixelIndex];
    }
  }
  kinectDevice->frameHandler(Frame(
      FrameType::RGB, frame->width, frame->height, convertedData));
  return false;
}
