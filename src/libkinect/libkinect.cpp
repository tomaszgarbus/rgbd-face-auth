#include "libkinect.hpp"

#include <iostream>
#include <cstring>
#include <cassert>
#include <libfreenect/libfreenect.h>

Frame::~Frame() {
  free(data);
}

KinectDevice::KinectDevice() {
  if (freenect_init(&freenectContext, nullptr) != 0) {
    fprintf(stderr, "freenect_init() failed.\n");
    exit(1);
  }
  // TODO: init Kinect 2

  int kinect1Devices, kinect2Devices;
  kinect1Devices = freenect_num_devices(freenectContext);
  kinect2Devices = 0; // TODO

  if (kinect1Devices <= 0 && kinect2Devices <= 0) {
    freenect_shutdown(freenectContext);
    printf("No Kinect devices found.\n");
    exit(1);
  } else if (kinect1Devices > 0) {
    if (freenect_open_device(freenectContext, &freenectDevice, 0) != 0) {
      fprintf(stderr, "freenect_open_device() failed.\n");
      freenect_shutdown(freenectContext);
      exit(1);
    }
    whichKinect = 1;
    freenect_set_user(freenectDevice, this);
    fprintf(stderr, "Using a Kinect 1 device.\n");
  } else if (kinect2Devices > 0) {
    // TODO
    fprintf(stderr, "Using a Kinect 2 device.\n");
  }
}

KinectDevice::~KinectDevice() {
  if (whichKinect == 1) {
    freenect_close_device(freenectDevice);
    freenect_shutdown(freenectContext);
  }
}

void KinectDevice::startStreams(bool depth, bool rgb, bool ir) {
  if (whichKinect == 1) {
    if (!depth)
      freenect_stop_depth(freenectDevice);
    if (!rgb && !ir)
      freenect_stop_video(freenectDevice);

    if (rgb && ir) {
      fprintf(stderr, "Can't stream RGB and IR at the same time.\n");
      return;
    }

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
    while (freenect_process_events(freenectContext) >= 0) {}
    // TODO: this ^ loop should be moved to a thread
  } else if (whichKinect == 2) {
    // TODO
  }
}

void KinectDevice::kinect1DepthCallback(
    freenect_device *device, void *data, uint32_t timestamp) {

  auto kinectDevice = static_cast<KinectDevice *>(freenect_get_user(device));
  freenect_frame_mode frameMode = freenect_get_current_depth_mode(device);
  auto convertedData = static_cast<float *>(
      malloc(frameMode.width * frameMode.height * sizeof(float)));
  for (int i = 0; i < frameMode.height; ++i) {
    for (int j = 0; j < frameMode.width; ++j) {
      convertedData[frameMode.width * i + j] =
          float(static_cast<uint16_t *>(data)[frameMode.width * i + j]);
    }
  }
  kinectDevice->frameHandler(Frame(
      FrameType::DEPTH, frameMode.width, frameMode.height, convertedData));
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
    for (int i = 0; i < frameMode.height; ++i) {
      for (int j = 0; j < frameMode.width; ++j) {
        static_cast<float *>(convertedData)[frameMode.width * i + j] =
            float(static_cast<uint16_t *>(buffer)[frameMode.width * i + j]);
      }
    }
  } else {
    fprintf(stderr, "Invalid video format.\n");
    return;
  }
  kinectDevice->frameHandler(
      Frame(frameType, frameMode.width, frameMode.height, convertedData));
}