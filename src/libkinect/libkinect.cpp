#include "libkinect.hpp"

#include <exception>
#include <iostream>
#include <cstring>

KinectDevice::KinectDevice(int deviceNumber = 0) {
  if (freenect_init(&freenectContext, nullptr) != 0) {
    throw std::runtime_error("freenect_init() failed");
  }

  int kinect1Devices, kinect2Devices;
  kinect1Devices = freenect_num_devices(freenectContext);
  kinect2Devices = freenect2.enumerateDevices();

  if (deviceNumber < kinect1Devices) {
    if (freenect_open_device(
        freenectContext, &freenectDevice, deviceNumber) != 0) {
      freenect_shutdown(freenectContext);
      throw std::runtime_error("freenect_open_device() failed");
    }
    whichKinect = 1;
    freenect_set_user(freenectDevice, this);
    std::cout << "Using a Kinect v1 device.\n";
  } else if (deviceNumber < kinect1Devices + kinect2Devices) {
    freenect2Pipeline = new libfreenect2::CpuPacketPipeline();
    freenect2Device = freenect2.openDevice(
        deviceNumber + kinect1Devices,freenect2Pipeline);
    if (!freenect2Device) {
      throw std::runtime_error("Error opening a Kinect v2 device");
    }
    whichKinect = 2;
    std::cout << "Using a Kinect v2 device.\n";
  } else {
    throw std::invalid_argument(
        "Could not find a Kinect device with that number.");
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
      throw std::invalid_argument(
          "Kinect v1: can't stream RGB and IR at the same time");
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
      if (depth && ir) {
        resolution = FREENECT_RESOLUTION_MEDIUM;
      }
      // TODO: check if the high resolution IR stream ever works properly
      auto videoMode = rgb ? FREENECT_VIDEO_RGB : FREENECT_VIDEO_IR_10BIT;
      freenect_frame_mode frameMode =
          freenect_find_video_mode(resolution, videoMode);
      freenect_set_video_mode(freenectDevice, frameMode);
      videoBufferMine = new uint8_t[frameMode.bytes];
      videoBufferFreenect = new uint8_t[frameMode.bytes];
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
      throw std::invalid_argument(
          "Kinect v2 can't stream only one of (depth, IR)");
    }

    stopStreams();

    if (depth && ir) {
      kinect2IrAndDepthListener = new Kinect2IrAndDepthListener(this);
      freenect2Device->setIrAndDepthFrameListener(kinect2IrAndDepthListener);
    }
    if (rgb) {
      kinect2RgbListener = new Kinect2RgbListener(this);
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

int KinectDevice::getKinectVersion() const {
  if (whichKinect == 1 || whichKinect == 2) {
    return whichKinect;
  } else {
    throw std::runtime_error("Unexpected value of whichKinect");
  }
}

void KinectDevice::kinect1DepthCallback(
    freenect_device *device, void *data, uint32_t timestamp) {

  auto kinectDevice = static_cast<KinectDevice *>(freenect_get_user(device));
  freenect_frame_mode frameMode = freenect_get_current_depth_mode(device);
  auto convertedData = new float[frameMode.width * frameMode.height];
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

  if (buffer != kinectDevice->videoBufferFreenect) {
    throw std::runtime_error("An error occured with Kinect's video buffer.");
  }
  kinectDevice->videoBufferFreenect = kinectDevice->videoBufferMine;
  freenect_set_video_buffer(device, kinectDevice->videoBufferFreenect);
  kinectDevice->videoBufferMine = buffer;

  freenect_frame_mode frameMode = freenect_get_current_video_mode(device);
  FrameType frameType;
  FrameData convertedData{};
  if (frameMode.video_format == FREENECT_VIDEO_RGB) {
    frameType = FrameType::RGB;
    size_t dataSize = (size_t) frameMode.width * frameMode.height * 3;
    convertedData.rgbData = new uint8_t[dataSize];
    memcpy(convertedData.rgbData, buffer, dataSize * sizeof(uint8_t));
  } else if (frameMode.video_format == FREENECT_VIDEO_IR_10BIT) {
    frameType = FrameType::IR;
    convertedData.depthOrIrData = new float[frameMode.width * frameMode.height];
    for (size_t i = 0; i < frameMode.height; ++i) {
      for (size_t j = 0; j < frameMode.width; ++j) {
        convertedData.depthOrIrData[frameMode.width * i + j] =
            float(static_cast<uint16_t *>(buffer)[frameMode.width * i + j]);
      }
    }
  } else {
    std::cerr << "kinect1VideoCallback received an unexcepted video format.\n";
    return;
  }
  kinectDevice->frameHandler(Frame(
      frameType, size_t(frameMode.width),
      size_t(frameMode.height), convertedData));
}

KinectDevice::Kinect2IrAndDepthListener::Kinect2IrAndDepthListener(
    KinectDevice *kinectDevice) : kinectDevice(kinectDevice) {}

bool KinectDevice::Kinect2IrAndDepthListener::onNewFrame(
    libfreenect2::Frame::Type type, libfreenect2::Frame *frame) {

  size_t dataSize = frame->width * frame->height;
  auto convertedData = new float[dataSize];
  memcpy(convertedData, frame->data, dataSize * sizeof(float));
  FrameType frameType;
  if (type == libfreenect2::Frame::Type::Ir) {
    frameType = FrameType::IR;
  } else if (type == libfreenect2::Frame::Type::Depth) {
    frameType = FrameType::DEPTH;
  } else {
    std::cerr << "Kinect2IrAndDepthListener::onNewFrame"
                 "received an unexcepted video format.\n";
    return false;
  }
  kinectDevice->frameHandler(Frame(
      frameType, frame->width, frame->height, convertedData));
  return false;
}

KinectDevice::Kinect2RgbListener::Kinect2RgbListener(
    KinectDevice *kinectDevice) : kinectDevice(kinectDevice) {}

bool KinectDevice::Kinect2RgbListener::onNewFrame(
    libfreenect2::Frame::Type type, libfreenect2::Frame *frame) {

  if (type != libfreenect2::Frame::Type::Color
      || frame->format != libfreenect2::Frame::BGRX) {
    std::cerr << "Kinect2RgbListener::onNewFrame"
                 "received an unexcepted video format.\n";
    return false;
  }
  auto convertedData = new uint8_t[frame->width * frame->height * 3];
  auto data = static_cast<uint8_t *>(frame->data);
  for (size_t i = 0; i < frame->height; ++i) {
    for (size_t j = 0; j < frame->width; ++j) {
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
