#ifndef LIBKINECT_HPP
#define LIBKINECT_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>

#include <libfreenect/libfreenect.h>
#include <libfreenect2/libfreenect2.hpp>

#include "picture.hpp"

// Declarations

class KinectDevice {
 public:
   explicit KinectDevice(int device_number);
   ~KinectDevice();
   void start_streams(bool depth, bool color, bool ir);
   void stop_streams();
   int get_kinect_version() const;
   virtual void frame_handler(const Frame &frame) const = 0;

 protected:
   int which_kinect   = 0;  // 1 or 2
   bool depth_running = false, ir_running = false, color_running = false;
   // Kinect v1:
   freenect_context *freenect1_context;
   freenect_device *freenect1_device;
   void *video_buffer_freenect1, *video_buffer_mine;
   // Kinect v2:
   libfreenect2::Freenect2 freenect2;
   libfreenect2::Freenect2Device *freenect2_device  = nullptr;
   libfreenect2::PacketPipeline *freenect2_pipeline = nullptr;

 private:
   static void kinect1_depth_callback(freenect_device *device, void *depth_void, uint32_t timestamp);
   static void kinect1_video_callback(freenect_device *device, void *buffer, uint32_t timestamp);

   class Kinect2DepthAndIrListener : public libfreenect2::FrameListener {
    public:
      explicit Kinect2DepthAndIrListener(KinectDevice *kinect_device);
      bool onNewFrame(libfreenect2::Frame::Type type, libfreenect2::Frame *frame) override;

    private:
      KinectDevice *kinect_device;
   };

   class Kinect2ColorListener : public libfreenect2::FrameListener {
    public:
      explicit Kinect2ColorListener(KinectDevice *kinect_device);
      bool onNewFrame(libfreenect2::Frame::Type type, libfreenect2::Frame *frame) override;

    private:
      KinectDevice *kinect_device;
   };

   Kinect2ColorListener *kinect2_color_listener             = nullptr;
   Kinect2DepthAndIrListener *kinect2_depth_and_ir_listener = nullptr;
};

// Definitions

KinectDevice::KinectDevice(int device_number = 0) {
   if (freenect_init(&freenect1_context, nullptr) != 0) {
      throw std::runtime_error("freenect_init() failed");
   }

   int kinect1_devices, kinect2_devices;
   kinect1_devices = freenect_num_devices(freenect1_context);
   kinect2_devices = freenect2.enumerateDevices();

   if (device_number < kinect1_devices) {
      if (freenect_open_device(freenect1_context, &freenect1_device, device_number) != 0) {
         freenect_shutdown(freenect1_context);
         throw std::runtime_error("freenect_open_device() failed");
      }
      freenect_set_user(freenect1_device, this);
      which_kinect = 1;
      std::cout << "Using a Kinect v1 device.\n";
   } else if (device_number < kinect1_devices + kinect2_devices) {
      freenect2_pipeline = new libfreenect2::CpuPacketPipeline();
      freenect2_device   = freenect2.openDevice(device_number + kinect1_devices, freenect2_pipeline);
      if (!freenect2_device) {
         throw std::runtime_error("Error opening a Kinect v2 device");
      }
      which_kinect = 2;
      std::cout << "Using a Kinect v2 device.\n";
   } else {
      throw std::invalid_argument("Could not find a Kinect device with that number.");
   }
}

KinectDevice::~KinectDevice() {
   if (which_kinect == 1) {
      freenect_close_device(freenect1_device);
      freenect_shutdown(freenect1_context);
   } else if (which_kinect == 2) {
      freenect2_device->stop();
      freenect2_device->close();
   }
}

void KinectDevice::start_streams(bool depth, bool color, bool ir) {
   if (which_kinect == 1) {
      if (color && ir) {
         throw std::invalid_argument("Kinect v1: can't stream RGB and IR at the same time");
      }

      stop_streams();

      if (depth) {
         freenect_set_depth_mode(
            freenect1_device, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED));
         freenect_set_depth_callback(freenect1_device, kinect1_depth_callback);
         freenect_start_depth(freenect1_device);
      }

      if (color || ir) {
         auto resolution = FREENECT_RESOLUTION_MEDIUM;
         // TODO: maybe give a choice for color stream resolution?
         // TODO: check if the high resolution IR stream ever works properly
         auto video_mode = color ? FREENECT_VIDEO_RGB : FREENECT_VIDEO_IR_10BIT;
         auto frame_mode = freenect_find_video_mode(resolution, video_mode);
         freenect_set_video_mode(freenect1_device, frame_mode);
         video_buffer_mine      = new uint8_t[frame_mode.bytes];
         video_buffer_freenect1 = new uint8_t[frame_mode.bytes];
         freenect_set_video_callback(freenect1_device, kinect1_video_callback);
         freenect_set_video_buffer(freenect1_device, video_buffer_freenect1);
         freenect_start_video(freenect1_device);
      }

      depth_running = depth;
      color_running = color;
      ir_running    = ir;

      while (freenect_process_events(freenect1_context) >= 0) {
      }
      // TODO: this loop should be moved to a thread
   } else if (which_kinect == 2) {
      if (int(depth) + int(ir) == 1) {
         throw std::invalid_argument("Kinect v2 can't stream only one of (depth, IR)");
      }

      stop_streams();

      if (depth && ir) {
         kinect2_depth_and_ir_listener = new Kinect2DepthAndIrListener(this);
         freenect2_device->setIrAndDepthFrameListener(kinect2_depth_and_ir_listener);
      }
      if (color) {
         kinect2_color_listener = new Kinect2ColorListener(this);
         freenect2_device->setColorFrameListener(kinect2_color_listener);
      }

      freenect2_device->startStreams(color, depth && ir);
      depth_running = depth;
      color_running = color;
      ir_running    = ir;
      while (true) {
      }
      // TODO: don't do this loop after moving the Kinect v1 loop to a thread
   }
}

void KinectDevice::stop_streams() {
   if (which_kinect == 1) {
      if (depth_running) {
         freenect_stop_depth(freenect1_device);
      }
      if (color_running || ir_running) {
         freenect_stop_video(freenect1_device);
      }
   } else if (which_kinect == 2) {
      if (depth_running || color_running || ir_running) {
         freenect2_device->stop();
      }
      delete kinect2_depth_and_ir_listener;
      delete kinect2_color_listener;
      kinect2_depth_and_ir_listener = nullptr;
      kinect2_color_listener        = nullptr;
   }
   depth_running = false;
   color_running = false;
   ir_running    = false;
}

int KinectDevice::get_kinect_version() const {
   if (which_kinect == 1 || which_kinect == 2) {
      return which_kinect;
   } else {
      throw std::runtime_error("Unexpected value of which_kinect");
   }
}

void KinectDevice::kinect1_depth_callback(freenect_device *device, void *depth_void, uint32_t timestamp) {
   auto kinect_device  = static_cast<KinectDevice *>(freenect_get_user(device));
   auto frame_mode     = freenect_get_current_depth_mode(device);
   auto converted_data = new float[frame_mode.width * frame_mode.height];
   for (size_t i = 0; i < frame_mode.height; ++i) {
      for (size_t j = 0; j < frame_mode.width; ++j) {
         converted_data[frame_mode.width * i + j] =
            float(static_cast<uint16_t *>(depth_void)[frame_mode.width * i + j]);
      }
   }
   kinect_device->frame_handler(
      Frame(FrameType::depth, size_t(frame_mode.width), size_t(frame_mode.height), converted_data));
}

void KinectDevice::kinect1_video_callback(freenect_device *device, void *buffer, uint32_t timestamp) {
   auto kinect_device = static_cast<KinectDevice *>(freenect_get_user(device));

   if (buffer != kinect_device->video_buffer_freenect1) {
      throw std::runtime_error("An error occured with Kinect's video buffer.");
   }
   kinect_device->video_buffer_freenect1 = kinect_device->video_buffer_mine;
   freenect_set_video_buffer(device, kinect_device->video_buffer_freenect1);
   kinect_device->video_buffer_mine = buffer;

   auto frame_mode = freenect_get_current_video_mode(device);
   FrameType frame_type;
   FrameData converted_data{};
   if (frame_mode.video_format == FREENECT_VIDEO_RGB) {
      frame_type                = FrameType::color;
      size_t data_size          = (size_t)frame_mode.width * frame_mode.height * 3;
      converted_data.color_data = new uint8_t[data_size];
      memcpy(converted_data.color_data, buffer, data_size * sizeof(uint8_t));
   } else if (frame_mode.video_format == FREENECT_VIDEO_IR_10BIT) {
      frame_type                      = FrameType::ir;
      converted_data.depth_or_ir_data = new float[frame_mode.width * frame_mode.height];
      for (size_t i = 0; i < frame_mode.height; ++i) {
         for (size_t j = 0; j < frame_mode.width; ++j) {
            converted_data.depth_or_ir_data[frame_mode.width * i + j] =
               float(static_cast<uint16_t *>(buffer)[frame_mode.width * i + j]);
         }
      }
   } else {
      std::cerr << "kinect1_video_callback received an unexcepted video format.\n";
      return;
   }
   kinect_device->frame_handler(Frame(frame_type, size_t(frame_mode.width), size_t(frame_mode.height), converted_data));
}

KinectDevice::Kinect2DepthAndIrListener::Kinect2DepthAndIrListener(KinectDevice *kinect_device)
      : kinect_device(kinect_device) {}

bool KinectDevice::Kinect2DepthAndIrListener::onNewFrame(libfreenect2::Frame::Type type, libfreenect2::Frame *frame) {
   size_t data_size    = frame->width * frame->height;
   auto converted_data = new float[data_size];
   memcpy(converted_data, frame->data, data_size * sizeof(float));
   FrameType frame_type;
   if (type == libfreenect2::Frame::Type::Ir) {
      frame_type = FrameType::ir;
   } else if (type == libfreenect2::Frame::Type::Depth) {
      frame_type = FrameType::depth;
   } else {
      std::cerr << "Kinect2DepthAndIrListener::onNewFrame"
                   "received an unexcepted video format.\n";
      return false;
   }
   kinect_device->frame_handler(Frame(frame_type, frame->width, frame->height, converted_data));
   return false;
}

KinectDevice::Kinect2ColorListener::Kinect2ColorListener(KinectDevice *kinect_device) : kinect_device(kinect_device) {}

bool KinectDevice::Kinect2ColorListener::onNewFrame(libfreenect2::Frame::Type type, libfreenect2::Frame *frame) {
   if (type != libfreenect2::Frame::Type::Color || frame->format != libfreenect2::Frame::BGRX) {
      std::cerr << "Kinect2ColorListener::onNewFrame"
                   "received an unexcepted video format.\n";
      return false;
   }
   auto converted_data = new uint8_t[frame->width * frame->height * 3];
   auto data           = static_cast<uint8_t *>(frame->data);
   for (size_t i = 0; i < frame->height; ++i) {
      for (size_t j = 0; j < frame->width; ++j) {
         // Convert BGRX to RGB.
         size_t pixel_index                  = i * frame->width + j;
         converted_data[3 * pixel_index]     = data[4 * pixel_index + 2];
         converted_data[3 * pixel_index + 1] = data[4 * pixel_index + 1];
         converted_data[3 * pixel_index + 2] = data[4 * pixel_index];
      }
   }
   kinect_device->frame_handler(Frame(FrameType::color, frame->width, frame->height, converted_data));
   return false;
}

#endif