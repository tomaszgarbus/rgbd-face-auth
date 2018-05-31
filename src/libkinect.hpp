#ifndef LIBKINECT_HPP
#define LIBKINECT_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <thread>

#include <libfreenect/libfreenect.h>
#include <libfreenect2/libfreenect2.hpp>

#include "picture.hpp"

// Declarations

class KinectDevice {
 public:
   explicit KinectDevice(int device_number);
   ~KinectDevice();

   void start_streams(bool color, bool depth, bool ir);
   void stop_streams();
   virtual void frame_handler(Picture const &picture) const = 0;

   int which_kinect = 0;  // 1 or 2 set in constructor

 protected:
   bool color_running = false, depth_running = false, ir_running = false;
   // Kinect v1:
   freenect_context *freenect1_context = nullptr;
   freenect_device *freenect1_device = nullptr;
   void *video_buffer_freenect1 = nullptr, *video_buffer_mine = nullptr;
   // Kinect v2:
   libfreenect2::Freenect2 freenect2;
   libfreenect2::Freenect2Device *freenect2_device = nullptr;
   libfreenect2::PacketPipeline *freenect2_pipeline = nullptr;

 private:
   std::atomic_flag kinect1_run_event_loop = ATOMIC_FLAG_INIT;
   std::thread *kinect1_event_thread = nullptr;
   void kinect1_process_events();

   static void kinect1_depth_callback(freenect_device *device, void *depth_void, uint32_t timestamp);
   static void kinect1_video_callback(freenect_device *device, void *buffer, uint32_t timestamp);

   class Kinect2ColorListener : public libfreenect2::FrameListener {
    public:
      explicit Kinect2ColorListener(KinectDevice *kinect_device);
      bool onNewFrame(libfreenect2::Frame::Type type, libfreenect2::Frame *frame) override;

    private:
      KinectDevice *kinect_device;
   };

   class Kinect2DepthAndIrListener : public libfreenect2::FrameListener {
    public:
      explicit Kinect2DepthAndIrListener(KinectDevice *kinect_device);
      bool onNewFrame(libfreenect2::Frame::Type type, libfreenect2::Frame *frame) override;

    private:
      KinectDevice *kinect_device;
   };

   Kinect2ColorListener *kinect2_color_listener = nullptr;
   Kinect2DepthAndIrListener *kinect2_depth_and_ir_listener = nullptr;
};

// Definitions

KinectDevice::KinectDevice(int device_number = 0) {
   if (freenect_init(&freenect1_context, nullptr) != 0) {
      throw std::runtime_error("freenect_init() failed");
   }

   int kinect1_devices, kinect2_devices;
   kinect1_devices = freenect_num_devices(freenect1_context);
   if (kinect1_devices < 0) {
      throw std::runtime_error("freenect_num_devices() failed");
   }
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
      freenect2_device = freenect2.openDevice(device_number - kinect1_devices);
      if (!freenect2_device) {
         throw std::runtime_error("freenect2.openDevice() failed");
      }
      which_kinect = 2;
      std::cout << "Using a Kinect v2 device.\n";
   } else {
      throw std::invalid_argument("Could not find a Kinect device with that number.");
   }
}

KinectDevice::~KinectDevice() {
   stop_streams();
   if (which_kinect == 1) {
      if (freenect_close_device(freenect1_device) != 0) {
         std::cerr << "A problem occured in freenect_close_device()\n";
      }
      if (freenect_shutdown(freenect1_context) != 0) {
         std::cerr << "A problem occured in freenect_shutdown()\n";
      }
   } else if (which_kinect == 2) {
      if (!freenect2_device->stop()) {
         std::cerr << "A problem occured in freenect2_device->stop()\n";
      }
      if (!freenect2_device->close()) {
         std::cerr << "A problem occured in freenect2_device->close()\n";
      }
   }
}

void KinectDevice::start_streams(bool color, bool depth, bool ir) {
   if (which_kinect == 1) {
      if (color && ir) {
         throw std::invalid_argument("Kinect v1: can't stream RGB and IR at the same time");
      }

      stop_streams();

      if (depth) {
         if (freenect_set_depth_mode(
                   freenect1_device, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED))
               != 0) {
            throw std::runtime_error("freenect_set_depth_mode() failed");
         }
         freenect_set_depth_callback(freenect1_device, kinect1_depth_callback);
         if (freenect_start_depth(freenect1_device) != 0) {
            throw std::runtime_error("freenect_start_depth() failed");
         }
      }

      if (color || ir) {
         auto resolution = FREENECT_RESOLUTION_MEDIUM;
         // TODO: maybe give a choice for color stream resolution?
         // TODO: check if the high resolution IR stream ever works properly
         auto video_mode = color ? FREENECT_VIDEO_RGB : FREENECT_VIDEO_IR_10BIT;
         auto frame_mode = freenect_find_video_mode(resolution, video_mode);
         if (freenect_set_video_mode(freenect1_device, frame_mode) != 0) {
            throw std::runtime_error("freenect_set_video_mode() failed");
         }
         video_buffer_mine = new uint8_t[frame_mode.bytes];
         video_buffer_freenect1 = new uint8_t[frame_mode.bytes];
         freenect_set_video_callback(freenect1_device, kinect1_video_callback);
         if (freenect_set_video_buffer(freenect1_device, video_buffer_freenect1) != 0) {
            throw std::runtime_error("freenect_set_video_buffer() failed");
         }
         if (freenect_start_video(freenect1_device) != 0) {
            throw std::runtime_error("freenect_start_video() failed");
         }
      }

      color_running = color;
      depth_running = depth;
      ir_running = ir;

      kinect1_run_event_loop.test_and_set();
      kinect1_event_thread = new std::thread(&KinectDevice::kinect1_process_events, this);
   } else if (which_kinect == 2) {
      if (int(depth) + int(ir) == 1) {
         throw std::invalid_argument("Kinect v2 can't stream only one of (depth, IR)");
      }

      stop_streams();

      if (color) {
         kinect2_color_listener = new Kinect2ColorListener(this);
         freenect2_device->setColorFrameListener(kinect2_color_listener);
      }
      if (depth && ir) {
         kinect2_depth_and_ir_listener = new Kinect2DepthAndIrListener(this);
         freenect2_device->setIrAndDepthFrameListener(kinect2_depth_and_ir_listener);
      }

      if (!freenect2_device->startStreams(color, depth && ir)) {
         throw std::runtime_error("freenect2_device->startStreams() failed");
      }
      depth_running = depth;
      color_running = color;
      ir_running = ir;
   }
}

void KinectDevice::stop_streams() {
   if (which_kinect == 1) {
      kinect1_run_event_loop.clear();
      if (kinect1_event_thread != nullptr) {
         kinect1_event_thread->join();
      }
      kinect1_run_event_loop.clear();
      if (depth_running) {
         if (freenect_stop_depth(freenect1_device) != 0) {
            throw std::runtime_error("freenect_stop_depth() failed");
         }
      }
      if (color_running || ir_running) {
         if (freenect_stop_video(freenect1_device) != 0) {
            throw std::runtime_error("freenect_stop_video() failed");
         }
      }
   } else if (which_kinect == 2) {
      if (depth_running || color_running || ir_running) {
         if (!freenect2_device->stop()) {
            throw std::runtime_error("freenect2_device->stop() failed");
         }
      }
      delete kinect2_depth_and_ir_listener;
      delete kinect2_color_listener;
      kinect2_depth_and_ir_listener = nullptr;
      kinect2_color_listener = nullptr;
   }
   depth_running = false;
   color_running = false;
   ir_running = false;
}

void KinectDevice::kinect1_process_events() {
   while (freenect_process_events(freenect1_context) == 0) {
      if (!kinect1_run_event_loop.test_and_set()) {
         break;
      }
   }
}

void KinectDevice::kinect1_depth_callback(freenect_device *device, void *depth_void, uint32_t timestamp) {
   auto kinect_device = static_cast<KinectDevice *>(freenect_get_user(device));
   auto frame_mode = freenect_get_current_depth_mode(device);
   auto width = static_cast<size_t>(frame_mode.width);
   auto height = static_cast<size_t>(frame_mode.height);
   auto pixels = new Matrix<float>(height, width);
   for (size_t i = 0; i < frame_mode.height; ++i) {
      for (size_t j = 0; j < frame_mode.width; ++j) {
         (*pixels)[i][j] = float(static_cast<uint16_t *>(depth_void)[frame_mode.width * i + j]);
      }
   }
   Picture picture;
   picture.depth_frame = new Picture::DepthOrIrFrame(pixels, true);
   auto frame_handler_thread = std::thread(&KinectDevice::frame_handler, kinect_device, picture);
   frame_handler_thread.detach();
}

void KinectDevice::kinect1_video_callback(freenect_device *device, void *buffer, uint32_t timestamp) {
   auto kinect_device = static_cast<KinectDevice *>(freenect_get_user(device));

   if (buffer != kinect_device->video_buffer_freenect1) {
      throw std::runtime_error("An error occured with Kinect's video buffer.");
   }
   kinect_device->video_buffer_freenect1 = kinect_device->video_buffer_mine;
   if (freenect_set_video_buffer(device, kinect_device->video_buffer_freenect1) != 0) {
      throw std::runtime_error("freenect_set_video_buffer() failed");
   }
   kinect_device->video_buffer_mine = buffer;

   auto frame_mode = freenect_get_current_video_mode(device);
   auto width = static_cast<size_t>(frame_mode.width), height = static_cast<size_t>(frame_mode.height);
   Picture picture;
   if (frame_mode.video_format == FREENECT_VIDEO_RGB) {
      auto pixels = new Matrix<Picture::ColorFrame::ColorPixel>(height, width);
      memcpy(pixels->data(), buffer, static_cast<size_t>(frame_mode.bytes));
      picture.color_frame = new Picture::ColorFrame(pixels);
   } else if (frame_mode.video_format == FREENECT_VIDEO_IR_10BIT) {
      auto pixels = new Matrix<float>(height, width);
      for (size_t i = 0; i < frame_mode.height; ++i) {
         for (size_t j = 0; j < frame_mode.width; ++j) {
            (*pixels)[i][j] = float(reinterpret_cast<uint16_t *>(buffer)[frame_mode.width * i + j]);
         }
      }
      picture.ir_frame = new Picture::DepthOrIrFrame(pixels, false);
   } else {
      std::cerr << "kinect1_video_callback() received an unexcepted video format, skipping frame\n";
      return;
   }
   auto frame_handler_thread = std::thread(&KinectDevice::frame_handler, kinect_device, picture);
   frame_handler_thread.detach();
}

KinectDevice::Kinect2DepthAndIrListener::Kinect2DepthAndIrListener(KinectDevice *kinect_device)
      : kinect_device(kinect_device) {}

bool KinectDevice::Kinect2DepthAndIrListener::onNewFrame(libfreenect2::Frame::Type type, libfreenect2::Frame *frame) {
   size_t bytes = frame->width * frame->height * sizeof(float);
   auto pixels = new Matrix<float>(frame->height, frame->width);
   memcpy(pixels->data(), frame->data, bytes);
   Picture picture;
   if (type == libfreenect2::Frame::Type::Depth) {
      picture.depth_frame = new Picture::DepthOrIrFrame(pixels, true);
      picture.depth_frame->freenect2_frame = frame;
   } else if (type == libfreenect2::Frame::Type::Ir) {
      picture.ir_frame = new Picture::DepthOrIrFrame(pixels, false);
      picture.ir_frame->freenect2_frame = frame;
   } else {
      std::cerr << "Kinect2DepthAndIrListener::onNewFrame() received an unexcepted video format.\n";
      return false;
   }
   kinect_device->frame_handler(picture);
   return true;
}

KinectDevice::Kinect2ColorListener::Kinect2ColorListener(KinectDevice *kinect_device) : kinect_device(kinect_device) {}

bool KinectDevice::Kinect2ColorListener::onNewFrame(libfreenect2::Frame::Type type, libfreenect2::Frame *frame) {
   if (type != libfreenect2::Frame::Type::Color || frame->format != libfreenect2::Frame::BGRX) {
      std::cerr << "Kinect2ColorListener::onNewFrame received an unexcepted video format.\n";
      return false;
   }
   auto pixels = new Matrix<Picture::ColorFrame::ColorPixel>(frame->height, frame->width);
   auto data = static_cast<uint8_t *>(frame->data);

   // Convert BGRX to BGR.
   for (size_t i = 0; i < frame->height; ++i) {
      for (size_t j = 0; j < frame->width; ++j) {
         size_t pixel_index = 4 * (i * frame->width + j);
         (*pixels)[i][j].blue = data[pixel_index];
         (*pixels)[i][j].green = data[pixel_index + 1];
         (*pixels)[i][j].red = data[pixel_index + 2];
      }
   }

   Picture picture;
   picture.color_frame = new Picture::ColorFrame(pixels);
   kinect_device->frame_handler(picture);
   return false;
}

#endif
