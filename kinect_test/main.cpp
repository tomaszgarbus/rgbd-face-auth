#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <libfreenect/libfreenect.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

freenect_context *ctx;
freenect_device *device;

bool depthPhotoTaken = false, videoPhotoTaken = false;

int videoWidth, videoHeight, videoColors;
freenect_resolution freenectVideoResolution;
freenect_video_format freenectVideoFormat;

void *videoBufferFreenect, *videoBufferMine;

void depthCallback(freenect_device *device, void *depthVoid, uint32_t timestamp) {
  if (depthPhotoTaken)
    return;

  auto depth = (uint16_t *) depthVoid;
  FILE *photoFile = fopen("../photo_kinect1_depth.txt", "w");
  for (int i = 0; i < 480; ++i) {
    for (int j = 0; j < 640; ++j) {
      fprintf(photoFile, "%u\t", depth[640 * i + j]);
    }
    fprintf(photoFile, "\n");
  }
  fclose(photoFile);

  depthPhotoTaken = true;
  printf("Depth photo taken.\n");
}

void videoCallback(freenect_device *device, void *buffer, uint32_t timestamp) {
  if (videoPhotoTaken)
    return;

  assert(buffer == videoBufferFreenect);
  videoBufferFreenect = videoBufferMine;
  freenect_set_video_buffer(device, videoBufferFreenect);
  videoBufferMine = buffer;

  if (freenectVideoFormat == FREENECT_VIDEO_IR_10BIT) {
    FILE *photoFile = fopen("../photo_kinect1_ir.txt", "w");
    for (int i = 0; i < videoHeight; ++i) {
      for (int j = 0; j < videoWidth; ++j) {
        fprintf(photoFile,
                "%u\t",
                ((uint16_t *)videoBufferMine)[videoWidth * i + j]);
      }
      fprintf(photoFile, "\n");
    }
    fclose(photoFile);
  } else if (freenectVideoFormat == FREENECT_VIDEO_RGB) {
    cv::Mat image(cv::Size(videoWidth, videoHeight), CV_8UC3, videoBufferMine);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite("../photo_kinect1_rgb.png", image);
  }

  videoPhotoTaken = true;
  printf("Video photo taken.\n");
}

int main(int argc, char *argv[]) {
  // Option 0 (default): 640x480 depth + 640x480 RBG, aligned together
  // Option 1: 640x480 depth + 1280x1024 RGB
  // Option 2: 640x480 depth + 640x488 IR
  int option = 1;
  if (argc == 2) {
    option = (int) strtol(argv[1], nullptr, 10);
  } else if (argc > 2) {
    printf("Too many arguments.\n");
    return 1;
  }

  switch (option) {
    case 0:
      videoWidth = 640; videoHeight = 480; videoColors = 3;
      freenectVideoResolution = FREENECT_RESOLUTION_MEDIUM;
      freenectVideoFormat = FREENECT_VIDEO_RGB;
      break;
    case 1:
      videoWidth = 1280; videoHeight = 1024; videoColors = 3;
      freenectVideoResolution = FREENECT_RESOLUTION_HIGH;
      freenectVideoFormat = FREENECT_VIDEO_RGB;
      break;
    case 2:
      videoWidth = 640; videoHeight = 488; videoColors = 1;
      freenectVideoResolution = FREENECT_RESOLUTION_MEDIUM;
      freenectVideoFormat = FREENECT_VIDEO_IR_10BIT;
      break;
    default:
      printf("Invalid option.\n");
      return 1;
  }

  if (freenect_init(&ctx, nullptr) != 0) {
    fprintf(stderr, "freenect_init() failed.\n");
    return 1;
  }

  int numberOfDevices = freenect_num_devices(ctx);
  printf("Found %d device%s.\n",
         numberOfDevices,
         numberOfDevices == 1 ? "" : "s");

  if (numberOfDevices <= 0) {
    freenect_shutdown(ctx);
    return 1;
  }

  if(freenect_open_device(ctx, &device, 0) != 0) {
    fprintf(stderr, "freenect_open_device() failed.\n");
    freenect_shutdown(ctx);
    return 1;
  }

  freenect_set_led(device, LED_RED);
  freenect_set_tilt_degs(device, 15.0);

  freenect_set_depth_callback(device, depthCallback);
  freenect_set_depth_mode(device, freenect_find_depth_mode(
      FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM));

  if (freenectVideoFormat == FREENECT_VIDEO_RGB) {
    videoBufferFreenect = (uint8_t *) malloc(
        videoWidth * videoHeight * videoColors * sizeof(uint8_t));
    videoBufferMine = (uint8_t *) malloc(
        videoWidth * videoHeight * videoColors * sizeof(uint8_t));
  } else if (freenectVideoFormat == FREENECT_VIDEO_IR_10BIT) {
    videoBufferFreenect = (uint16_t *) malloc(
        videoWidth * videoHeight * videoColors * sizeof(uint16_t));
    videoBufferMine = (uint16_t *) malloc(
        videoWidth * videoHeight * videoColors * sizeof(uint16_t));
  }
  freenect_set_video_callback(device, videoCallback);
  freenect_set_video_mode(device, freenect_find_video_mode(
      freenectVideoResolution, freenectVideoFormat));
  freenect_set_video_buffer(device, videoBufferFreenect);

//  freenect_start_depth(device);
  freenect_start_video(device);

  while (!depthPhotoTaken
         && !videoPhotoTaken
         && freenect_process_events(ctx) >= 0) {}

//  freenect_stop_depth(device);
  freenect_stop_video(device);

  freenect_set_led(device, LED_GREEN);
  freenect_close_device(device);
  freenect_shutdown(ctx);
  return 0;
}
