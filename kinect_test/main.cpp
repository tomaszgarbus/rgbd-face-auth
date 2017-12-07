#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <libfreenect/libfreenect.h>
#include <cassert>

freenect_context *ctx;
freenect_device *device;

bool depthPhotoTaken = false;
bool irPhotoTaken = false;

uint16_t *videoBufferFreenect, *videoBufferMine;

void depthCallback(freenect_device *device, void *depthVoid, uint32_t timestamp) {
  if (depthPhotoTaken)
    return;

  uint16_t *depth = (uint16_t *) depthVoid;
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
  if (irPhotoTaken)
    return;

  assert(buffer == videoBufferFreenect);
  videoBufferFreenect = videoBufferMine;
  freenect_set_video_buffer(device, videoBufferFreenect);
  videoBufferMine = (uint16_t *) buffer;

  FILE *photoFile = fopen("../photo_kinect1_ir.txt", "w");
  for (int i = 0; i < 488; ++i) {
    for (int j = 0; j < 640; ++j) {
      fprintf(photoFile, "%u\t", videoBufferMine[640 * i + j]);
    }
    fprintf(photoFile, "\n");
  }
  fclose(photoFile);

  irPhotoTaken = true;
  printf("IR photo taken.\n");
}

int main() {
  if (freenect_init(&ctx, NULL) != 0) {
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

  videoBufferFreenect = (uint16_t *) malloc(640 * 488 * sizeof(uint16_t));
  videoBufferMine = (uint16_t *) malloc(640 * 488 * sizeof(uint16_t));
  freenect_set_video_callback(device, videoCallback);
  freenect_set_video_mode(device, freenect_find_video_mode(
      FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_IR_10BIT));
  freenect_set_video_buffer(device, videoBufferFreenect);

  freenect_start_depth(device);
  freenect_start_video(device);

  while (!depthPhotoTaken && !irPhotoTaken && freenect_process_events(ctx) >= 0) {}

  freenect_stop_depth(device);
  freenect_stop_video(device);

  freenect_set_led(device, LED_GREEN);
  freenect_close_device(device);
  freenect_shutdown(ctx);
  return 0;
}
