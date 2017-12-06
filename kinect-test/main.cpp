#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <libfreenect/libfreenect.h>

freenect_context *ctx;
freenect_device *device;

bool photoTaken = false;

void depthCallback(freenect_device *device, void *depthVoid, uint32_t timestamp) {
  uint16_t *depth = (uint16_t *) depthVoid;
  FILE *photoFile = fopen("photo.txt", "w");
  for (int i = 0; i < 640; ++i) {
    for (int j = 0; j < 480; ++j) {
      fprintf(photoFile, "%u\t", depth[480 * i + j]);
    }
    fprintf(photoFile, "\n");
  }
  fclose(photoFile);
  photoTaken = true;
  printf("Photo taken.\n");
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
  freenect_set_depth_mode(
      device,
      freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM,
                               FREENECT_DEPTH_11BIT));

  freenect_start_depth(device);
  while (!photoTaken && freenect_process_events(ctx) >= 0) {}

  freenect_stop_depth(device);
  freenect_set_led(device, LED_GREEN);
  freenect_close_device(device);
  freenect_shutdown(ctx);
  return 0;
}
