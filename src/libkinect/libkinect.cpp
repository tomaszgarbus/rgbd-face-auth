#include "libkinect.hpp"

#include <iostream>
#include <libfreenect/libfreenect.h>

KinectDevice::KinectDevice() {
  if (freenect_init(&freenectContext, nullptr) != 0) {
    fprintf(stderr, "freenect_init() failed.\n");
    exit(1);
  }

  int kinect1devices;
  kinect1devices = freenect_num_devices(freenectContext);

  if (kinect1devices <= 0) {
    freenect_shutdown(freenectContext);
    printf("No Kinect devices found.\n");
    exit(1);
  }
}

void KinectDevice::startStreams(bool depth, bool video, bool ir) {

}
