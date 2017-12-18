#ifndef LIBKINECT_H
#define LIBKINECT_H

#include <libfreenect/libfreenect.h>

class KinectDevice {
 public:
  KinectDevice();
  void startStreams(bool depth, bool video, bool ir);

 private:
  int whichKinect;  // 1 or 2
  freenect_context *freenectContext;
  freenect_device *freenectDevice;
};

#endif