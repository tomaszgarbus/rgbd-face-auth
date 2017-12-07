#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <cstring>

libfreenect2::Freenect2 freenect2;
libfreenect2::Freenect2Device *dev = 0;
libfreenect2::PacketPipeline *pipeline = 0;

bool irPhotoTaken;
bool depthPhotoTaken;

class IrAndDepthCameraListener: public libfreenect2::FrameListener {
public:
    bool onNewFrame(libfreenect2::Frame::Type type,
                    libfreenect2::Frame *frame) {
        float *data = (float*) frame->data;

        if (type == libfreenect2::Frame::Type::Ir && !irPhotoTaken) {
            FILE *photoFile = fopen("photo_ir.txt", "w");

            for (int i = 0; i < frame->height; i++){
                for (int j = 0; j < frame->width; j++) {
                    fprintf(photoFile, "%d\t ", int(data[i*frame->height + j]));
                }
            }
            fclose(photoFile);
            irPhotoTaken = true;
            printf("\033[01;32mIr photo taken\033[00;29m\n");
        }
        if (type == libfreenect2::Frame::Type::Depth && !depthPhotoTaken) {
            FILE *photoFile = fopen("photo_depth.txt", "w");

            for (int i = 0; i < frame->height; i++){
                for (int j = 0; j < frame->width; j++) {
                    fprintf(photoFile, "%d\t ", int(data[i*frame->height + j]));
                }
            }
            fclose(photoFile);
            depthPhotoTaken = true;
            printf("\033[01;32mDepth photo taken\033[00;29m\n");
        }
    }
};

int main() {
    int numberOfDevices;
    while (!(numberOfDevices = freenect2.enumerateDevices())) {
        printf("Waiting for device...\n");
    }

    printf("\033[01;32mFound %d device%s.%s\n",
           numberOfDevices,
           numberOfDevices == 1 ? "" : "s",
           "\033[00;29m");

    pipeline = new libfreenect2::CpuPacketPipeline();
    dev = freenect2.openDevice(0, pipeline);
    if (!dev) {
        fprintf(stderr, "Error opening a device\n");
        return 1;
    }

    IrAndDepthCameraListener *listener = new IrAndDepthCameraListener();
    dev->setIrAndDepthFrameListener(listener);

    if (!dev->startStreams(/*RGB=*/false, /*depth=*/true)) {
        fprintf(stderr, "Error starting streams");
        return 1;
    }

    while (!irPhotoTaken || !depthPhotoTaken);

    dev->stop();
    dev->close();
    return 0;
}