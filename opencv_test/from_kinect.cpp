#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/logger.h>

// Using iostream, as openCV prints matrices nicely.
#include <iostream>
#include <cstdio>

const char *WINDOW_NAME = "Display Image";
const std::string FACE_CASCADE_FILE = "haarcascades/haarcascade_frontalface_alt.xml";

const cv::Scalar FACE_ELLIPSE_COLOR(255,0,255);

cv::CascadeClassifier faceCascade;

const unsigned int FRAME_WIDTH = 1920;
const unsigned int FRAME_HEIGHT = 1080;

cv::Mat img = cv::Mat(FRAME_HEIGHT/4, FRAME_WIDTH/4, CV_8UC4);
cv::Mat img_buf = cv::Mat(FRAME_HEIGHT ,FRAME_WIDTH, CV_8UC4);

libfreenect2::Freenect2 freenect2;
libfreenect2::Freenect2Device *dev = 0;
libfreenect2::PacketPipeline *pipeline = 0;

struct BGRX_pixel {
    int R, G, B, X;
};

bool testDetectFace(cv::Mat &img);

class RGBCameraListener: public libfreenect2::FrameListener {
public:
    bool onNewFrame(libfreenect2::Frame::Type type, libfreenect2::Frame *frame) override {
        assert(sizeof(BGRX_pixel) == 4*sizeof(int));
        assert(frame->format == libfreenect2::Frame::BGRX);
        assert(frame->height == FRAME_HEIGHT);
        assert(frame->width == FRAME_WIDTH);
        img_buf.data = frame->data;
        cv::resize(img_buf, img, cv::Size(FRAME_WIDTH/4, FRAME_HEIGHT/4));
        testDetectFace(img);
        cv::imshow(WINDOW_NAME, img);
        cv::waitKey(30);
        return false;
    }
};

void drawEllipseAroundRect(cv::Mat &img, const cv::Rect &rect, const cv::Scalar &color) {
    cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
    cv::ellipse(img, center, cv::Size(rect.width / 2, rect.height / 2), 0, 0, 360, color,
            /*thickness=*/4, cv::LINE_8);
}

bool testDetectFace(cv::Mat &img) {
    std::vector<cv::Rect> faces;
    cv::Mat img_gray;

    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(img_gray, img_gray);

    faceCascade.detectMultiScale(img_gray, faces);

    printf("Detected %lu faces\n", faces.size());
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Scalar color(0,0,255);
        drawEllipseAroundRect(img, faces[i], FACE_ELLIPSE_COLOR);
    }
    cv::imshow(WINDOW_NAME, img);
    return true;
}

bool loadCascades() {
    if (!faceCascade.load(FACE_CASCADE_FILE)) {
        fprintf(stderr, "Error loading face cascade\n");
        return false;
    }
    return true;
}

bool initializeKinectCamera() {
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
        return false;
    }
    RGBCameraListener *listener = new RGBCameraListener();
    dev->setColorFrameListener(listener);
    if (!dev->startStreams(/*rbg=*/true, /*depth=*/false)) {
        fprintf(stderr, "Error starting streams");
        return 0;
    }
    return true;
}

int main() {
    if (!loadCascades()) {
        return 1;
    }
    if (!initializeKinectCamera()) {
        return 1;
    }

    //testDraw(img);
    /*if (!testDetectFace(img)) {
        return 1;
    }


    cv::waitKey(0);*/
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

    while (true);

    dev->stop();
    dev->close();
    return 0;
}