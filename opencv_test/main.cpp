#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"

// Using iostream, as openCV prints matrices nicely.
#include <iostream>
#include <cstdio>

const char *WINDOW_NAME = "Display Image";
const std::string FACE_CASCADE_FILE = "haarcascades/haarcascade_frontalface_alt.xml";
const std::string EYES_CASCADE_FILE = "haarcascades/haarcascade_eye_tree_eyeglasses.xml";

/* Draws a red thick line from left-upper corner to the middle of |img|. */
void testDraw(cv::Mat &img) {
    cv::Point pt1 = cv::Point(0, 0);
    cv::Point pt2 = cv::Point(img.cols / 2, img.rows / 2);
    cv::Scalar color = cv::Scalar(0, 0, 255);
    cv::line(img, pt1, pt2, color, /*thickness=*/4);
}

bool testDetectFace(cv::Mat &img) {
    cv::CascadeClassifier faceCascade, eyesCascade;
    if (!faceCascade.load(FACE_CASCADE_FILE)) {
        fprintf(stderr, "Error loading face cascade\n");
        return false;
    }
    if (!eyesCascade.load(EYES_CASCADE_FILE)) {
        fprintf(stderr, "Error loading eyes cascade\n");
        return false;
    }

    std::vector<cv::Rect> faces;
    cv::Mat img_gray;

    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(img_gray, img_gray);

    faceCascade.detectMultiScale(img_gray, faces);

    printf("Detected %d faces\n", faces.size());
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        cv::Scalar color(0, 0, 255);
        cv::ellipse(img, center, cv::Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, color,
                    /*thickness=*/4, cv::LINE_8);
    }
    return true;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cout << "usage: opencv_test <Image_Path>" << std::endl;
    }
    cv::Mat img;
    img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (!img.data) {
        fprintf(stderr, "No image data\n");
        return -1;
    }
    //testDraw(img);
    if (!testDetectFace(img)) {
        return -1;
    }

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::imshow(WINDOW_NAME, img);
    cv::waitKey(0);
    return 0;
}