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
const std::string SMILE_CASCADE_FILE = "haarcascades/haarcascade_smile.xml";

const cv::Scalar FACE_ELLIPSE_COLOR(255,0,255);
const cv::Scalar EYES_ELLIPSE_COLOR(255,0,0);
const cv::Scalar SMILE_ELLIPSE_COLOR(0,0,255);

cv::CascadeClassifier faceCascade, eyesCascade, smilesCascade;

/* Draws a red thick line from left-upper corner to the middle of |img|. */
void testDraw(cv::Mat &img) {
    cv::Point pt1 = cv::Point(0, 0);
    cv::Point pt2 = cv::Point(img.cols / 2, img.rows / 2);
    cv::Scalar color = cv::Scalar(0, 0, 255);
    cv::line(img, pt1, pt2, color, /*thickness=*/4);
}

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

    printf("Detected %u faces\n", faces.size());
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Scalar color(0,0,255);
        drawEllipseAroundRect(img, faces[i], FACE_ELLIPSE_COLOR);

        cv::Mat faceRect = img_gray(faces[i]);

        std::vector<cv::Rect> eyes;
        eyesCascade.detectMultiScale(faceRect, eyes);
        printf("Detected %u eyes\n", eyes.size());
        for (size_t j = 0; j < eyes.size(); j++) {
            eyes[j].x += faces[i].x;
            eyes[j].y += faces[i].y;
            drawEllipseAroundRect(img, eyes[j], EYES_ELLIPSE_COLOR);
        }

        std::vector<cv::Rect> smiles;
        smilesCascade.detectMultiScale(faceRect, smiles, /*scale=*/4);
        printf("Detected %u smiles\n", smiles.size());
        for (size_t j = 0; j < smiles.size(); j++) {
            smiles[j].x += faces[i].x;
            smiles[j].y += faces[i].y;
            drawEllipseAroundRect(img, smiles[j], SMILE_ELLIPSE_COLOR);
        }
    }
    cv::imshow(WINDOW_NAME, img);
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
    if (!faceCascade.load(FACE_CASCADE_FILE)) {
        fprintf(stderr, "Error loading face cascade\n");
        return false;
    }
    if (!eyesCascade.load(EYES_CASCADE_FILE)) {
        fprintf(stderr, "Error loading eyes cascade\n");
        return false;
    }
    if (!smilesCascade.load(SMILE_CASCADE_FILE)) {
        fprintf(stderr, "Error loading smiles cascade\n");
        return false;
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