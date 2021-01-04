//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#include "ultraface.h"
#include "faceembedding.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>

using namespace std;

float cosineSimiliarity(const std::vector<float>& A, const std::vector<float>& B) {
    if (A.size() != B.size()) {
        cout << "vector A has no same size with vector B" << endl;
        return 0;
    }

    float num = 0.0f, d_A = 0.0f, d_B = 0.0f;
#pragma omp parallel for reduction(+:num, d_A, d_B)
    for (int idx = 0; idx < A.size(); idx++) {
        num += A[idx] * B[idx];
        d_A += A[idx] * A[idx];
        d_B += B[idx] * B[idx];
    }

    if (d_A == 0.0f || d_B == 0.0f) {
        cout << "Either vector A or vector B is zero" << endl;
        return 0;
    }

    return num / sqrt(d_A * d_B);
}

// this version for 5 points landmarks of R right eye, L right eye, R left eye, L left eye, nose
cv::Mat align_face(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks) {
    //calculate the center position
    int cx = img.cols / 2;
    int cy = img.rows / 2;

    //calculate the angle
    int dx = landmarks[0].x - landmarks[3].x;
    int dy = landmarks[0].y - landmarks[3].y;
    double degree = 180 * ((atan2(dy, dx)) / CV_PI);

    //calculate the rotation matrix
    cv::Point2f center(cx, cy);
    cv::Mat M = cv::getRotationMatrix2D(center, degree, 1.0);
    cv::Rect bbox = cv::RotatedRect(cv::Point2f(), img.size(), degree).boundingRect2f();
    M.at <double>(0, 2) += (bbox.width / 2.0 - center.x);
    M.at <double>(1, 2) += (bbox.height / 2.0 - center.y);

    //Align
    cv::Mat result;
    cv::warpAffine(img, result, M, bbox.size());
    //std::cout << "origin wxh = " << img.cols << "x" << img.rows << " -- result wxh = " << result.cols << "x" << result.rows << std::endl;
    return result;
}

int main(int argc, char** argv) {
 
    string mnn_path = "./model/RFB-320-quant-ADMM-32.mnn";
    UltraFace ultraface(mnn_path, 320, 240, 4, 0.65); // config model input
    string ebd_path = "./model/mnn_resnet100_20201223.mnn";
    FaceEmbedding faceEmbedding(ebd_path);
    dlib::shape_predictor pose_model;
    dlib::deserialize("./model/shape_predictor_5_face_landmarks.dat") >> pose_model;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }

    std::vector<float> stored;
    bool bFirst = true;

    while (true) {
        cv::Mat frame;
        cap >> frame;

        auto start = chrono::steady_clock::now();
        vector<FaceInfo> face_info;
        ultraface.detect(frame, face_info);

        // convert from cv::Mat images into form that dlib can deal with 
        dlib::cv_image<dlib::bgr_pixel> dimg(frame);

        for (auto face = face_info.begin(); face != face_info.end(); ++face) {
            std::vector<float> emb;
            cv::Point pt1(face->x1, face->y1);
            cv::Point pt2(face->x2, face->y2);
            dlib::rectangle det((long)face->x1, (long)face->y1, (long)face->x2, (long)face->y2);
            dlib::full_object_detection shape = pose_model(dimg, det);
            cv::Rect roi;
            roi.x = det.left();
            roi.y = det.top();
            roi.width = det.width();
            roi.height = det.height();
#if 1
            int square_w = (roi.width > roi.height ? roi.width : roi.height);
            roi.x -= (square_w - roi.width) >> 1;
            roi.y -= (square_w - roi.height) >> 1;

            if (roi.x < 0) {
                roi.x = 0;
            }

            if (roi.y < 0) {
                roi.y = 0;
            }

            if (roi.x + square_w > frame.cols)
                square_w = frame.cols - roi.x;

            if (roi.y + square_w > frame.rows)
                square_w = frame.rows - roi.y;

            roi.width = roi.height = square_w;
#else
            if (roi.x + roi.width > frame.cols) {
                roi.width = frame.cols - roi.x;
            }

            if (roi.y + roi.height > frame.rows) {
                roi.height = frame.rows - roi.y;
            }
#endif
            std::vector<cv::Point2f> landmarks;
            for (unsigned long i = 0; i < shape.num_parts(); i++) {
                cv::Point2f pt;
                pt.x = shape.part(i).x() - roi.x;
                pt.y = shape.part(i).y() - roi.y;
                landmarks.push_back(pt);
            }

            cv::Mat cropped = frame(roi), aligned;// , output;
            aligned = align_face(cropped, landmarks);
            cv::imshow("aligned", aligned);
            faceEmbedding.getEmbedding(aligned, emb);
            if (bFirst) {
                stored = emb;
                bFirst = false;
            }
            else {
                float dist = cosineSimiliarity(stored, emb);
                cout << "Cosine distance = " << dist << endl;
            }
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("UltraFace", frame);

        char keying = char(cv::waitKey(1) & 0xFF);
        if (keying == 'q' || keying == 'Q')
            break;
    }

    return 0;
}