//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#include "ultraface.h"
#include "faceembedding.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>

using namespace std;

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

    while (true) {
        cv::Mat frame;
        cap >> frame;

        auto start = chrono::steady_clock::now();
        vector<FaceInfo> face_info;
        ultraface.detect(frame, face_info);

        // convert from cv::Mat images into form that dlib can deal with 
        dlib::cv_image<dlib::bgr_pixel> dimg(frame);

        for (auto face : face_info) {
            std::vector<float> emb;
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            dlib::rectangle det((long)face.x1, (long)face.y1, (long)face.x2, (long)face.y2);
            dlib::full_object_detection shape = pose_model(dimg, det);
            cv::Rect roi;
            roi.x = det.left();
            roi.y = det.top();
            roi.width = det.width();
            roi.height = det.height();

            //cout << "Number parts = " << shape.num_parts() << endl;

            cv::Mat cropped = frame(roi);
            int s_width = cropped.rows;
            int s_height = cropped.cols;
 //           cv::warpAffine(cropped, M, (s_width, s_height), cv::INTER_CUBIC);
            cv::Mat output;
            cv::resize(cropped, output, cv::Size(112, 112));
            faceEmbedding.getEmbedding(output, emb);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }

        //auto end = chrono::steady_clock::now();
        //chrono::duration<double> elapsed = end - start;
        //cout << "all time: " << elapsed.count() << " s" << endl;

        cv::imshow("UltraFace", frame);

        char keying = char(cv::waitKey(1) & 0xFF);
        if (keying == 'q' || keying == 'Q')
            break;
    }

    return 0;
}