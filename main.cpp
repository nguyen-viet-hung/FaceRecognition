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

//def align_face_by_landmarks(cv_img, dst, dst_w, dst_h) :
//    if dst_w == 96 and dst_h == 112 :
//        src = np.array([
//            [30.2946, 51.6963],
//                [65.5318, 51.5014],
//                [48.0252, 71.7366],
//                [33.5493, 92.3655],
//                [62.7299, 92.2041]], dtype = np.float32)
//        elif dst_w == 112 and dst_h == 112:
//            src = np.array([
//                [38.2946, 51.6963],
//                    [73.5318, 51.5014],
//                    [56.0252, 71.7366],
//                    [41.5493, 92.3655],
//                    [70.7299, 92.2041]], dtype = np.float32)
//                elif dst_w == 150 and dst_h == 150:
//                src = np.array([
//                    [51.287415, 69.23612],
//                        [98.48009, 68.97509],
//                        [75.03375, 96.075806],
//                        [55.646385, 123.7038],
//                        [94.72754, 123.48763]], dtype = np.float32)
//                    elif dst_w == 160 and dst_h == 160:
//                    src = np.array([
//                        [54.706573, 73.85186],
//                            [105.045425, 73.573425],
//                            [80.036, 102.48086],
//                            [59.356144, 131.95071],
//                            [101.04271, 131.72014]], dtype = np.float32)
//                        elif dst_w == 224 and dst_h == 224:
//                        src = np.array([
//                            [76.589195, 103.3926],
//                                [147.0636, 103.0028],
//                                [112.0504, 143.4732],
//                                [83.098595, 184.731],
//                                [141.4598, 184.4082]], dtype = np.float32)
//    else:
//return None
//tform = trans.SimilarityTransform()
//tform.estimate(dst, src)
//M = tform.params[0:2, : ]
//
//if M is None :
//img = cv_img
//
//#use center crop
//det = np.zeros(4, dtype = np.int32)
//det[0] = int(img.shape[1] * 0.0625)
//det[1] = int(img.shape[0] * 0.0625)
//det[2] = img.shape[1] - det[0]
//det[3] = img.shape[0] - det[1]
//
//margin = 44
//bb = np.zeros(4, dtype = np.int32)
//bb[0] = np.maximum(det[0] - margin / 2, 0)
//bb[1] = np.maximum(det[1] - margin / 2, 0)
//bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
//bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
//ret = img[bb[1]:bb[3], bb[0] : bb[2], : ]
//ret = cv2.resize(ret, (dst_w, dst_h))
//return ret
//
//face_img = cv2.warpAffine(cv_img, M, (dst_w, dst_h), borderValue = 0.0)
//return face_img

namespace FacePreprocess {

    cv::Mat meanAxis0(const cv::Mat& src) {
        int num = src.rows;
        int dim = src.cols;

        // x1 y1
        // x2 y2

        cv::Mat output(1, dim, CV_32F);
        for (int i = 0; i < dim; i++) {
            float sum = 0;
            for (int j = 0; j < num; j++) {
                sum += src.at<float>(j, i);
            }
            output.at<float>(0, i) = sum / num;
        }

        return output;
    }

    cv::Mat elementwiseMinus(const cv::Mat& A, const cv::Mat& B) {
        cv::Mat output(A.rows, A.cols, A.type());

        assert(B.cols == A.cols);
        if (B.cols == A.cols) {
            for (int i = 0; i < A.rows; i++) {
                for (int j = 0; j < B.cols; j++) {
                    output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
                }
            }
        }
        return output;
    }


    cv::Mat varAxis0(const cv::Mat& src) {
        cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
        cv::multiply(temp_, temp_, temp_);
        return meanAxis0(temp_);
    }



    int MatrixRank(cv::Mat M) {
        cv::Mat w, u, vt;
        cv::SVD::compute(M, w, u, vt);
        cv::Mat1b nonZeroSingularValues = w > 0.0001;
        int rank = countNonZero(nonZeroSingularValues);
        return rank;
    }

    //    References
    //    ----------
    //    .. [1] "Least-squares estimation of transformation parameters between two
    //    point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    //
    //    """
    //
    //    Anthor:Jack Yu
    cv::Mat similarTransform(cv::Mat src, cv::Mat dst) {
        int num = src.rows;
        int dim = src.cols;
        cv::Mat src_mean = meanAxis0(src);
        cv::Mat dst_mean = meanAxis0(dst);
        cv::Mat src_demean = elementwiseMinus(src, src_mean);
        cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
        cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
        cv::Mat d(dim, 1, CV_32F);
        d.setTo(1.0f);
        if (cv::determinant(A) < 0) {
            d.at<float>(dim - 1, 0) = -1;

        }
        cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
        cv::Mat U, S, V;
        cv::SVD::compute(A, S, U, V);

        // the SVD function in opencv differ from scipy .


        int rank = MatrixRank(A);
        if (rank == 0) {
            assert(rank == 0);

        }
        else if (rank == dim - 1) {
            if (cv::determinant(U) * cv::determinant(V) > 0) {
                T.rowRange(0, dim).colRange(0, dim) = U * V;
            }
            else {
                //            s = d[dim - 1]
                //            d[dim - 1] = -1
                //            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
                //            d[dim - 1] = s
                int s = d.at<float>(dim - 1, 0) = -1;
                d.at<float>(dim - 1, 0) = -1;

                T.rowRange(0, dim).colRange(0, dim) = U * V;
                cv::Mat diag_ = cv::Mat::diag(d);
                cv::Mat twp = diag_ * V; //np.dot(np.diag(d), V.T)
                cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
                cv::Mat C = B.diag(0);
                T.rowRange(0, dim).colRange(0, dim) = U * twp;
                d.at<float>(dim - 1, 0) = s;
            }
        }
        else {
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V.t(); //np.dot(np.diag(d), V.T)
            cv::Mat res = U * twp; // U
            T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
        }

        cv::Mat var_ = varAxis0(src_demean);
        float val = cv::sum(var_).val[0];
        cv::Mat res;
        cv::multiply(d, S, res);
        float scale = 1.0 / val * cv::sum(res).val[0];
        T.rowRange(0, dim).colRange(0, dim) = -T.rowRange(0, dim).colRange(0, dim).t();
        cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
        cv::Mat  temp2 = src_mean.t(); //src_mean.T
        cv::Mat  temp3 = temp1 * temp2; // np.dot(T[:dim, :dim], src_mean.T)
        cv::Mat temp4 = scale * temp3;
        T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
        T.rowRange(0, dim).colRange(0, dim) *= scale;
        return T;
    }
}

// this version for 5 points landmarks of left eye, right eye, nose, left mouth conner, right mouth conner
cv::Mat align_face(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks, int width, int height) {
    const int ReferenceWidth = 112;
    const int ReferenceHeight = 112;

    int wex = 0;
    if (ReferenceWidth == 112)
        wex = 8;

    float curs[5][2] = {
        {landmarks[0].x, landmarks[0].y},
        {landmarks[1].x, landmarks[1].y},
        {landmarks[2].x, landmarks[2].y},
        {landmarks[3].x, landmarks[3].y},
        {landmarks[4].x, landmarks[4].y}
    };

    for (int i = 0; i < 5; i++) {
        cout << landmarks[i].x << " " << landmarks[i].y << endl;
    }

    cv::Mat src(5, 2, CV_32FC1, curs);
    //memcpy(src.data, curs, 2 * 5 * sizeof(float));

    float refs[5][2] = {
        { 30.29459953f + wex,  51.69630051f },
        { 65.53179932f + wex,  51.50139999f },
        { 48.02519989f + wex,  71.73660278f },
        { 33.54930115f + wex,  92.3655014f },
        { 62.72990036f + wex,  92.20410156f }
    };

    cv::Mat dst(5, 2, CV_32FC1, refs);
    //memcpy(dst.data, refs, 2 * 5 * sizeof(float));

    cv::Mat tfm = FacePreprocess::similarTransform(src, dst);
    //for (int y = 0; y < tfm.rows; y++) {
    //    for (int x = 0; x < tfm.cols; x++) {
    //        cout << tfm.at<float>(y, x) << "\t ";
    //    }
    //    cout << endl;
    //}
    cv::Mat M = cv::Mat(0, tfm.cols, CV_32FC1);

    //cout << "Transform matrix has rows = " << tfm.rows << " and cols = " << tfm.cols << endl;
    M.push_back(tfm.row(0));
    M.push_back(tfm.row(1));

    //for (int y = 0; y < M.rows; y++) {
    //    for (int x = 0; x < M.cols; x++) {
    //        cout << M.at<float>(y, x) << "\t ";
    //    }
    //    cout << endl;
    //}
    cv::Mat aligned_face;
    cv::warpAffine(img, aligned_face, M, cv::Size(ReferenceWidth, ReferenceHeight));
    return aligned_face;
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
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), degree, 1.0);
    cv::Point2f center(cx, cy);
    cv::Rect bbox = cv::RotatedRect(center, cv::Size2f(112.0f, 112.0f), degree).boundingRect();
    M.at <double>(0, 2) += (bbox.width / 2.0 - center.x);
    M.at <double>(1, 2) += (bbox.height / 2.0 - center.y);

    //Align
    cv::Mat result;
    cv::warpAffine(img, result, M, bbox.size());
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

            if (roi.x + roi.width > frame.cols)
                roi.width = frame.cols - roi.x;

            if (roi.y + roi.height > frame.rows)
                roi.height = frame.rows - roi.y;

            //cout << "Number parts = " << shape.num_parts() << endl;
            std::vector<cv::Point2f> landmarks;
            for (unsigned long i = 0; i < shape.num_parts(); i++) {
                cv::Point2f pt;
                pt.x = shape.part(i).x() - det.left();
                pt.y = shape.part(i).y() - det.top();
                landmarks.push_back(pt);
            }

            cv::Mat cropped = frame(roi), aligned;// , output;
            int s_width = cropped.cols;
            int s_height = cropped.rows;

            cv::imshow("Cropped Face", cropped);
            //aligned = align_face(cropped, landmarks, s_width, s_height);
            aligned = align_face(cropped, landmarks);
            cv::imshow("Aligned Face", aligned);
            //cv::resize(aligned, output, cv::Size(112, 112));
            faceEmbedding.getEmbedding(aligned, emb);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("UltraFace", frame);

        char keying = char(cv::waitKey(1) & 0xFF);
        if (keying == 'q' || keying == 'Q')
            break;
    }

    return 0;
}