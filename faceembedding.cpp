//  Created by HungNV on 2020/12/23.
//  Copyright © 2020 HungNV. All rights reserved.

#include "faceembedding.h"
#include <chrono>

FaceEmbedding::FaceEmbedding(const std::string& mnn_path) {
    mMNNTnterpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = 4; // num_thread; -> use ultility to get number cores
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
    config.backendConfig = &backendConfig;

    mMNNSession = mMNNTnterpreter->createSession(config);

    mInputTensor = mMNNTnterpreter->getSessionInput(mMNNSession, nullptr);
}

FaceEmbedding::~FaceEmbedding() {
    mMNNTnterpreter->releaseModel();
    mMNNTnterpreter->releaseSession(mMNNSession);
    mMNNTnterpreter = nullptr;
    mMNNSession = nullptr;
    mInputTensor = nullptr;
}

int FaceEmbedding::getEmbedding(cv::Mat& img, std::vector<float>& embedding_list) {
    if (!mMNNTnterpreter || !mMNNSession) {
        return -1;
    }

    int image_h = img.rows;
    int image_w = img.cols;
    int in_h = 112;
    int in_w = 112;
    cv::Mat image;

    if (image_h == 112 && image_w == 112)
        image = img;
    else 
        cv::resize(img, image, cv::Size(in_h, in_w));

    mMNNTnterpreter->resizeTensor(mInputTensor, { 1, 3, in_h, in_w });
    mMNNTnterpreter->resizeSession(mMNNSession);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB));
    pretreat->convert(image.data, in_w, in_h, image.step[0], mInputTensor);

    //auto start = std::chrono::steady_clock::now();


    // run network
    mMNNTnterpreter->runSession(mMNNSession);

    // get output data
    std::string ebd_layer = "fc1";
    MNN::Tensor* tensor_embedding = mMNNTnterpreter->getSessionOutput(mMNNSession, ebd_layer.c_str());

    MNN::Tensor tensor_embedding_host(tensor_embedding, tensor_embedding->getDimensionType());
    tensor_embedding->copyToHostTensor(&tensor_embedding_host);

    //auto end = std::chrono::steady_clock::now();
    //std::chrono::duration<double> elapsed = end - start;
    //std::cout << "embedding inference time:" << elapsed.count() << " s" << std::endl;

    embedding_list.clear();
    for (int i = 0; i < 512; i++)
        embedding_list.push_back(tensor_embedding_host.host<float>()[i]);

    return 0;
}
