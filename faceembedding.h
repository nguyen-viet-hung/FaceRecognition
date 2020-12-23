//  Created by HungNV on 2020/12/23.
//  Copyright © 2020 HungNV. All rights reserved.

#ifndef FaceEmbedding_hpp
#define FaceEmbedding_hpp

#pragma once

#include <MNN/Interpreter.hpp>

#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class FaceEmbedding
{
protected:
	std::shared_ptr<MNN::Interpreter> mMNNTnterpreter;
	MNN::Session* mMNNSession = nullptr;
	MNN::Tensor* mInputTensor = nullptr;
public:
	FaceEmbedding(const std::string& mnn_path);
	virtual ~FaceEmbedding();

	int getEmbedding(cv::Mat& img, std::vector<float>& embedding_list);
};

#endif//FaceEmbedding_hpp