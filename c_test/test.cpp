#include <iostream>
#include <chrono>

#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for cvtColor
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace chrono;

class SuperPointExtractor
{
private:
    torch::jit::script::Module model;

public:
    SuperPointExtractor(const std::string &model_path);
    virtual void operator()(const cv::Mat &image, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);
};

SuperPointExtractor::SuperPointExtractor(const std::string &model_path)
{
    // load model
    model = torch::jit::load(model_path);
    cout << "finish loading the module" << endl;
}

void SuperPointExtractor::operator()(const cv::Mat &image, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc)
{
    cv::Mat image_gray = image.clone();
    // resize image if size > max size (1024)
    // record the original size
    float scale = 1;
    int img_width_ori = image_gray.cols;
    int img_height_ori = image_gray.rows;
    if (max(img_width_ori, img_height_ori) > 1024) {
        scale = 1024.f / max(img_width_ori, img_height_ori);
        int img_width_new = img_width_ori * scale;
        int img_height_new = img_height_ori * scale;
        cv::resize(image_gray, image_gray, cv::Size(img_width_new, img_height_new), 0, 0, cv::INTER_LINEAR);
    }
    // convert cv::Mat to torch::Tensor [batch x C x H x W]
    torch::Tensor img_tensor = torch::from_blob(image_gray.data, {1, 1, image_gray.rows, image_gray.cols}).to(torch::kCUDA);

    // put image into model
    vector<torch::jit::IValue> torch_inputs;
    torch::jit::IValue torch_outputs;
    torch_inputs.emplace_back(img_tensor);
    torch_outputs = model.forward(torch_inputs);// it defult set no grad
    auto outputs_tuple = torch_outputs.toTuple();
    auto keypoints = outputs_tuple->elements()[0].toTensorVector()[0].to(torch::kCPU);   // N x 2
    auto descriptors = outputs_tuple->elements()[2].toTensorVector()[0].to(torch::kCPU); // 256 x N

    // reduce position of keypoints
    keypoints = (keypoints / scale).round();
    //convert torch::Tensor to std::vector<cv::KeyPoint>
    for (int i = 0; i < keypoints.size(0); i++) {
        kpts.push_back(cv::KeyPoint(keypoints[i][0].item<float>(), keypoints[i][1].item<float>(), 8));
    }
    //convert torch::Tensor to cv::Mat???
    cv::Mat mat_desc(descriptors.size(0), descriptors.size(1), CV_32FC1);
    desc = mat_desc;
}

// TODO: class NetVLADExtractor

int main()
{
    SuperPointExtractor SuperPoint_1024("../../models/SuperPoint_1024.pt");

    string img_path = "../../night.jpg";
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR), image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    // to float32 Mat
    image_gray.convertTo(image_gray, CV_32FC1, 1.f / 255.f, 0);

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;

    auto t1 = std::chrono::high_resolution_clock::now();

    // put image_gray in the model
    // use kpts and desc to receive the output
    SuperPoint_1024(image_gray, kpts, desc);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    std::cout << "extract " << kpts.size() << " keypoints, took " << fp_ms.count() << " ms, " <<endl;

    // draw keypoints
    cv::Mat outimg1;
    cv::drawKeypoints(image, kpts, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::resize(outimg1, outimg1, cv::Size(outimg1.cols/4, outimg1.rows/4), 0, 0, cv::INTER_LINEAR);
    cv::imshow("SuperPoints", outimg1);
    cv::waitKey(0);

    // print desc shape
    cout << "desc shape: "<< desc.cols << 'x' << desc.rows << endl;

    return 0;
}