#include "hloc.h"

hloc::hloc(const std::string &model_path)
{
    // load model
    model = torch::jit::load(model_path);
    std::cout << "finish load " << model_path << std::endl;
}

void SuperPointExtractor::operator()(const cv::Mat &image, std::vector<cv::KeyPoint> &kpts, std::vector<float> &scrs, cv::Mat &desc)
{
    cv::Mat image_gray = image.clone();
    // resize image if size > max size (1024)
    // record the original size
    float scale = 1;
    int img_width_ori = image_gray.cols;
    int img_height_ori = image_gray.rows;
    if (std::max(img_width_ori, img_height_ori) > 1024)
    {
        scale = 1024.f / std::max(img_width_ori, img_height_ori);
        int img_width_new = img_width_ori * scale;
        int img_height_new = img_height_ori * scale;
        cv::resize(image_gray, image_gray, cv::Size(img_width_new, img_height_new), 0, 0, cv::INTER_LINEAR);
    }
    // convert cv::Mat to torch::Tensor [batch x C x H x W]
    torch::Tensor img_tensor = torch::from_blob(image_gray.data, {1, 1, image_gray.rows, image_gray.cols}).to(torch::kCUDA);

    // put image into model
    std::vector<torch::jit::IValue> torch_inputs;
    torch::jit::IValue torch_outputs;
    torch_inputs.emplace_back(img_tensor);
    torch_outputs = model.forward(torch_inputs); // it defult set no grad
    auto outputs_tuple = torch_outputs.toTuple();
    auto keypoints = outputs_tuple->elements()[0].toTensorVector()[0].to(torch::kCPU);   // N x 2
    auto scores = outputs_tuple->elements()[1].toTensorVector()[0].to(torch::kCPU);
    auto descriptors = outputs_tuple->elements()[2].toTensorVector()[0].to(torch::kCPU); // 256 x N

    // reduce position of keypoints
    keypoints = (keypoints / scale).round();
    // convert torch::Tensor to std::vector<cv::KeyPoint>
    for (int i = 0; i < keypoints.size(0); i++)
    {
        kpts.push_back(cv::KeyPoint(keypoints[i][0].item<float>(), keypoints[i][1].item<float>(), 8));
    }
    // convert torch::Tensor to std::vector<std::float>???
    for (int i = 0; i < scores.size(0); i++)
    {
        scrs.push_back(scores[i].item<float>());
    }
    // convert torch::Tensor to cv::Mat???
    cv::Mat mat_desc(descriptors.size(0), descriptors.size(1), CV_32FC1);
    desc = mat_desc;
}

void NetVLADExtractor::operator()(const cv::Mat &image, cv::Mat &desc)
{
    cv::Mat image_gray = image.clone();
    // resize image if size > max size (1024)
    // record the original size
    float scale = 1;
    int img_width_ori = image_gray.cols;
    int img_height_ori = image_gray.rows;
    if (std::max(img_width_ori, img_height_ori) > 1024)
    {
        scale = 1024.f / std::max(img_width_ori, img_height_ori);
        int img_width_new = img_width_ori * scale;
        int img_height_new = img_height_ori * scale;
        cv::resize(image_gray, image_gray, cv::Size(img_width_new, img_height_new), 0, 0, cv::INTER_LINEAR);
    }
    // convert cv::Mat to torch::Tensor [batch x C x H x W]
    torch::Tensor img_tensor = torch::from_blob(image_gray.data, {1, 3, image_gray.rows, image_gray.cols}).to(torch::kCUDA);

    // put image into model
    std::vector<torch::jit::IValue> torch_inputs;
    torch::jit::IValue torch_outputs;
    torch_inputs.emplace_back(img_tensor);
    torch_outputs = model.forward(torch_inputs);                 // it defult set no grad
    auto descriptors = torch_outputs.toTensor().to(torch::kCPU); // 256 x N

    // convert torch::Tensor to cv::Mat???
    cv::Mat mat_desc(descriptors.size(0), descriptors.size(1), CV_32FC1);
    desc = mat_desc;
}
