#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class hloc
{
public:
    torch::jit::script::Module model;
    hloc(const std::string &model_path);
};

class SuperPointExtractor : hloc
{
public:
    SuperPointExtractor(const std::string &model_path) : hloc(model_path) {}
    virtual void operator()(const cv::Mat &image, std::vector<cv::KeyPoint> &kpts, std::vector<float> &scrs, cv::Mat &desc);
};

class NetVLADExtractor : hloc
{
public:
    NetVLADExtractor(const std::string &model_path) : hloc(model_path) {}
    virtual void operator()(const cv::Mat &image, cv::Mat &desc);
};

// TODO:
// class SuperGlueMatcher : hloc
// {
// public:
//     SuperGlueMatcher(const std::string &model_path) : hloc(model_path) {}
//     virtual void operator()(const cv::Mat &image, cv::Mat &desc);
// };
// What should matcher return??