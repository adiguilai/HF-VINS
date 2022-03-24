#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Extractor
{
public:
    torch::jit::script::Module model;
    Extractor(const std::string &model_path);
};

class SuperPointExtractor : Extractor
{
public:
    SuperPointExtractor(const std::string &model_path) : Extractor(model_path) {}
    virtual void operator()(const cv::Mat &image, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);
};

class NetVLADExtractor : Extractor
{
public:
    NetVLADExtractor(const std::string &model_path) : Extractor(model_path) {}
    virtual void operator()(const cv::Mat &image, cv::Mat &desc);
};