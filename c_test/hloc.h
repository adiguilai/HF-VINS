#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class hloc {
public:
    torch::jit::script::Module model;

    hloc(const std::string &model_path);
};

class SuperPointExtractor : hloc {
public:
    explicit SuperPointExtractor(const std::string &model_path) : hloc(model_path) {}

    virtual void operator()(
            const cv::Mat &image,
            std::vector<cv::Point2f> &kpts,
            std::vector<float> &scrs,
            cv::Mat &desc
    );
};

class NetVLADExtractor : hloc {
public:
    explicit NetVLADExtractor(const std::string &model_path) : hloc(model_path) {}

    virtual void operator()(
            const cv::Mat &image,
            cv::Mat &desc
    );
};

class SuperGlueMatcher : hloc {
public:
    explicit SuperGlueMatcher(const std::string &model_path) : hloc(model_path) {}

    virtual void operator()(
            std::vector<cv::Point2f> &kpts0,
            std::vector<float> &scrs0,
            cv::Mat &desc0,
            int height0, int width0,
            std::vector<cv::Point2f> &kpts1,
            std::vector<float> &scrs1,
            cv::Mat &desc1,
            int height1, int width1,
            std::vector<int> &match_index_01,
            std::vector<int> &match_index_10
    );
};
// What should matcher return??