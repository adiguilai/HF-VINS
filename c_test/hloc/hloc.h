#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "../utility/tic_toc.h"

#define SuperPointPath "../../models/SuperPoint_1024.pt"
#define NetVLADPath "../../models/NetVLAD.pt"
#define SuperGluePath "../../models/SuperGlue_outdoor.pt"

class SuperPoint {
public:
    static SuperPoint& Get();
    static void Extract(
            const cv::Mat &image,
            std::vector<cv::Point2f> &kpts,
            std::vector<float> &scrs,
            cv::Mat &desc
            );
private:
    torch::jit::script::Module model;
    SuperPoint();
    void IExtract(
            const cv::Mat &image,
            std::vector<cv::Point2f> &kpts,
            std::vector<float> &scrs,
            cv::Mat &desc
            );

};

class NetVLAD {
public:
    static NetVLAD& Get();
    static void Extract(
            const cv::Mat &image,
            cv::Mat &desc
    );
private:
    torch::jit::script::Module model;
    NetVLAD();
    void IExtract(
            const cv::Mat &image,
            cv::Mat &desc
    );

};

class SuperGlue {
public:
    static SuperGlue& Get();
    static void Match(
            std::vector<cv::Point2f> &kpts0,
            std::vector<float> &scrs0,
            cv::Mat &desc0,
            int height0, int width0,
            std::vector<cv::Point2f> &kpts1,
            std::vector<float> &scrs1,
            cv::Mat &desc1,
            int height1, int width1,
            std::vector<int> &match_index,
            std::vector<float> &match_score
    );
private:
    torch::jit::script::Module model;
    SuperGlue();
    void IMatch(
            std::vector<cv::Point2f> &kpts0,
            std::vector<float> &scrs0,
            cv::Mat &desc0,
            int height0, int width0,
            std::vector<cv::Point2f> &kpts1,
            std::vector<float> &scrs1,
            cv::Mat &desc1,
            int height1, int width1,
            std::vector<int> &match_index,
            std::vector<float> &match_score
    );
};
