#include <iostream>
#include <chrono>

// #include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for cvtColor
#include <opencv2/features2d/features2d.hpp>

#include "hloc/hloc.h"

using namespace std;
using namespace chrono;

void readImage(const string path, cv::Mat &image, cv::Mat &image_gray) {
    image = cv::imread(path, cv::IMREAD_COLOR);
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
}

int main() {
    SuperPointExtractor SuperPoint_1024("../../models/SuperPoint_1024.pt");
    NetVLADExtractor NetVLAD("../../models/NetVLAD.pt");
    SuperGlueMatcher SuperGlue("../../models/SuperGlue_outdoor.pt");

    cv::Mat image_1, image_gray_1, image_2, image_gray_2;
    std::vector<cv::Point2f> kpts_1, kpts_2;
    std::vector<float> scrs_1, scrs_2;
    cv::Mat local_desc_1, local_desc_2, global_desc_1, global_desc_2;

    readImage("../../night.jpg", image_1, image_gray_1);
    readImage("../../day.jpg", image_2, image_gray_2);

    SuperPoint_1024(image_gray_1, kpts_1, scrs_1, local_desc_1);
    SuperPoint_1024(image_gray_2, kpts_2, scrs_2, local_desc_2);
    NetVLAD(image_1, global_desc_1);
    NetVLAD(image_2, global_desc_2);

    std::vector<int> match_index;
    std::vector<float> match_score;
    auto t1 = std::chrono::high_resolution_clock::now();

    SuperGlue(kpts_1, scrs_1, local_desc_1, image_gray_1.rows, image_gray_1.cols,
              kpts_2, scrs_2, local_desc_2, image_gray_2.rows, image_gray_2.cols,
              match_index, match_score
    );
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    std::cout << "sim " << global_desc_1.dot(global_desc_2) << std::endl;
    std::cout << "match " << match_index.size() << " pairs, took " << fp_ms.count() << " ms, " << endl;


    vector<cv::KeyPoint> kpts1, kpts2;
    vector<cv::DMatch> match;
    for (auto & i : kpts_1) {
        kpts1.emplace_back(i, 1.f);
    }
    for (auto & i : kpts_2) {
        kpts2.emplace_back(i, 1.f);
    }
    for (int i = 0; i < match_index.size(); i++){
        if (match_index[i] > -1)
            match.emplace_back(i, match_index[i], 1-match_score[i]);
    }

    cv::Mat out_image;
    cv::drawMatches(image_1, kpts1, image_2, kpts2, match, out_image);
    cv::resize(out_image, out_image, cv::Size(), 0.3, 0.3);
    cv::imshow("matches", out_image);
    cv::waitKey(0);

    return 0;
}