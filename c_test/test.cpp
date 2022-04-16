#include <iostream>
#include "utility/tic_toc.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for cvtColor
#include <opencv2/features2d/features2d.hpp>

#include "hloc/hloc.h"
#include "hloc/hlocDatabase.h"

using namespace std;

void readImage(const string& path, cv::Mat &image, cv::Mat &image_gray) {
    image = cv::imread(path, cv::IMREAD_COLOR);
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
}

void test_db() {
    vector<cv::Mat> images;
    hlocDatabase db;
    cv::Mat image, image_gray, global_desc;
    vector<Result> ret;
    for(int i = 0; i < 10; i++) {
        string path = "../test_images/" + to_string(i) + ".jpg";
        readImage(path, image, image_gray);
        TicToc t_gd;
        NetVLAD::Extract(image, global_desc);
        printf("%d image took %f ms\n", i, t_gd.toc());
        db.add(global_desc);
    }
    db.query(global_desc, ret, 0, 10);
    for (auto &i: ret)
        cout << i << endl;
}

int main() {
//    SuperPointExtractor SuperPoint_1024("../../models/SuperPoint_1024.pt");
//    NetVLADExtractor NetVLAD("../../models/NetVLAD.pt");
//    SuperGlueMatcher SuperGlue("../../models/SuperGlue_outdoor.pt");
//
//    cv::Mat image_1, image_gray_1, image_2, image_gray_2;
//    std::vector<cv::Point2f> kpts_1, kpts_2;
//    std::vector<float> scrs_1, scrs_2;
//    cv::Mat local_desc_1, local_desc_2, global_desc_1, global_desc_2;
//
//    readImage("../../night.jpg", image_1, image_gray_1);
//    readImage("../../day.jpg", image_2, image_gray_2);
//
//    TicToc t_ld;
//    SuperPoint::Extract(image_gray_1, kpts_1, scrs_1, local_desc_1);
//    printf("Extracting SuperPoint and local descriptor took %f ms\n", t_ld.toc());
//
//    TicToc t_ud;
//    cv::goodFeaturesToTrack(image_gray_2, kpts_2, 500, 0.01, 10);
//    UltraPoint::Extract(image_gray_2, kpts_2, scrs_2, local_desc_2);
//    printf("Just extracting local descriptor took %f ms\n", t_ld.toc());
//
//    TicToc t_gd;
//    NetVLAD::Extract(image_1, global_desc_1);
//    printf("Extracting global descriptor took %f ms\n", t_gd.toc());
//    NetVLAD::Extract(image_2, global_desc_2);
//
//    std::vector<int> match_index;
//    std::vector<float> match_score;
//    TicToc t_match;
//    SuperGlue::Match(kpts_1, scrs_1, local_desc_1, image_gray_1.rows, image_gray_1.cols,
//              kpts_2, scrs_2, local_desc_2, image_gray_2.rows, image_gray_2.cols,
//              match_index, match_score
//    );
//    printf("Matching took %f ms\n", t_match.toc());
//    printf("cos sim: %f\n", global_desc_1.dot(global_desc_2));
//
//    vector<cv::KeyPoint> kpts1, kpts2;
//    vector<cv::DMatch> match;
//    for (auto & i : kpts_1) {
//        kpts1.emplace_back(i, 1.f);
//    }
//    for (auto & i : kpts_2) {
//        kpts2.emplace_back(i, 1.f);
//    }
//    for (int i = 0; i < match_index.size(); i++){
//        if (match_index[i] > -1)
//            match.emplace_back(i, match_index[i], 1-match_score[i]);
//    }
//
//    cv::Mat out_image;
//    cv::drawMatches(image_1, kpts1, image_2, kpts2, match, out_image);
//    cv::resize(out_image, out_image, cv::Size(), 0.5, 0.5);
//    cv::imshow("matches", out_image);
//    cv::waitKey(0);
    test_db();
    return 0;
}