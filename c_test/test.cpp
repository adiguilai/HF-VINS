#include <iostream>
#include <chrono>

// #include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for cvtColor
#include <opencv2/features2d/features2d.hpp>

#include "hloc.h"

using namespace std;
using namespace chrono;

int main()
{
    SuperPointExtractor SuperPoint_1024("../../models/SuperPoint_1024.pt");
    NetVLADExtractor NetVLAD("../../models/NetVLAD.pt");

    string img_path = "../../night.jpg";
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR), image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    // to float32 Mat
    image_gray.convertTo(image_gray, CV_32FC1, 1.f / 255.f, 0);
    image.convertTo(image, CV_32FC3, 1.f / 255.f, 0);

    std::vector<cv::KeyPoint> kpts;
    std::vector<float> scrs;
    cv::Mat local_desc, global_desc;

    auto t1 = std::chrono::high_resolution_clock::now();

    // put image_gray in the model
    // use kpts, scrs and desc to receive the output
    SuperPoint_1024(image_gray, kpts, scrs, local_desc);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    std::cout << "extract " << kpts.size() << " keypoints, took " << fp_ms.count() << " ms, " << endl;
    std::cout << "scores" << scrs << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    NetVLAD(image, global_desc);
    t2 = std::chrono::high_resolution_clock::now();
    fp_ms = t2 - t1;
    std::cout << "extract global_desc, took " << fp_ms.count() << " ms, " << endl;

    // draw keypoints
    cv::Mat outimg1;
    image.convertTo(image, CV_8UC3, 255.f / 1.f, 0); // for drawKeypoints
    cv::drawKeypoints(image, kpts, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::resize(outimg1, outimg1, cv::Size(outimg1.cols / 4, outimg1.rows / 4), 0, 0, cv::INTER_LINEAR);
    cv::imshow("SuperPoints", outimg1);
    cv::waitKey(0);

    // print loacl_desc shape
    cout << "local_desc shape: " << local_desc.cols << 'x' << local_desc.rows << endl;
    cout << "global_desc shape: " << global_desc.cols << 'x' << global_desc.rows << endl;

    return 0;
}