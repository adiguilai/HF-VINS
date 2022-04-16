# HF-VINS
using **[Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)** in **[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)**

## **TODO List**

- [x] Refactoring hloc with TensorScript
- [x] Read and run models in C++
- [x] class SuperPointExtractor
- [x] class NetVLADExtractor
- [x] class SuperGlueMatcher?
- [x] class UltraPoint
- [x] class Keyframe Database: It should include two functions 1) to get the similarity of a new frame to the global descriptors of all frames in the database 2) to add the current keyframe to the database.
- [ ] Rewrite VINS-Mono/pose_graph/keyframe
- [ ] Rewrite VINS-Mono/pose_graph/*

## **visualization**

night: SuperPoint keypoints + SuperPoint descriptors

day: goodFeaturesToTrack keypoints +  + SuperPoint descriptors (I call it UltraPoint, because the model just extracts the descriptors, it's not that super)

using SuperGlue match those keypoints

![screenshot](screenshot.png)

