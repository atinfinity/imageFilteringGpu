#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

const cv::Size sz1080p = cv::Size(1920, 1080);
const cv::Size sz2160p = cv::Size(3840, 2160);
const cv::Size sz4320p = cv::Size(7680, 4320);

void verify(const cv::InputArray img1, const cv::InputArray img2);
