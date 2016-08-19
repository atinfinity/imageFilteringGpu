#pragma once

#include <opencv2/core.hpp>

void imageFilteringCpu(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel);
