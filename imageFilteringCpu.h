#pragma once

#include <opencv2/core.hpp>

double launchImageFilteringCpu(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel, const int border_size, const int loop_num);
