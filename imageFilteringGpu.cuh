#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

void launchImageFilteringGpu(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& kernel);
