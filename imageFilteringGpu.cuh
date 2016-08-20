#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

double launchImageFilteringGpu(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& kernel, const int border_size, const int loop_num);

// use __ldg
double launchImageFilteringGpu_ldg(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& kernel, const int border_size, const int loop_num);

// use texture
double launchImageFilteringGpu_tex(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& kernel, const int border_size, const int loop_num);
