#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

void launchImageFilteringGpu(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& kernel, const int border_size);

// use __ldg
void launchImageFilteringGpu_ldg(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& kernel, const int border_size);

// use texture
void launchImageFilteringGpu_tex(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& kernel, const int border_size);
