#include "imageFilteringCpu.h"
#include "imageFilteringGpu.cuh"
#include "utility.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    cv::Mat src = cv::imread("graf1.png", cv::IMREAD_GRAYSCALE);
    if(src.empty()){
        std::cerr << "Failed to open image file." << std::endl;
        return -1; 
    }

    cv::Mat dst(src.size(), src.type(), cv::Scalar(0));
    const int kernel_size = 7;
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
    const int border_size = (kernel_size-1)/2;

    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;

    // Naive Implementation
    start = cv::getTickCount();
    imageFilteringCpu(src, dst, kernel, border_size);
    end = cv::getTickCount();
    std::cout << "Naive: " << ((end - start) * f) << " ms." << std::endl;

    cv::cuda::GpuMat d_src(src);
    cv::cuda::GpuMat d_dst(dst.size(), dst.type(), cv::Scalar(0));
    cv::cuda::GpuMat d_kernel(kernel);

    // CUDA Implementation
    start = cv::getTickCount();
    launchImageFilteringGpu(d_src, d_dst, d_kernel, border_size);
    end = cv::getTickCount();
    std::cout << "CUDA: " << ((end - start) * f) << " ms." << std::endl;
    std::cout << std::endl;

    // Verification
    verify(dst, d_dst);

    return 0;
}
