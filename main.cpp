#include "imageFilteringCpu.h"
#include "imageFilteringGpu.cuh"
#include "utility.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    const int loop_num = 5;

    cv::Mat src = cv::imread("graf1.png", cv::IMREAD_GRAYSCALE);
    if(src.empty()){
        std::cerr << "Failed to open image file." << std::endl;
        return -1; 
    }

    cv::Mat dst(src.size(), src.type(), cv::Scalar(0));
    const int kernel_size = 7;
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
    const int border_size = (kernel_size-1)/2;

    // Naive Implementation
    double time = launchImageFilteringCpu(src, dst, kernel, border_size, loop_num);
    std::cout << "Naive: " << time << " ms." << std::endl;

    cv::cuda::GpuMat d_src(src);
    cv::cuda::GpuMat d_dst(dst.size(), dst.type(), cv::Scalar(0));
    cv::cuda::GpuMat d_dst_ldg(dst.size(), dst.type(), cv::Scalar(0));
    cv::cuda::GpuMat d_dst_tex(dst.size(), dst.type(), cv::Scalar(0));
    cv::cuda::GpuMat d_kernel(kernel);

    // CUDA Implementation
    time = launchImageFilteringGpu(d_src, d_dst, d_kernel, border_size, loop_num);
    std::cout << "CUDA: " << time << " ms." << std::endl;

    // CUDA Implementation(use __ldg)
    time = launchImageFilteringGpu_ldg(d_src, d_dst_ldg, d_kernel, border_size, loop_num);
    std::cout << "CUDA(ldg): " << time << " ms." << std::endl;

    // CUDA Implementation(use texture)
    time = launchImageFilteringGpu_tex(d_src, d_dst_tex, d_kernel, border_size, loop_num);
    std::cout << "CUDA(texture): " << time << " ms." << std::endl;

    std::cout << std::endl;

    // Verification
    verify(dst, d_dst);
    verify(dst, d_dst_ldg);
    verify(dst, d_dst_tex);

    return 0;
}
