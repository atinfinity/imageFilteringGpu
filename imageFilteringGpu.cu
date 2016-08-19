#include "imageFilteringGpu.cuh"

#include <opencv2/cudev.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void imageFilteringGpu
(
    const cv::cudev::PtrStepSz<uchar> src,
    cv::cudev::PtrStepSz<uchar> dst,
    const cv::cudev::PtrStepSz<float> kernel
)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int border_size = (kernel.rows-1)/2;

    if((y >= border_size) && y < (dst.rows-border_size)){
        if((x >= border_size) && (x < (dst.cols-border_size))){
            double sum = 0.0;
            for(int yy = 0; yy < kernel.rows; yy++){
                for(int xx = 0; xx < kernel.cols; xx++){
                    sum += (kernel.ptr(yy)[xx] * src.ptr(y+yy-border_size)[x+xx-border_size]);
                }
            }
            dst.ptr(y)[x] = sum;
        }
    }
}

void launchImageFilteringGpu
(
    cv::cuda::GpuMat& src,
    cv::cuda::GpuMat& dst,
    cv::cuda::GpuMat& kernel
)
{
    cv::cudev::PtrStepSz<uchar> pSrc =
        cv::cudev::PtrStepSz<uchar>(src.rows, src.cols * src.channels(), src.ptr<uchar>(), src.step);

    cv::cudev::PtrStepSz<uchar> pDst =
        cv::cudev::PtrStepSz<uchar>(dst.rows, dst.cols * dst.channels(), dst.ptr<uchar>(), dst.step);

    cv::cudev::PtrStepSz<float> pKernel =
        cv::cudev::PtrStepSz<float>(kernel.rows, kernel.cols * kernel.channels(), kernel.ptr<float>(), kernel.step);

    const dim3 block(64, 2);
    const dim3 grid(cv::cudev::divUp(dst.cols, block.x), cv::cudev::divUp(dst.rows, block.y));

    imageFilteringGpu<<<grid, block>>>(pSrc, pDst, pKernel);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}
