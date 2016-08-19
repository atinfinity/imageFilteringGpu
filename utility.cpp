#include "utility.h"
#include <iostream>

void verify
(
    const cv::InputArray img1, 
    const cv::InputArray img2
)
{
    cv::Mat h_img1, h_img2;
    if(img1.kind() == cv::_InputArray::CUDA_GPU_MAT){
        img1.getGpuMat().download(h_img1);
    }
    else{
        h_img1 = img1.getMat();
    }

    if(img2.kind() == cv::_InputArray::CUDA_GPU_MAT){
        img2.getGpuMat().download(h_img2);
    }
    else{
        h_img2 = img2.getMat();
    }

    cv::Mat diff(h_img1.size(), h_img1.type(), cv::Scalar(0));
    cv::absdiff(h_img1, h_img2, diff);
    std::cout << "[Verify] " <<
        ((cv::countNonZero(diff) == 0) ? "Passed" : "Failed") << "." << std::endl;
}
