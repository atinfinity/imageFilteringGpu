#include "imageFilteringCpu.h"
#include <opencv2/imgproc.hpp>

void imageFilteringCpu
(
    const cv::Mat& src, 
    cv::Mat& dst, 
    const cv::Mat& kernel, 
    const int border_size
)
{
    for(int y = border_size; y < (dst.rows - border_size); y++){
        uchar* pdst = dst.ptr<uchar>(y);
        for(int x = border_size; x < (dst.cols - border_size); x++){
            double sum = 0.0;
            for(int yy = 0; yy < kernel.rows; yy++){
                for(int xx = 0; xx < kernel.cols; xx++){
                    sum += (kernel.ptr<float>(yy)[xx] * src.ptr<uchar>(y+yy-border_size)[x+xx-border_size]);
                }
            }
            pdst[x] = sum;
        }
    }
}

double launchImageFilteringCpu
(
    const cv::Mat& src, 
    cv::Mat& dst, 
    const cv::Mat& kernel, 
    const int border_size, 
    const int loop_num
)
{
    double f = 1000.0f/cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for(int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        imageFilteringCpu(src, dst, kernel, border_size);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}

double launchImageFilteringCV
(
    const cv::Mat& src,
    cv::Mat& dst,
    const cv::Mat& kernel,
    const int border_size,
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        cv::filter2D(src, dst, -1, kernel);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}
