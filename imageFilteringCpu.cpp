#include "imageFilteringCpu.h"

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
