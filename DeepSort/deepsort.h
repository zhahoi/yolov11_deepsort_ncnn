#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "model.h"
#include "dataType.h"

#include "net.h"
#include "cpu.h"
#include "layer.h"
#include <benchmark.h>

typedef unsigned char uint8;

#define PARAM_PATH "C:/CPlusPlus/YOLOv11-DeepSORT/weights/deepsort_reid_opt_fp16.param"
#define BIN_PATH "C:/CPlusPlus/YOLOv11-DeepSORT/weights/deepsort_reid_opt_fp16.bin"

class DeepSort 
{
public:
    DeepSort();
    ~DeepSort();

    bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);
    // virtual bool predict(cv::Mat& frame) { }

private:
    ncnn::Net feature_extractor;

    bool use_gpu = true;
    const int feature_dim = 512;
    const float norm[3] = { 0.229, 0.224, 0.225 };
    const float mean[3] = { 0.485, 0.456, 0.406 };

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};
