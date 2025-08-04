#ifndef YOLO_H
#define YOLO_H
#define NOMINMAX

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <net.h>

#define PARAM_PATH "C:/CPlusPlus/YOLOv11-DeepSORT/weights/yolo11n.ncnn.param"
#define BIN_PATH "C:/CPlusPlus/YOLOv11-DeepSORT/weights/yolo11n.ncnn.bin"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

class Yolo
{
public:
    Yolo();

    ~Yolo();

    int detect(const cv::Mat& rgb, std::vector<Object>& objects);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

private:
    ncnn::Net yolo;

    const int target_size = 640;
    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.5f;
    const bool use_gpu = true;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // NANODET_H