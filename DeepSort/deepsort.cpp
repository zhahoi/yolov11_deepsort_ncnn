#include "deepsort.h"
#include <iostream>

DeepSort::DeepSort()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    feature_extractor.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    feature_extractor.opt = ncnn::Option();

#if NCNN_VULKAN
    feature_extractor.opt.use_vulkan_compute = use_gpu;
#endif

    feature_extractor.opt.num_threads = ncnn::get_big_cpu_count();
    feature_extractor.opt.blob_allocator = &blob_pool_allocator;
    feature_extractor.opt.workspace_allocator = &workspace_pool_allocator;

    feature_extractor.load_param(PARAM_PATH);
    feature_extractor.load_model(BIN_PATH);
}

DeepSort::~DeepSort() {
    feature_extractor.clear();
}

bool DeepSort::getRectsFeature(const cv::Mat& img, DETECTIONS& d) {
    std::vector<cv::Mat> mats;
    for (DETECTION_ROW& dbox : d) 
    {
        cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
            int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
        rc.width = rc.height * 0.5;
        rc.x = (rc.x >= 0 ? rc.x : 0);
        rc.y = (rc.y >= 0 ? rc.y : 0);
        rc.width = (rc.x + rc.width <= img.cols ? rc.width : (img.cols - rc.x));
        rc.height = (rc.y + rc.height <= img.rows ? rc.height : (img.rows - rc.y));

        cv::Mat mattmp = img(rc).clone();
        cv::resize(mattmp, mattmp, cv::Size(64, 128));
        mats.push_back(mattmp);
    }

    int count = mats.size();

    for (int i = 0; i < count; i++)
    {
        ncnn::Mat in_net = ncnn::Mat::from_pixels(mats[i].data, ncnn::Mat::PIXEL_BGR2RGB, 64, 128);
        in_net.substract_mean_normalize(mean, norm);

        ncnn::Mat out_net;
        ncnn::Extractor ex = feature_extractor.create_extractor();
        ex.set_light_mode(true);

        // if (toUseGPU) {  // Ïû³ýÌáÊ¾
        //    ex.set_vulkan_compute(toUseGPU);
        // }

        ex.input("in0", in_net);
        ex.extract("out0", out_net);

        cv::Mat tmp(out_net.h, out_net.w, CV_32FC1, (void*)(const float*)out_net.channel(0));
        const float* tp = tmp.ptr<float>(0);
        for (int j = 0; j < feature_dim; j++)
        {
            d[i].feature[j] = tp[j];
        }
    }

    return true;
}
