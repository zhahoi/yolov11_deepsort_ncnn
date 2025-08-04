#include "YOLOv11/yolov11.h"
#include "DeepSort/deepsort.h"
#include "KalmanFilter/tracker.h"

#include <chrono>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

std::unique_ptr<Yolo> yolov11(new Yolo());
std::unique_ptr<DeepSort> deepSort(new DeepSort());
std::unique_ptr<tracker> id_tracker(new tracker());

// 定义队列大小，避免内存无限制增长
const int QUEUE_MAX_SIZE = 10;
std::queue<cv::Mat> frameQueue;
std::mutex mtx;
std::condition_variable cv_frame;
bool stopProcessing = false;

// 读取视频帧的函数
void readFrames(cv::VideoCapture& cap) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_frame.wait(lock, [] { return frameQueue.size() < QUEUE_MAX_SIZE || stopProcessing; });

        if (stopProcessing) {
            break;
        }

        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            stopProcessing = true;
            cv_frame.notify_all();
            break;
        }

        frameQueue.push(frame);
        cv_frame.notify_all();
    }
}

void get_detections(const cv::Rect_<float>& rect, float confidence, DETECTIONS& d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh << rect.x, rect.y, rect.width, rect.height;
    tmpRow.confidence = confidence;
    tmpRow.feature.setZero();  // 初始化为0，等待后续特征提取填充
    d.push_back(tmpRow);
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    std::ostringstream label_ss;
    label_ss << classId << ": " << std::fixed << std::setprecision(2) << conf;
    std::string label = label_ss.str();

    //Display the label at the top of the bounding box
    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = std::max(top, labelSize.height);

    cv::rectangle(frame,
        cv::Point(left, top - labelSize.height - baseLine),
        cv::Point(left + labelSize.width, top),
        cv::Scalar(255, 255, 255), cv::FILLED);

    cv::putText(frame, label, cv::Point(left, top - 4),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

void postprocess(cv::Mat& frame, const std::vector<Object>& outs, DETECTIONS& d)
{
    for (const Object& obj : outs)
    {
        if (obj.label != 0)
            continue;

        // 可选：可视化函数
        /*
        drawPred(obj.label, obj.prob,
            static_cast<int>(obj.rect.x),
            static_cast<int>(obj.rect.y),
            static_cast<int>(obj.rect.x + obj.rect.width),
            static_cast<int>(obj.rect.y + obj.rect.height),
            frame);
        */
        get_detections(obj.rect, obj.prob, d);
    }
}

// 处理视频帧的函数
void processFrames() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_frame.wait(lock, [] { return !frameQueue.empty() || stopProcessing; });

        if (stopProcessing && frameQueue.empty()) {
            break;
        }

        cv::Mat frame = frameQueue.front();
        frameQueue.pop();
        lock.unlock();
        cv_frame.notify_all();

        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();

        // 调用 detect 函数
        std::vector<Object> objects;

        objects.clear();
        yolov11->detect(frame, objects);

        DETECTIONS detections;
        postprocess(frame, objects, detections);

        if (!detections.empty() && deepSort->getRectsFeature(frame, detections))
        {
            id_tracker->predict();
            id_tracker->update(detections);

            std::vector<RESULT_DATA> result;
            for (Track& track : id_tracker->tracks) {
                if (!track.is_confirmed() || track.time_since_update > 1) continue;
                result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
            }

            for (const auto& r : result) {
                const auto& tlwh = r.second;
                cv::Rect rect(tlwh(0), tlwh(1), tlwh(2), tlwh(3));
                cv::rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);
                cv::putText(frame, std::to_string(r.first), cv::Point(rect.x, rect.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 1);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Processing time per frame: " << elapsed.count() << " ms" << std::endl;
     
        // 显示当前帧
        cv::imshow("YOLOv11-Deepsort-Video Inference", frame);

        // 按下 'q' 键退出
        if (cv::waitKey(1) == 'q') {
            stopProcessing = true;
            cv_frame.notify_all();
            break;
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " video <0 or video_path>" << std::endl;
        return -1;
    }

    std::string mode = argv[1];
    std::string path = argv[2];

    if (mode != "video") {
        std::cerr << "Only 'video' mode is supported." << std::endl;
        return -1;
    }

    cv::VideoCapture cap;
    if (path == "0") {
        cap.open(0);
    }
    else {
        cap.open(path);
    }

    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << path << std::endl;
        return -1;
    }

    std::thread readerThread(readFrames, std::ref(cap));
    std::thread processorThread(processFrames);

    readerThread.join();
    processorThread.join();

    cap.release();
    cv::destroyAllWindows();

    std::cout << "Processing complete" << std::endl;
    return 0;
}