#ifndef TRACKER_H
#define TRACKER_H
#include <vector>

//#include "KalmanFilters/kalmanfilter.h"

#include "kalmanfilter.h"
#include "track.h"
#include "../DeepSort/model.h"

class NearNeighborDisMetric;

class tracker
{
public:
    NearNeighborDisMetric* metric;
    DPKalmanFilter* kf;

    int _next_idx;
public:
    std::vector<Track> tracks;
    tracker(/*NearNeighborDisMetric* metric,*/);
    void predict();
    void update(const DETECTIONS& detections);
    typedef DYNAMICM (tracker::* GATED_METRIC_FUNC)(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
private:    
    void _match(const DETECTIONS& detections, TRACHER_MATCHD& res);
    void _initiate_track(const DETECTION_ROW& detection);
public:
    DYNAMICM gated_matric(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    DYNAMICM iou_cost(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(DETECTBOX& bbox,
            DETECTBOXSS &candidates);

private:
    float max_iou_distance = 0.55;
    int max_age = 80;
    int n_init = 4;
    float max_cosine_distance = 0.35;
    int nn_budget = 100;
};

#endif // TRACKER_H
