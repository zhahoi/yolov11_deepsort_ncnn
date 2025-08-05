cmake_minimum_required(VERSION 3.10.1)
project(yolov11_deepsort)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-std=c++11)
add_definitions(-pthread)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# 设置 Release 优化
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

# OpenMP 支持
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message("OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# OpenCV
set(OpenCV_DIR "/home/hit/Softwares/opencv-3.4.10/build")
find_package(OpenCV REQUIRED)

# ncnn
set(ncnn_DIR "/home/hit/Softwares/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND! Please set ncnn_DIR environment variable")
else()
    message("ncnn found")
endif()

# Include paths
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/YOLOv11
    ${PROJECT_SOURCE_DIR}/DeepSort
    ${PROJECT_SOURCE_DIR}/KalmanFilter
    ${PROJECT_SOURCE_DIR}/MunkresAssignment
    ${PROJECT_SOURCE_DIR}/MunkresAssignment/munkres
	/usr/local/include/eigen3
)

# 源文件收集
set(SOURCE_FILES
    main.cpp

    # YOLO
    YOLOv11/yolov11.cpp

    # DeepSort
    DeepSort/deepsort.cpp
    DeepSort/model.cpp

    # KalmanFilter
    KalmanFilter/kalmanfilter.cpp
    KalmanFilter/linear_assignment.cpp
    KalmanFilter/nn_matching.cpp
    KalmanFilter/track.cpp
    KalmanFilter/tracker.cpp

    # Hungarian / Munkres
    MunkresAssignment/hungarianoper.cpp
    MunkresAssignment/munkres/munkres.cpp
)

# 可执行文件
add_executable(yolov11_deepsort ${SOURCE_FILES})

# 链接库
target_link_libraries(
    yolov11_deepsort
    ${OpenCV_LIBS}
    ncnn
)
