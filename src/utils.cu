#include "utils.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>

Timer::Timer() {}

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

double Timer::stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                   (end_time - start_time_).count();
    return duration / 1000.0; // Convert to milliseconds
}

bool ImageUtils::saveImage(const unsigned char* data, int width, int height, 
                         const std::string& filename) {
    cv::Mat image(height, width, CV_8UC1, const_cast<unsigned char*>(data));
    return cv::imwrite(filename, image);
}

void ImageUtils::createDirectory(const std::string& path) {
    #ifdef _WIN32
        _mkdir(path.c_str());
    #else
        mkdir(path.c_str(), 0777);
    #endif
}

bool ImageUtils::saveBatch(const unsigned char* data, int num_images, int width, int height, 
                         const std::string& directory) {
    createDirectory(directory);
    
    bool success = true;
    for (int i = 0; i < num_images; i++) {
        std::string filename = directory + "/image_" + 
                             std::to_string(i) + ".png";
        success &= saveImage(data + i * width * height, width, height, filename);
    }
    return success;
}

void Logger::info(const std::string& message) {
    std::cout << "[INFO] " << message << std::endl;
}

void Logger::error(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}