#pragma once
#include <string>
#include <chrono>

class Timer {
public:
    Timer();
    void start();
    double stop(); // Returns elapsed time in milliseconds

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

namespace ImageUtils {
    bool saveImage(const unsigned char* data, int width, int height, 
                  const std::string& filename);
    bool saveBatch(const unsigned char* data, int num_images, int width, int height, 
                  const std::string& directory);
    void createDirectory(const std::string& path);
}

class Logger {
public:
    static void info(const std::string& message);
    static void error(const std::string& message);
};