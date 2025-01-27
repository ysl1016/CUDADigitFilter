#ifndef UTILS_CUH
#define UTILS_CUH

#include <string>
#include <chrono>

class Logger {
public:
    static void info(const std::string& message);
    static void error(const std::string& message);
};

class Timer {
public:
    Timer();
    void start();
    double stop(); // 밀리초 단위로 반환

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

class ImageUtils {
public:
    static bool saveBatch(const unsigned char* images, int num_images, 
                         int width, int height, 
                         const std::string& output_path);
};

#endif // UTILS_CUH