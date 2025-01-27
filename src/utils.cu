#include "utils.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>   // mkdir 사용을 위해

void Logger::info(const std::string& message) {
    std::cout << "[INFO] " << message << std::endl;
}

void Logger::error(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}

Timer::Timer() {}

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

double Timer::stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                   (end_time - start_time);
    return duration.count() / 1000.0; // 밀리초로 변환
}

bool ImageUtils::saveBatch(const unsigned char* images, int num_images, 
                         int width, int height, 
                         const std::string& output_path) {
    // 디렉토리 생성 (이미 존재하는 경우 무시)
    mkdir(output_path.c_str(), 0777);
    
    for (int i = 0; i < num_images; i++) {
        std::string filename = output_path + "/image_" + 
                             std::to_string(i) + ".pgm";
        std::ofstream file(filename, std::ios::binary);
        
        if (!file.is_open()) {
            return false;
        }

        // PGM 헤더 작성
        file << "P5\n" << width << " " << height << "\n255\n";
        
        // 이미지 데이터 작성
        file.write(reinterpret_cast<const char*>(images + i * width * height), 
                  width * height);
        
        file.close();
    }
    
    return true;
}