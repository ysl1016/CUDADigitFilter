#include <fstream>
#include <iostream>  // std::cerr을 위해 추가
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include "mnist_loader.cuh"
#include "filters.cuh"

// 로그 파일 스트림을 전역 변수로 선언
std::ofstream logFile;

// 로그 작성 함수
void writeLog(const std::string& message) {
    // 현재 시간 가져오기
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    
    // 로그 파일에 시간과 메시지 작성
    logFile << std::ctime(&time) << message << std::endl;
}

int main(int argc, char** argv) {
    // 로그 파일 열기
    logFile.open("output.txt");
    
    if (!logFile.is_open()) {
        std::cout << "Failed to open log file" << std::endl;  // cerr 대신 cout 사용
        return -1;
    }

    writeLog("CUDA Digit Filter Processing Started");
    
    // CUDA 디바이스 정보 로깅
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    writeLog("CUDA Device Information:");
    writeLog("  Device Name: " + std::string(prop.name));
    writeLog("  Compute Capability: " + std::to_string(prop.major) + "." + std::to_string(prop.minor));
    writeLog("  Max Threads per Block: " + std::to_string(prop.maxThreadsPerBlock));

    // MNIST 데이터 로드 시작
    writeLog("Loading MNIST data...");
    
    // 타이머 시작
    auto start = std::chrono::high_resolution_clock::now();
    
    // 변수 선언
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    int width = 28;  // MNIST 이미지 너비
    int height = 28; // MNIST 이미지 높이
    size_t size = width * height * sizeof(unsigned char);

    // CUDA 메모리 할당
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    
    // MNIST 데이터 로드 (mnist_loader.cuh의 함수 사용)
    std::vector<unsigned char> mnist_data;
    if (!loadMNISTImages("data/train-images.idx3-ubyte", mnist_data)) {
        writeLog("Failed to load MNIST data");
        cudaFree(d_input);
        cudaFree(d_output);
        logFile.close();
        return -1;
    }

    // 데이터를 GPU로 복사
    cudaMemcpy(d_input, mnist_data.data(), size, cudaMemcpyHostToDevice);
    
    auto load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_time = load_end - start;
    writeLog("MNIST data loaded in " + std::to_string(load_time.count()) + " seconds");

    // 필터 적용
    writeLog("Applying filters...");
    
    // 각 필터별 처리 시간 측정
    for (int i = 0; i < 5; i++) {
        writeLog("Processing image " + std::to_string(i+1));
        
        auto filter_start = std::chrono::high_resolution_clock::now();
        
        // Sobel 필터
        writeLog("  Applying Sobel filter...");
        applyFilter(d_input, d_output, width, height, FilterType::SOBEL);
        
        auto sobel_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sobel_time = sobel_end - filter_start;
        writeLog("  Sobel filter completed in " + std::to_string(sobel_time.count()) + " seconds");
        
        // Gaussian 필터
        writeLog("  Applying Gaussian blur...");
        applyFilter(d_input, d_output, width, height, FilterType::GAUSSIAN);
        
        auto gaussian_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gaussian_time = gaussian_end - sobel_end;
        writeLog("  Gaussian blur completed in " + std::to_string(gaussian_time.count()) + " seconds");
        
        // Sharpen 필터
        writeLog("  Applying Sharpen filter...");
        applyFilter(d_input, d_output, width, height, FilterType::SHARPEN);
        
        auto sharpen_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sharpen_time = sharpen_end - gaussian_end;
        writeLog("  Sharpen filter completed in " + std::to_string(sharpen_time.count()) + " seconds");
    }

    // 메모리 해제
    writeLog("Cleaning up resources...");
    cudaFree(d_input);
    cudaFree(d_output);

    // 총 실행 시간 계산
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end - start;
    writeLog("Total execution time: " + std::to_string(total_time.count()) + " seconds");

    // CUDA 오류 체크
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        writeLog("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }

    writeLog("Processing completed successfully");
    
    // 로그 파일 닫기
    logFile.close();
    
    return 0;
}