#include <iostream>
#include <string>
#include "mnist_loader.cuh"
#include "filters.cuh"
#include "utils.cuh"
#include <fstream>
#include <chrono>
#include <ctime>

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
        std::cerr << "Failed to open log file" << std::endl;
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
    
    // MNIST 데이터 로드
    unsigned char* d_input;
    size_t size;
    loadMNISTImages("path/to/mnist/data", &d_input, &size);
    
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

struct CommandLineArgs {
    std::string input_path;
    std::string output_path;
    std::string filter_type;
    int num_images = -1;  // -1 means process all images
};

CommandLineArgs parseArgs(int argc, char** argv) {
    CommandLineArgs args;
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "--input") {
            args.input_path = argv[i + 1];
        } else if (arg == "--output") {
            args.output_path = argv[i + 1];
        } else if (arg == "--filter") {
            args.filter_type = argv[i + 1];
        } else if (arg == "--num-images") {
            args.num_images = std::stoi(argv[i + 1]);
        }
    }
    return args;
}

FilterType stringToFilterType(const std::string& filter_name) {
    if (filter_name == "sobel") return FilterType::SOBEL;
    if (filter_name == "gaussian") return FilterType::GAUSSIAN;
    if (filter_name == "sharpen") return FilterType::SHARPEN;
    throw std::runtime_error("Unknown filter type: " + filter_name);
}

int main(int argc, char** argv) {
    try {
        CommandLineArgs args = parseArgs(argc, argv);
        
        // Validate arguments
        if (args.input_path.empty() || args.output_path.empty() || args.filter_type.empty()) {
            throw std::runtime_error("Missing required arguments");
        }

        Logger::info("Loading MNIST data...");
        MNISTLoader loader;
        if (!loader.loadImages(args.input_path)) {
            throw std::runtime_error("Failed to load MNIST images");
        }

        MNISTData data = loader.getData();
        int num_images = (args.num_images == -1) ? data.num_images : 
                        std::min(args.num_images, data.num_images);

        // GPU Memory allocation
        unsigned char *d_input, *d_output;
        cudaMalloc(&d_input, num_images * data.image_size * sizeof(unsigned char));
        cudaMalloc(&d_output, num_images * data.image_size * sizeof(unsigned char));

        // Copy data to GPU
        Timer timer;
        timer.start();
        
        cudaMemcpy(d_input, data.images, 
                   num_images * data.image_size * sizeof(unsigned char), 
                   cudaMemcpyHostToDevice);

        // Apply filter
        FilterType filter_type = stringToFilterType(args.filter_type);
        Logger::info("Applying " + args.filter_type + " filter...");
        
        applyFilter(d_input, d_output, 28, 28, filter_type);

        // Copy results back to CPU
        unsigned char* result = new unsigned char[num_images * data.image_size];
        cudaMemcpy(result, d_output, 
                   num_images * data.image_size * sizeof(unsigned char), 
                   cudaMemcpyDeviceToHost);

        double elapsed_time = timer.stop();
        Logger::info("Processing completed in " + 
                    std::to_string(elapsed_time) + " ms");

        // Save results
        Logger::info("Saving processed images...");
        if (!ImageUtils::saveBatch(result, num_images, 28, 28, args.output_path)) {
            throw std::runtime_error("Failed to save output images");
        }

        // Clean up
        cudaFree(d_input);
        cudaFree(d_output);
        delete[] result;

        Logger::info("Processing completed successfully");
        return 0;

    } catch (const std::exception& e) {
        Logger::error(e.what());
        return 1;
    }
}