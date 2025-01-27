#include <iostream>
#include <string>
#include "mnist_loader.cuh"
#include "filters.cuh"
#include "utils.cuh"

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

        // Load MNIST data
        MNISTLoader loader;
        if (!loader.loadImages(args.input_path)) {
            throw std::runtime_error("Failed to load MNIST data");
        }

        MNISTData data = loader.getData();
        int num_images = (args.num_images > 0) ? 
                        std::min(args.num_images, data.num_images) : 
                        data.num_images;

        // Allocate GPU memory
        unsigned char *d_input = nullptr, *d_output = nullptr;
        size_t total_size = num_images * data.image_size * sizeof(unsigned char);
        
        cudaMalloc(&d_input, total_size);
        checkCudaError("Input allocation failed");
        
        cudaMalloc(&d_output, total_size);
        checkCudaError("Output allocation failed");

        // Copy data to GPU
        cudaMemcpy(d_input, data.images, total_size, cudaMemcpyHostToDevice);
        checkCudaError("Data copy to GPU failed");

        // Apply filter
        FilterType filter_type = stringToFilterType(args.filter_type);
        Logger::info("Applying " + args.filter_type + " filter...");
        
        applyFilter(d_input, d_output, 28, 28, num_images, filter_type);

        // Copy results back to CPU
        unsigned char* result = new unsigned char[total_size];
        cudaMemcpy(result, d_output, total_size, cudaMemcpyDeviceToHost);
        checkCudaError("Data copy from GPU failed");

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