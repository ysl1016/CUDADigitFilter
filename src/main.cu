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