#pragma once
#include <cuda_runtime.h>

// Filter types
enum class FilterType {
    SOBEL,
    GAUSSIAN,
    SHARPEN
};

// CUDA kernel declarations
__global__ void sobelFilter(unsigned char* input, unsigned char* output, 
                          int width, int height, int num_images);
__global__ void gaussianBlur(unsigned char* input, unsigned char* output, 
                           int width, int height, int num_images);
__global__ void sharpenFilter(unsigned char* input, unsigned char* output, 
                            int width, int height, int num_images);

// Host functions
void applyFilter(unsigned char* d_input, unsigned char* d_output, 
                int width, int height, int num_images, FilterType filter_type);

// Error checking function
void checkCudaError(const char* msg);