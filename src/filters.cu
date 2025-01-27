#include "filters.cuh"
#include <algorithm>
#include <stdio.h>

void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

// 헬퍼 함수
__device__ int clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// CUDA 커널 구현
__global__ void sobelFilter(unsigned char* input, unsigned char* output, 
                          int width, int height, int num_images) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;
    
    if (x < width && y < height && img_idx < num_images) {
        size_t offset = img_idx * width * height;
        int gx = 0, gy = 0;
        
        // Sobel 연산
        if (x > 0 && x < width-1 && y > 0 && y < height-1) {
            // X 방향 그래디언트
            gx = -input[offset + (y-1)*width + (x-1)] + input[offset + (y-1)*width + (x+1)]
                 -2*input[offset + y*width + (x-1)] + 2*input[offset + y*width + (x+1)]
                 -input[offset + (y+1)*width + (x-1)] + input[offset + (y+1)*width + (x+1)];
                 
            // Y 방향 그래디언트
            gy = -input[offset + (y-1)*width + (x-1)] - 2*input[offset + (y-1)*width + x] - input[offset + (y-1)*width + (x+1)]
                 +input[offset + (y+1)*width + (x-1)] + 2*input[offset + (y+1)*width + x] + input[offset + (y+1)*width + (x+1)];
        }
        
        int sum = clamp((int)sqrt((float)(gx*gx + gy*gy)), 0, 255);
        output[offset + y * width + x] = static_cast<unsigned char>(sum);
    }
}

__global__ void gaussianBlur(unsigned char* input, unsigned char* output, 
                           int width, int height, int num_images) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;
    
    if (x < width && y < height && img_idx < num_images) {
        size_t offset = img_idx * width * height;
        float sum = 0.0f;
        float kernel[3][3] = {
            {1/16.0f, 2/16.0f, 1/16.0f},
            {2/16.0f, 4/16.0f, 2/16.0f},
            {1/16.0f, 2/16.0f, 1/16.0f}
        };
        
        if (x > 0 && x < width-1 && y > 0 && y < height-1) {
            for(int ky = -1; ky <= 1; ky++) {
                for(int kx = -1; kx <= 1; kx++) {
                    sum += input[offset + (y+ky)*width + (x+kx)] * kernel[ky+1][kx+1];
                }
            }
        } else {
            sum = input[offset + y*width + x];
        }
        
        output[offset + y * width + x] = static_cast<unsigned char>(clamp((int)sum, 0, 255));
    }
}

__global__ void sharpenFilter(unsigned char* input, unsigned char* output, 
                            int width, int height, int num_images) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;
    
    if (x < width && y < height && img_idx < num_images) {
        size_t offset = img_idx * width * height;
        float kernel[3][3] = {
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}
        };
        
        float sum = 0;
        if (x > 0 && x < width-1 && y > 0 && y < height-1) {
            for(int ky = -1; ky <= 1; ky++) {
                for(int kx = -1; kx <= 1; kx++) {
                    sum += input[offset + (y+ky)*width + (x+kx)] * kernel[ky+1][kx+1];
                }
            }
        } else {
            sum = input[offset + y*width + x];
        }
        
        output[offset + y * width + x] = static_cast<unsigned char>(clamp((int)sum, 0, 255));
    }
}

void applyFilter(unsigned char* d_input, unsigned char* d_output, 
                int width, int height, int num_images, FilterType filter_type) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y,
                  num_images);  // Z 차원에 이미지 수 추가

    switch(filter_type) {
        case FilterType::SOBEL:
            sobelFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height, num_images);
            break;
        case FilterType::GAUSSIAN:
            gaussianBlur<<<gridSize, blockSize>>>(d_input, d_output, width, height, num_images);
            break;
        case FilterType::SHARPEN:
            sharpenFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height, num_images);
            break;
    }
    
    cudaDeviceSynchronize();
    checkCudaError("Kernel execution failed");
}