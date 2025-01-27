#include "filters.cuh"
#include <algorithm>

// 헬퍼 함수
__device__ int clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// CUDA 커널 구현
__global__ void sobelFilterKernel(unsigned char* input, unsigned char* output, 
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int gx = 0, gy = 0;
        
        // Sobel 연산
        if (x > 0 && x < width-1 && y > 0 && y < height-1) {
            // X 방향 그래디언트
            gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
                 -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                 -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];
                 
            // Y 방향 그래디언트
            gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                 +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];
        }
        
        int sum = clamp((int)sqrt((float)(gx*gx + gy*gy)), 0, 255);
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

__global__ void gaussianFilterKernel(unsigned char* input, unsigned char* output, 
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        float kernel[3][3] = {
            {1/16.0f, 2/16.0f, 1/16.0f},
            {2/16.0f, 4/16.0f, 2/16.0f},
            {1/16.0f, 2/16.0f, 1/16.0f}
        };
        
        if (x > 0 && x < width-1 && y > 0 && y < height-1) {
            for(int ky = -1; ky <= 1; ky++) {
                for(int kx = -1; kx <= 1; kx++) {
                    sum += input[(y+ky)*width + (x+kx)] * kernel[ky+1][kx+1];
                }
            }
        } else {
            sum = input[y*width + x];
        }
        
        output[y * width + x] = static_cast<unsigned char>(clamp((int)sum, 0, 255));
    }
}

__global__ void sharpenFilterKernel(unsigned char* input, unsigned char* output, 
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float kernel[3][3] = {
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}
        };
        
        float sum = 0;
        if (x > 0 && x < width-1 && y > 0 && y < height-1) {
            for(int ky = -1; ky <= 1; ky++) {
                for(int kx = -1; kx <= 1; kx++) {
                    sum += input[(y+ky)*width + (x+kx)] * kernel[ky+1][kx+1];
                }
            }
        } else {
            sum = input[y*width + x];
        }
        
        output[y * width + x] = static_cast<unsigned char>(clamp((int)sum, 0, 255));
    }
}

void applyFilter(unsigned char* d_input, unsigned char* d_output, 
                int width, int height, FilterType filter_type) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    switch(filter_type) {
        case FilterType::SOBEL:
            sobelFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);
            break;
        // Add other filters as needed
    }
    
    cudaDeviceSynchronize();
}