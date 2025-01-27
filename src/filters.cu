#include "filters.cuh"
#include <cuda_runtime.h>

__global__ void sobelFilter(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Skip border pixels
        if (x == 0 || x == width-1 || y == 0 || y == height-1) {
            output[y * width + x] = 0;
            return;
        }

        // Sobel kernels
        int Gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
                 -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                 -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];

        int Gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                 +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

        int val = sqrt((float)(Gx*Gx + Gy*Gy));
        output[y * width + x] = (unsigned char)min(255, val);
    }
}
// 기존 코드에 추가

__global__ void gaussianBlur(unsigned char* input, unsigned char* output, 
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        if (x < 2 || x >= width-2 || y < 2 || y >= height-2) {
            output[y * width + x] = input[y * width + x];
            return;
        }

        float gaussian[5][5] = {
            {1, 4, 6, 4, 1},
            {4, 16, 24, 16, 4},
            {6, 24, 36, 24, 6},
            {4, 16, 24, 16, 4},
            {1, 4, 6, 4, 1}
        };
        
        float sum = 0;
        float weightSum = 0;
        
        for(int i = -2; i <= 2; i++) {
            for(int j = -2; j <= 2; j++) {
                float weight = gaussian[i+2][j+2];
                sum += input[(y+i)*width + (x+j)] * weight;
                weightSum += weight;
            }
        }
        
        output[y * width + x] = static_cast<unsigned char>(sum / weightSum);
    }
}

__global__ void sharpenFilter(unsigned char* input, unsigned char* output, 
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        if (x == 0 || x == width-1 || y == 0 || y == height-1) {
            output[y * width + x] = input[y * width + x];
            return;
        }

        int kernel[3][3] = {
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}
        };
        
        int sum = 0;
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                sum += input[(y+i)*width + (x+j)] * kernel[i+1][j+1];
            }
        }
        
        output[y * width + x] = static_cast<unsigned char>(std::min(255, std::max(0, sum)));
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