#include "filters.cuh"
#include <cuda_runtime.h>
#include <stdio.h> 
#include <arpa/inet.h>

__global__ void sobelFilter(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // 경계 픽셀 처리 개선
        if (x == 0 || x == width-1 || y == 0 || y == height-1) {
            output[y * width + x] = 0;  // 경계는 검은색으로 처리
            return;
        }

        // Sobel 연산자 강도 조절
        float Gx = 0.0f;
        float Gy = 0.0f;

        // X 방향 Sobel
        Gx += input[(y-1)*width + (x+1)] - input[(y-1)*width + (x-1)];
        Gx += 2.0f * (input[y*width + (x+1)] - input[y*width + (x-1)]);
        Gx += input[(y+1)*width + (x+1)] - input[(y+1)*width + (x-1)];

        // Y 방향 Sobel
        Gy += input[(y+1)*width + (x-1)] - input[(y-1)*width + (x-1)];
        Gy += 2.0f * (input[(y+1)*width + x] - input[(y-1)*width + x]);
        Gy += input[(y+1)*width + (x+1)] - input[(y-1)*width + (x+1)];

        // 결과 계산 및 스케일링
        float magnitude = sqrtf(Gx*Gx + Gy*Gy);
        
        // 스케일링 팩터 조정 (0.5f는 필터 강도를 조절하는 값)
        magnitude = magnitude * 0.5f;
        
        // 임계값 처리
        if (magnitude < 30.0f) {  // 노이즈 제거를 위한 임계값
            magnitude = 0.0f;
        }
        
        output[y * width + x] = (unsigned char)(magnitude > 255.0f ? 255 : magnitude);
    }
}

__global__ void gaussianBlur(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        if (x < 2 || x >= width-2 || y < 2 || y >= height-2) {
            output[y * width + x] = input[y * width + x];
            return;
        }

        // 가우시안 커널 크기 축소 및 가중치 조정
        float gaussian[3][3] = {
            {1.0f, 2.0f, 1.0f},
            {2.0f, 4.0f, 2.0f},
            {1.0f, 2.0f, 1.0f}
        };
        
        float sum = 0.0f;
        float weightSum = 0.0f;
        
        // 3x3 커널 적용
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                float weight = gaussian[i+1][j+1];
                sum += input[(y+i)*width + (x+j)] * weight;
                weightSum += weight;
            }
        }
        
        output[y * width + x] = static_cast<unsigned char>(sum / weightSum);
    }
}

__global__ void sharpenFilter(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        if (x == 0 || x == width-1 || y == 0 || y == height-1) {
            output[y * width + x] = input[y * width + x];
            return;
        }

        // 샤프닝 커널 가중치 조정
        float kernel[3][3] = {
            {0.0f, -1.0f, 0.0f},
            {-1.0f, 5.0f, -1.0f},
            {0.0f, -1.0f, 0.0f}
        };
        
        float sum = 0.0f;
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                sum += input[(y+i)*width + (x+j)] * kernel[i+1][j+1];
            }
        }
        
        // 결과값 클리핑
        sum = fmaxf(0.0f, fminf(255.0f, sum));
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

void applyFilter(unsigned char* d_input, unsigned char* d_output, 
                int width, int height, FilterType filter_type) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // CUDA 장치 초기화
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        printf("CUDA device reset error: %s\n", cudaGetErrorString(err));
        return;
    }

    // 필터 적용
    switch(filter_type) {
        case FilterType::SOBEL:
            sobelFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);
            break;
        case FilterType::GAUSSIAN:
            gaussianBlur<<<gridSize, blockSize>>>(d_input, d_output, width, height);
            break;
        case FilterType::SHARPEN:
            sharpenFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);
            break;
    }
    
    // CUDA 오류 체크
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
    }
}