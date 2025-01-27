#include "mnist_loader.cuh"
#include <fstream>
#include <iostream>
// 필요한 헤더 파일 포함
#include <cuda_runtime.h>
#include <stdio.h>

// 함수 정의
void loadMNISTImages(const char* filename, unsigned char** d_input, size_t* size) {
    // MNIST 데이터 로드
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    // 헤더 읽기
    int magic_number;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = ntohl(magic_number); // big-endian to little-endian
    if (magic_number != 2051) {
        printf("Invalid magic number: %d\n", magic_number);
        fclose(file);
        return;
    }

    int num_images, rows, cols;
    fread(&num_images, sizeof(num_images), 1, file);
    fread(&rows, sizeof(rows), 1, file);
    fread(&cols, sizeof(cols), 1, file);
    num_images = ntohl(num_images);
    rows = ntohl(rows);
    cols = ntohl(cols);

    // 호스트 메모리 할당
    size_t data_size = num_images * rows * cols;
    unsigned char* h_input = (unsigned char*)malloc(data_size);
    fread(h_input, sizeof(unsigned char), data_size, file);
    fclose(file);

    // 메모리 할당
    cudaError_t err = cudaMalloc((void**)d_input, data_size);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // 메모리 복사
    err = cudaMemcpy(*d_input, h_input, data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // 호스트 메모리 해제
    free(h_input);
}

MNISTLoader::MNISTLoader() {
    data_.images = nullptr;
    data_.num_images = 0;
    data_.image_size = 784; // 28x28
}

MNISTLoader::~MNISTLoader() {
    if (data_.images) {
        delete[] data_.images;
    }
}

int MNISTLoader::reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

bool MNISTLoader::loadImages(const std::string& image_path) {
    std::ifstream file(image_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << image_path << std::endl;
        return false;
    }

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char*)&data_.num_images, sizeof(data_.num_images));
    data_.num_images = reverseInt(data_.num_images);

    int n_rows = 0, n_cols = 0;
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));
    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);

    data_.images = new unsigned char[data_.num_images * data_.image_size];
    for(int i = 0; i < data_.num_images; i++) {
        file.read((char*)&data_.images[i * data_.image_size], data_.image_size);
    }

    return true;
}

MNISTData MNISTLoader::getData() const {
    return data_;
}
