#include "mnist_loader.cuh"
#include <fstream>
#include <iostream>

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