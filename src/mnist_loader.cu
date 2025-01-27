#include "mnist_loader.cuh"
#include <fstream>
#include <iostream>

bool loadMNISTImages(const std::string& filename, std::vector<unsigned char>& images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        return false;
    }

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    magic_number = __builtin_bswap32(magic_number);
    number_of_images = __builtin_bswap32(number_of_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic_number != 2051) {
        std::cout << "Invalid MNIST image file!" << std::endl;
        return false;
    }

    size_t image_size = rows * cols;
    images.resize(image_size * number_of_images);
    file.read((char*)images.data(), images.size());

    file.close();
    return true;
}