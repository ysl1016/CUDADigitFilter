// mnist_loader.cuh
#ifndef MNIST_LOADER_CUH
#define MNIST_LOADER_CUH

#include <vector>
#include <string>

bool loadMNISTImages(const std::string& filename, std::vector<unsigned char>& images);

#endif // MNIST_LOADER_CUH