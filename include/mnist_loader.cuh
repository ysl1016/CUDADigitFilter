#pragma once
#include <vector>
#include <string>

struct MNISTData {
    unsigned char* images;
    int num_images;
    int image_size;  // 28x28 = 784
};

class MNISTLoader {
public:
    MNISTLoader();
    ~MNISTLoader();
    
    bool loadImages(const std::string& image_path);
    MNISTData getData() const;

private:
    MNISTData data_;
    int reverseInt(int i);
};