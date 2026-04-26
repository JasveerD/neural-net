#pragma once
#include "matrix.hpp"
#include <cstdint>
#include <fstream>
#include <string>
#include <stdexcept>
#include <algorithm>

template <Numeric T>
class DataLoader {
public:
    Matrix<T> images;   // (n_images, 784)
    Matrix<T> labels;   // (n_images, 10)
    size_t batch_size;
    size_t current_index;
    size_t n_samples;

    // constructor — loads both files
    DataLoader(const std::string& image_path, const std::string& label_path, size_t batch_size);

    // returns next batch of images and labels
    // returns false when epoch is done
    bool next_batch(Matrix<T>& batch_images, Matrix<T>& batch_labels);

    // resets current_index to 0 for next epoch
    void reset();

private:
    // reads and byte-swaps a 4-byte big-endian int from file
    int32_t read_int(std::ifstream& f);

    // parses image file into (n, 784) matrix, normalized to [0,1]
    Matrix<T> load_images(const std::string& path);

    // parses label file into (n, 10) one-hot matrix
    Matrix<T> load_labels(const std::string& path);
};

template <Numeric T>
int32_t DataLoader<T>::read_int(std::ifstream& f) {
    int32_t val;
    f.read(reinterpret_cast<char*>(&val), 4);
    // swap bytes: 0x12345678 becomes 0x78563412
    return ((val & 0xFF) << 24) |
           ((val & 0xFF00) << 8) |
           ((val & 0xFF0000) >> 8) |
           ((val >> 24) & 0xFF);
}

template <Numeric T>
Matrix<T> DataLoader<T>::load_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open image file: " + path);

    int32_t magic    = read_int(f);
    int32_t n_images = read_int(f);
    int32_t n_rows   = read_int(f);
    int32_t n_cols   = read_int(f);

    assert(magic == 2051);

    size_t n_pixels = n_rows * n_cols; // 784
    Matrix<T> result(n_images, n_pixels);

    for (int32_t i = 0; i < n_images; i++) {
        for (int32_t j = 0; j < (int32_t)n_pixels; j++) {
            uint8_t pixel;
            f.read(reinterpret_cast<char*>(&pixel), 1);
            result(i, j) = static_cast<T>(pixel) / static_cast<T>(255);
        }
    }
    return result;
}

template <Numeric T>
Matrix<T> DataLoader<T>::load_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open label file: " + path);

    int32_t magic    = read_int(f);
    int32_t n_labels = read_int(f);

    assert(magic == 2049);

    Matrix<T> result = Matrix<T>::zeros(n_labels, 10);

    for (int32_t i = 0; i < n_labels; i++) {
        uint8_t label;
        f.read(reinterpret_cast<char*>(&label), 1);
        result(i, label) = static_cast<T>(1); // one-hot encode
    }
    return result;
}

template <Numeric T>
DataLoader<T>::DataLoader(const std::string& image_path, const std::string& label_path, size_t batch_size)
    : batch_size(batch_size), current_index(0) {
    images   = load_images(image_path);
    labels   = load_labels(label_path);
    n_samples = images.rows;
}

template <Numeric T>
bool DataLoader<T>::next_batch(Matrix<T>& batch_images, Matrix<T>& batch_labels) {
    if (current_index >= n_samples) return false;

    size_t end = std::min(current_index + batch_size, n_samples);
    size_t actual_batch = end - current_index;

    batch_images = Matrix<T>(actual_batch, 784);
    batch_labels = Matrix<T>(actual_batch, 10);

    for (size_t i = 0; i < actual_batch; i++) {
        for (size_t j = 0; j < 784; j++)
            batch_images(i, j) = images(current_index + i, j);
        for (size_t j = 0; j < 10; j++)
            batch_labels(i, j) = labels(current_index + i, j);
    }

    current_index = end;
    return true;
}

template <Numeric T>
void DataLoader<T>::reset() {
    current_index = 0;
}
