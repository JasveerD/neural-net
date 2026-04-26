#pragma once
#include "matrix.hpp"
#include "network.hpp"
#include <fstream>
#include <string>
#include <stdexcept>

template <Numeric T>
void save(const Network<T>& net, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for saving: " + path);

    // write number of layers
    size_t n_layers = net.layers.size();
    f.write(reinterpret_cast<const char*>(&n_layers), sizeof(size_t));

    for (const auto& layer : net.layers) {
        // write weight shape
        f.write(reinterpret_cast<const char*>(&layer.weight.rows), sizeof(size_t));
        f.write(reinterpret_cast<const char*>(&layer.weight.cols), sizeof(size_t));

        // write weight data
        for (size_t i = 0; i < layer.weight.rows; i++)
            for (size_t j = 0; j < layer.weight.cols; j++) {
                T val = layer.weight(i, j);
                f.write(reinterpret_cast<const char*>(&val), sizeof(T));
            }

        // write bias shape
        f.write(reinterpret_cast<const char*>(&layer.bias.rows), sizeof(size_t));
        f.write(reinterpret_cast<const char*>(&layer.bias.cols), sizeof(size_t));

        // write bias data
        for (size_t i = 0; i < layer.bias.rows; i++)
            for (size_t j = 0; j < layer.bias.cols; j++) {
                T val = layer.bias(i, j);
                f.write(reinterpret_cast<const char*>(&val), sizeof(T));
            }
    }
}

template <Numeric T>
void load(Network<T>& net, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for loading: " + path);

    // read number of layers
    size_t n_layers;
    f.read(reinterpret_cast<char*>(&n_layers), sizeof(size_t));

    if (n_layers != net.layers.size())
        throw std::runtime_error("Network architecture mismatch");

    for (auto& layer : net.layers) {
        // read weight shape
        size_t w_rows, w_cols;
        f.read(reinterpret_cast<char*>(&w_rows), sizeof(size_t));
        f.read(reinterpret_cast<char*>(&w_cols), sizeof(size_t));

        if (w_rows != layer.weight.rows || w_cols != layer.weight.cols)
            throw std::runtime_error("Weight shape mismatch");

        // read weight data
        for (size_t i = 0; i < w_rows; i++)
            for (size_t j = 0; j < w_cols; j++) {
                T val;
                f.read(reinterpret_cast<char*>(&val), sizeof(T));
                layer.weight(i, j) = val;
            }

        // read bias shape
        size_t b_rows, b_cols;
        f.read(reinterpret_cast<char*>(&b_rows), sizeof(size_t));
        f.read(reinterpret_cast<char*>(&b_cols), sizeof(size_t));

        if (b_rows != layer.bias.rows || b_cols != layer.bias.cols)
            throw std::runtime_error("Bias shape mismatch");

        // read bias data
        for (size_t i = 0; i < b_rows; i++)
            for (size_t j = 0; j < b_cols; j++) {
                T val;
                f.read(reinterpret_cast<char*>(&val), sizeof(T));
                layer.bias(i, j) = val;
            }
    }
}
