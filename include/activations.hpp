#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "matrix.hpp"

template<Numeric T>
T relu(T x){
	return std::max(static_cast<T>(0), x);
}

template<Numeric T>
T relu_derivative(T x){
	if(x>0) return static_cast<T>(1);
	return static_cast<T>(0);
}

template<Numeric T>
T sigmoid(T x){ return 1/(1+std::exp(-x)); }

template<Numeric T>
T sigmoid_derivative(T x){
	T s = sigmoid(x);
	return s * (1-s);
}

template<Numeric T>
T tanh_act(T x){
	return std::tanh(x);
}

template<Numeric T>
T tanh_derivative(T x){
	T t=std::tanh(x);
	return 1-t*t;
}

// ── Matrix-level activation wrappers ────────────────────────────────────────

template <Numeric T>
Matrix<T> relu_mat(const Matrix<T>& m) {
    return m.apply(relu<T>);
}

template <Numeric T>
Matrix<T> relu_derivative_mat(const Matrix<T>& m) {
    return m.apply(relu_derivative<T>);
}

template <Numeric T>
Matrix<T> sigmoid_mat(const Matrix<T>& m) {
    return m.apply(sigmoid<T>);
}

template <Numeric T>
Matrix<T> sigmoid_derivative_mat(const Matrix<T>& m) {
    return m.apply(sigmoid_derivative<T>);
}

template <Numeric T>
Matrix<T> tanh_mat(const Matrix<T>& m) {
    return m.apply(tanh_act<T>);
}

template <Numeric T>
Matrix<T> tanh_derivative_mat(const Matrix<T>& m) {
    return m.apply(tanh_derivative<T>);
}

// softmax operates row by row — not elementwise
template <Numeric T>
Matrix<T> softmax_mat(const Matrix<T>& m) {
    Matrix<T> res(m.rows, m.cols);
    for (size_t i = 0; i < m.rows; i++) {
        // find max in row for numerical stability
        T max_val = m(i, 0);
        for (size_t j = 1; j < m.cols; j++)
            if (m(i, j) > max_val) max_val = m(i, j);

        // compute exp(x - max) for each element
        T sum = 0;
        for (size_t j = 0; j < m.cols; j++) {
            res(i, j) = std::exp(m(i, j) - max_val);
            sum += res(i, j);
        }

        // normalize
        for (size_t j = 0; j < m.cols; j++)
            res(i, j) /= sum;
    }
    return res;
}

// softmax derivative is absorbed into cross-entropy derivative (p - y)
// so we just return ones matrix as a passthrough
template <Numeric T>
Matrix<T> softmax_derivative_mat(const Matrix<T>& m) {
    Matrix<T> res(m.rows, m.cols);
    res.apply_inplace([](T) { return static_cast<T>(1); });
    return res;
}
