#pragma once
#include <cmath>
#include <cstddef>
#include "matrix.hpp"

// predictions: guesses made by model
// targets: true classification (one-codded) one for the right class, 0 for all others
template <Numeric T>
T cross_entropy_loss(const Matrix<T>& predictions, const Matrix<T>& targets) {
	assert(predictions.rows == targets.rows && predictions.cols == targets.cols);

	T loss=0;
	for(size_t i=0; i<predictions.rows; i++){
		for(size_t j=0; j<predictions.cols; j++){
			loss -= targets(i,j) * std::log(predictions(i,j) + 1e-9); // 1e-9 so that log(0) doesn't happen
		}
	}
	return loss/static_cast<T>(predictions.rows);
}

template <Numeric T>
Matrix<T> cross_entropy_derivative(const Matrix<T>& predictions, const Matrix<T>& targets) {
	assert(predictions.rows == targets.rows && predictions.cols == targets.cols);

	T batch_size = static_cast<T>(predictions.rows);
	return (predictions - targets) * (static_cast<T>(1)/batch_size);
}
