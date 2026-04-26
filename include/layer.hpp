#pragma once
#include "matrix.hpp"
#include <cstddef>
#include <functional>

template<Numeric T>
class Layer{
public:
	// fields
	Matrix<T> weight;
	Matrix<T> bias;
	Matrix<T> cached_input;
	Matrix<T> pre_activation_cache;

	Matrix<T> dW;  // gradient of loss w.r.t weights
	Matrix<T> db;  // gradient of loss w.r.t bias

	std::function<Matrix<T>(const Matrix<T>&)> activation;
	std::function<Matrix<T>(const Matrix<T>&)> activation_derivative;

	// methods
	Matrix<T> forward(const Matrix<T>& input);
	Matrix<T> backward(const Matrix<T>& grad);

	// constructor
	Layer(size_t in_features, size_t out_features,
      std::function<Matrix<T>(const Matrix<T>&)> activation,
      std::function<Matrix<T>(const Matrix<T>&)> activation_derivative);
};


// Implimentations ----------------------------------------------------------------------
// constructor
template<Numeric T>
Layer<T>::Layer(size_t in_features, size_t out_features,
        std::function<Matrix<T>(const Matrix<T>&)> activation,
        std::function<Matrix<T>(const Matrix<T>&)> activation_derivative)
    : weight(Matrix<T>::random(out_features, in_features)),
      bias(Matrix<T>::zeros(1, out_features)),
      cached_input(Matrix<T>()),
      pre_activation_cache(Matrix<T>()),
      dW(Matrix<T>()),
      db(Matrix<T>()),
      activation(activation),
      activation_derivative(activation_derivative)
{}

template <Numeric T>
Matrix<T> Layer<T>::forward(const Matrix<T>& input){
	cached_input = input;
	Matrix<T> z = input.matmul(weight.transpose());

	// bias[1, out_features]
	// z[batch, out_features] 
	for(size_t i=0; i<z.rows; i++){
		for(size_t j=0; j<bias.cols; j++){
			z(i,j) += bias(0, j);	
		}
	}
	pre_activation_cache = z;

	return activation(z);
}

template <Numeric T>
Matrix<T> Layer<T>::backward(const Matrix<T>& grad) {
    Matrix<T> dZ = grad * activation_derivative(pre_activation_cache);

    dW = dZ.transpose().matmul(cached_input);

    db = Matrix<T>::zeros(1, dZ.cols);
    for(size_t i=0; i<dZ.rows; i++)
        for(size_t j=0; j<dZ.cols; j++)
            db(0,j) += dZ(i,j);

    return dZ.matmul(weight);
}
