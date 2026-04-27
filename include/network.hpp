#pragma once
#include "matrix.hpp"
#include "layer.hpp"
#include <algorithm>
#include <vector>

template <Numeric T>
class Network{
public:
	std::vector<Layer<T>> layers;
	
	// add layer into the network
	void add_layer(Layer<T> l);

	Matrix<T> forward(const Matrix<T>& input);
	void backward(const Matrix<T>& grad);
};

template <Numeric T>
void Network<T>::add_layer(Layer<T> l){
	layers.push_back(std::move(l));
}

template <Numeric T>
Matrix<T> Network<T>::forward(const Matrix<T>& input){
	Matrix<T> current = input;
	for(auto& l:layers){
		current = l.forward(current);
	}
	return current;
}

template <Numeric T>
void Network<T>::backward(const Matrix<T>& grad){
	Matrix<T> curr_grad = grad;
	for(auto l=layers.rbegin(); l!=layers.rend(); l++){
		curr_grad = (*l).backward(curr_grad);
	}
}
