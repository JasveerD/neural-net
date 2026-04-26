#pragma once
#include "matrix.hpp"
#include "layer.hpp"
#include "network.hpp"
#include <vector>

// SDG (Stochastic Gradient Descent) with velocity
template <Numeric T>
class SGD{
public:
	T learning_rate; T momentum;

	std::vector<Matrix<T>> velocity_W;
	std::vector<Matrix<T>> velocity_B;
	
	// constructor
	SGD(T learning_rate, T momentum=0.9);

	//updates all weights in the network using stored gradients
	void step(Network<T>& network);
};

template <Numeric T>
SGD<T>::SGD(T learning_rate, T momentum):learning_rate(learning_rate), momentum(momentum){}

template <Numeric T>
void SGD<T>::step(Network<T>& network) {
    // lazy initialize velocities
    if (velocity_W.empty()) {
        for (auto& layer : network.layers) {
            velocity_W.push_back(Matrix<T>::zeros(layer.weight.rows, layer.weight.cols));
            velocity_B.push_back(Matrix<T>::zeros(layer.bias.rows, layer.bias.cols));
        }
    }

    for (size_t i = 0; i < network.layers.size(); i++) {
        auto& layer = network.layers[i];

        // update velocities
        velocity_W[i] = velocity_W[i] * momentum + layer.dW * learning_rate;
        velocity_B[i] = velocity_B[i] * momentum + layer.db * learning_rate;

        // update weights
        layer.weight = layer.weight - velocity_W[i];
        layer.bias   = layer.bias   - velocity_B[i];
    }
}
