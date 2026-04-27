#include "activations.hpp"
#include "dataloader.hpp"
#include "layer.hpp"
#include "matrix.hpp"
#include "network.hpp"
#include "optimizer.hpp"
#include "serializer.hpp"
#include "loss.hpp"
#include <iostream>

int main() {

	Layer<float> data_layer(784, 128, relu_mat<float>,relu_derivative_mat<float>);
	Layer<float> d_layer1(128, 64, relu_mat<float>, relu_derivative_mat<float>);
	Layer<float> d_layer2(64, 64, relu_mat<float>, relu_derivative_mat<float>);
	Layer<float> out_layer(64, 10, softmax_mat<float>,softmax_derivative_mat<float>);

	Network<float> net;
	net.add_layer(std::move(data_layer));
	net.add_layer(std::move(d_layer1));
	net.add_layer(std::move(d_layer2));
	net.add_layer(std::move(out_layer));

	SGD<float> optimizer(0.01f, 0.9f);
	DataLoader<float> train_loader("data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte",32); // batch size
	DataLoader<float> test_loader("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte", 32);

	const int epochs = 20;

	for (int epoch = 0; epoch < epochs; epoch++) {
		train_loader.reset();
		float total_loss = 0.0f;
		int n_batches = 0;

		Matrix<float> batch_images, batch_labels;

		while (train_loader.next_batch(batch_images, batch_labels)) {
			Matrix<float> output = net.forward(batch_images);
			float loss = cross_entropy_loss(output, batch_labels);
			Matrix<float> grad = cross_entropy_derivative(output, batch_labels);
			net.backward(grad);
			optimizer.step(net);

		   total_loss += loss;
		   n_batches++;
		}

		std::cout << "epoch " << epoch + 1 << "/" << epochs << "  loss: " << total_loss / n_batches << "\n";
	}

	// accuracy evaluation
	test_loader.reset();
	int correct = 0;
	int total = 0;

	Matrix<float> test_images, test_labels;
	while (test_loader.next_batch(test_images, test_labels)) {
		Matrix<float> output = net.forward(test_images);

		for (size_t i = 0; i < output.rows; i++) {
			// find predicted digit — argmax of output row
			size_t predicted = 0;
			for (size_t j = 1; j < output.cols; j++)
				if (output(i, j) > output(i, predicted))
					predicted = j;

			// find true digit — argmax of label row
			size_t actual = 0;
			for (size_t j = 1; j < test_labels.cols; j++)
				if (test_labels(i, j) > test_labels(i, actual))
					actual = j;

			if (predicted == actual) correct++;
			total++;
		}
	}

	std::cout << "test accuracy: " << (float)correct / total * 100.0f << "%\n";

	// save the model
	save(net, "model.bin");
	std::cout << "model saved to model.bin\n";

	return 0;
}
