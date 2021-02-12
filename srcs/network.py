import numpy as np
from helper import sigmoid

class Network:
    """layers: array of the number of perceptron per layer
           layers[0] = number of perceptron in the input layer
           layers[-1] = number of perceptron in the output layer"""
    def __init__(self, layers, seed = None):
        np.random.seed(seed)

        self.layers = layers
        """One bias per perceptron in each layer but the input."""
        self.biases = np.array([np.random.randn(i, 1) for i in layers[1:]])
        """n matrices of m weights between each layer with:
               n the number of perceptron in the current layer
               m the number of perceptron in the previous layer"""
        self.weights = np.array([np.random.randn(i, j) for j, i in zip(layers[:-1], layers[1:])])

        self.biases_count = sum(layers[1:])
        self.weights_count = sum([i * j for i, j in zip(layers[:-1], layers[1:])])

    """Compute the output matrix givent the input one"""
    def feedForward(self, input_layer):
        activation = np.array(input_layer).reshape(-1, 1)
        for biases, weights in zip(self.biases, self.weights):
            """Compute the layer activation matrix based on the weights and biases"""
            activation = sigmoid(np.dot(weights, activation) + biases)

        return activation

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

# with a, the value of the neuron, l the layer, b the bias of the perceptron
# a(l) = sigmoid(w(l) * a(l - 1) + b(l))
