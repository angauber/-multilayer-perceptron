import numpy as np
from helper import sigmoid, sigmoidPrime

class Network:
    """layers: array of the number of perceptron per layer
           layers[0] = number of perceptron in the input layer
           layers[-1] = number of perceptron in the output layer"""
    def __init__(self, layers, seed = None):
        np.random.seed(seed)

        self.layers = layers
        """One bias per perceptron in each layer but the input."""
        self.biases = np.array([np.random.randn(i, 1) for i in layers[1:]], dtype=object)
        """n matrices of m weights between each layer with:
               n the number of perceptron in the current layer
               m the number of perceptron in the previous layer"""
        self.weights = np.array([np.random.randn(i, j) for j, i in zip(layers[:-1], layers[1:])], dtype=object)

        """activation matrix"""
        self.activation = np.array([])
        """Activation matrix without the sigmoid"""
        self.z = np.array([])

        self.cross_entropy = []

        self.biases_count = sum(layers[1:])
        self.weights_count = sum([i * j for i, j in zip(layers[:-1], layers[1:])])

    """Compute the activation matrix given the input layer"""
    def feedForward(self, input_layer: np.array):
        activation = [input_layer.reshape(-1, 1)]
        z = []
        for biases, weights in zip(self.biases, self.weights):
            """Compute the layer activation matrix based on the weights and biases"""
            z.append((weights @ activation[-1]) + biases)
            activation.append(sigmoid(z[-1]))

        activation[-1] = self.softmax(z[-1])

        self.activation = np.array(activation, dtype=object)
        self.z = np.array(z, dtype=object)

        return self.activation[-1]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    """Cost function"""
    def computeCost(self, y: np.ndarray) -> np.ndarray:
        y.shape = (-1, 1)

        return self.activation[-1] - y

    """Binary cross entropy"""
    def crossEntropy(self, dataset: list) -> float:
        n = len(dataset)
        dataset = np.array([[self.feedForward(x).flatten(), y] for x, y in dataset])

        p = dataset[:,0]
        y = dataset[:,1]

        return (-1 / y.size) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def getCrossEntropyHistory(self) -> list:
        return self.cross_entropy

    """Compute the percentage of correct predicitons in the given dataset"""
    def evaluate(self, dataset: list):
        predicted = []

        for x, y in dataset:
            predicted.append(1 if np.argmax(y) == np.argmax(self.feedForward(x)) else 0)

        return ((np.array(predicted) == 1).sum() / len(dataset)) * 100

    """Apply gradient descent"""
    def fit(self, train_data: list, validation_data: list, epoch: int, learning_rate: float):
        for e in range(epoch):
            b_delta = np.array([np.zeros(bl.shape) for bl in self.biases], dtype=object)
            w_delta = np.array([np.zeros(wl.shape) for wl in self.weights], dtype=object)

            for x, y in train_data:
                b_grad, w_grad = self.backPropagate(x, y)
                b_delta = [nb + dnb for nb, dnb in zip(b_delta, b_grad)]
                w_delta = [nw + dnw for nw, dnw in zip(w_delta, w_grad)]

            self.weights = np.array([w - (learning_rate / len(train_data)) * nw for w, nw in zip(self.weights, w_delta)], dtype=object)
            self.biases = np.array([b - (learning_rate / len(train_data)) * nb for b, nb in zip(self.biases, b_delta)], dtype=object)

            self.cross_entropy.append(self.crossEntropy(validation_data))

            print('Epoch: [{}/{}] Precision: [{:.2f}%] Cross entropy: {:.3f}'.format(e + 1, epoch, self.evaluate(validation_data), self.cross_entropy[-1]))

        return self

    """Returns a tuple of two matrices corresponding to the gradiant error of weights and biases"""
    def backPropagate(self, x: np.ndarray, y: np.ndarray) -> tuple:
        b_grad = np.array([np.zeros(bl.shape) for bl in self.biases], dtype=object)
        w_grad = np.array([np.zeros(wl.shape) for wl in self.weights], dtype=object)

        """feed forward"""
        self.feedForward(x)

        """Compute the output error matrix and apply it to the last hidden layer wieghts and biases"""
        delta = self.computeCost(y) * sigmoidPrime(self.z[-1])
        b_grad[-1] = delta
        w_grad[-1] = delta @ self.activation[-2].T

        """Propagate the error correction backwards among all the weights and biases"""
        for l in range(len(self.layers) - 2, 0):
            delta = (self.weights[l + 1].T @ delta) * sigmoidPrime(self.z[l])
            b_grad[l] = delta
            w_grad[l] = delta @ self.activation[l - 1]

        return (b_grad, w_grad)

    """Export biases asn weights"""
    def exportNetwork(self):
        net = np.array([self.biases, self.weights])

        np.save('net.npy', net)

        print('Network exported to `net.npy`')

        return self

    """Import biases and weights"""
    def importNetwork(self):
        net = np.load('net.npy', allow_pickle=True)

        self.biases = net[0]
        self.weights = net[1]

        print('Network exported from `net.npy`')

        return self
