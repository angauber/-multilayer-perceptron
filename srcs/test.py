from network import Network
import numpy as np

def test():
    seed = 667
    np.random.seed(seed)

    layers = [30, 10, 10, 2]
    net = Network(layers, seed)

    input_layer = np.random.rand(layers[0])

    output_layer = net.feedForward(input_layer)

    print(output_layer)
    print(np.argmax(output_layer))

if __name__ == '__main__':
    test()
