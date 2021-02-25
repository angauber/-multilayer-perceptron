import numpy as np
import argparse
from network import Network
from data import getDataFrame, divide
from os.path import abspath

def predict(dataset_path):
    seed = 565

    df = getDataFrame(abspath(dataset_path))

    _, _, test_data = divide(df, seed)

    del df

    net = Network([30, 10, 10, 2], seed).importNetwork()

    print('Precision on test dataset: {:.2f}'.format(net.evaluate(test_data)))
    print('Cross entropy on test dataset: {:.3f}'.format(net.crossEntropy(test_data)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()

    predict(args.dataset)

if __name__ == '__main__':
    main()
