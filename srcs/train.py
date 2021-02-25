import argparse
import numpy as np
from network import Network
from data import getDataFrame, divide
from os.path import abspath


def train(dataset_path):
    seed = 565
    np.random.seed(seed)

    df = getDataFrame(abspath(dataset_path))
    train_data, validation_data, test_data = divide(df, seed)

    del df

    net = Network([30, 10, 10, 2], seed)
    net.fit(train_data, validation_data, 1000, 4).exportNetwork()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    
    train(args.dataset)

if __name__ == '__main__':
    main()
