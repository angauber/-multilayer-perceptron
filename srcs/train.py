import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from network import Network
from data import getDataFrame, divide

def plotCrossEntropy(entropy_history: list):
    entropy_df = pd.DataFrame({'cross_entropy': entropy_history})
    entropy_df['epoch'] = range(len(entropy_history))

    sns.set_style('dark')
    sns.lineplot(data=entropy_df, x='epoch', y='cross_entropy')

    plt.show()

def train(dataset_path: str, plot: bool):
    seed = 351
    np.random.seed(seed)

    df = getDataFrame(dataset_path)
    train_data, validation_data, test_data = divide(df, seed)

    del df

    net = Network([30, 10, 10, 2], seed)
    net.fit(train_data, validation_data, 1000, 3).exportNetwork()

    if (plot):
        plotCrossEntropy(net.getCrossEntropyHistory())
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--plot',  action='store_true')
    args = parser.parse_args()
    
    train(args.dataset, args.plot)

if __name__ == '__main__':
    main()
