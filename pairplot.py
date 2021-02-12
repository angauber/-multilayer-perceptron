import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import abspath
from srcs.data import getDataFrame

def main():
    df = getDataFrame(abspath('dataset/data.csv'))

    plot = sns.pairplot(df, hue = 'diagnosis')
    plot.savefig('plots/pairplot.png')

if __name__ == '__main__':
    main()
