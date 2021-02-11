import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def normalize(df):
    return (df-df.min())/(df.max()-df.min())

def main():
    diagnosis_converter = lambda diagnosis: 1. if diagnosis == 'M' else 0.
    columns = [index for index in range(1, 32)]
    column_names = ['diagnosis' if col == 1 else 'feature_' + str(col - 1) for col in columns]

    df = pd.read_csv('data.csv', header = None, index_col = 0, names = column_names, converters = {1: diagnosis_converter})
    df = normalize(df)

    plot = sns.pairplot(df, hue = 'diagnosis')
    plot.savefig('pairplot.png')

if __name__ == '__main__':
    main()
