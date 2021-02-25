import pandas
import numpy as np

def normalize(df: pandas.DataFrame) -> pandas.DataFrame:
    return (df - df.min()) / (df.max() - df.min())

def getDataFrame(path: str) -> pandas.DataFrame:
    diagnosis_converter = lambda diagnosis: 1. if diagnosis == 'M' else 0.
    columns = [index for index in range(1, 32)]
    column_names = ['diagnosis' if col == 1 else 'feature_' + str(col - 1) for col in columns]

    df = pandas.read_csv(path, header = None, index_col = 0, names = column_names, converters = {1: diagnosis_converter})
    df = normalize(df)

    return df

def formatRows(df: pandas.DataFrame) -> list:
    """[P(M), P(B)]"""
    diagnosis = lambda diagnosis: np.array([1., 0.]) if diagnosis == 1. else np.array([0., 1.])

    return [(np.array([x for key, x in row.items() if key is not 'diagnosis']), diagnosis(row['diagnosis'])) for row in df.to_dict('records')]

"""Divide the given dataset in 3 parts of 60,20,20 %"""
def divide(df: pandas.DataFrame, seed: int) -> tuple:
    return tuple([formatRows(dataset) for dataset in np.split(df.sample(frac=1, random_state=seed), [int(.6 * len(df)), int(.8 * len(df))])])
