import pandas

def normalize(df: pandas.DataFrame) -> pandas.DataFrame:
    return (df - df.min()) / (df.max() - df.min())

def getDataFrame(path: str) -> pandas.DataFrame:
    diagnosis_converter = lambda diagnosis: 1. if diagnosis == 'M' else 0.
    columns = [index for index in range(1, 32)]
    column_names = ['diagnosis' if col == 1 else 'feature_' + str(col - 1) for col in columns]

    df = pandas.read_csv(path, header = None, index_col = 0, names = column_names, converters = {1: diagnosis_converter})
    df = normalize(df)

    return df
