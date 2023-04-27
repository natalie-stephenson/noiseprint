import pandas as pd

def split_column_of_tuples(df, column_to_split, list_of_new_column_names):
    df[list_of_new_column_names] = pd.DataFrame(df[column_to_split].tolist(),index=df.index)
    return df