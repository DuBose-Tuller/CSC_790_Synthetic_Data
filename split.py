import pandas as pd

PATH = "data/fitness_dataset.csv"
TARGET = "is_fit"
MINORITY_VALUE = 1
TARGET_MINORITY_PROPORITON = 0.2

full_df = pd.read_csv(PATH)
p = TARGET_MINORITY_PROPORITON
minority_rows = full_df[full_df[TARGET] == MINORITY_VALUE]

num_minority_rows_to_keep = int( (p * (len(full_df) - len(minority_rows))) / (1 - p) )
minority_rows_to_keep = minority_rows.sample(n=num_minority_rows_to_keep, replace=False)

minority_rows_to_keep.to_csv("data/fitness_minority_rows.csv")