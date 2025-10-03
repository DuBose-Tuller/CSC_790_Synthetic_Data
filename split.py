import pandas as pd

full_df = pd.read_csv("fitness_combined.csv")
target = "is_fit"
minority_value = 1

minority_rows = full_df[full_df[target] == minority_value]
print(len(minority_rows) / len(full_df))

# minority_rows.to_csv("fitness_minority_rows.csv")