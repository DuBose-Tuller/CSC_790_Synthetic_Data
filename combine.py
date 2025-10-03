import pandas as pd

original_df = pd.read_csv("data/fitness_dataset.csv")
synthetic_df = pd.read_csv("data/fitness_synthetic.csv")

combined_df = pd.concat([original_df, synthetic_df], axis=0)
combined_df.to_csv("data/fitness_combined.csv")
