import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_data_split(datapath, val_size=0.2, test_size=0.2, random_state=42):
    # Use the data directory path
    data_dir = '/home/sahau24/csc790project/Fall2025_DCAI/CSC_790_Synthetic_Data/data'

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            # Get dataset name without extension
            dataset_name = os.path.splitext(filename)[0]
            
            # Full path to the CSV file
            full_csv_path = os.path.join(data_dir, filename)
            
            # Load the data
            data = pd.read_csv(full_csv_path)
            print(f"Processing {filename} with shape {data.shape}")
            
            # make this output directory if it does not exist
            output_dir = f"data/{dataset_name}/original/"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
            
            train_path = os.path.join(output_dir, 'train.csv')
            test_path = os.path.join(output_dir, 'test.csv')
            
            # Clean data (optional - remove rows with missing values)
            # data = data.dropna()  # Uncomment to remove all missing values
            
            # Split entire dataset into train and test
            try:
                train_data, test_data = train_test_split(
                    data, test_size=test_size, random_state=random_state
                )
                
                train_data.to_csv(train_path, index=False)
                test_data.to_csv(test_path, index=False)
                
                print(f"Saved train/test data for {dataset_name}:")
                print(f"  Train: {train_path} ({len(train_data)} rows)")
                print(f"  Test: {test_path} ({len(test_data)} rows)")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue



if __name__ == "__main__":

    datapath = '/home/sahau24/csc790project/Fall2025_DCAI/CSC_790_Synthetic_Data/data'
    train_test_data_split(datapath)


