from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file
import pandas as pd

def generate_synthetic_data():
    # input dataset
    input_data = 'fitness_dataset.csv'
    description_file = 'description.json'
    synthetic_data = 'synthetic_data.csv'
    
    # Configuration
    threshold_value = 30
    histogram_bins = 10  # Correct parameter name
    num_tuples_to_generate = 1000

    # Explicitly specify which columns are categorical
    categorical_attributes = {
        'smokes': True,
        'gender': True,
        'is_fit': True
    }
    
    candidate_keys = {}  # None for this dataset

    # Describe the dataset - independent mode
    describer = DataDescriber(
        category_threshold=threshold_value,
        histogram_bins=histogram_bins 
    )
    describer.describe_dataset_in_independent_attribute_mode(
        dataset_file=input_data,
        attribute_to_is_categorical=categorical_attributes,
        attribute_to_is_candidate_key=candidate_keys
    )
    describer.save_dataset_description_to_file(description_file)

    # Generate synthetic dataset - note the method name difference!
    generator = DataGenerator()
    generator.generate_dataset_in_independent_mode(  # Different from describe method name
        num_tuples_to_generate, 
        description_file
    )
    generator.save_synthetic_data(synthetic_data)

    # Compare results
    input_df = pd.read_csv(input_data, skipinitialspace=True)
    synthetic_df = pd.read_csv(synthetic_data)

if __name__ == '__main__':
    generate_synthetic_data()