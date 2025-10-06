"""
Generate synthetic tabular data using DataSynthesizer
"""
import argparse
import os
import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator


def generate_synthetic_data(input_file, output_file, num_rows, mode='independent',
                           categorical_threshold=30, histogram_bins=10, 
                           max_parents=2, categorical_cols=None):
    """
    Generate synthetic data from input CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save synthetic CSV
        num_rows: Number of synthetic rows to generate
        mode: 'independent' or 'correlated'
        categorical_threshold: Max unique values to treat as categorical
        histogram_bins: Number of bins for numeric distributions
        max_parents: Max parents in Bayesian network (correlated mode only)
        categorical_cols: List of column names to force as categorical
    """
    # Use temp description file
    description_file = f'.tmp_description_{os.getpid()}.json'
    
    try:
        # Build categorical attributes dict
        categorical_attributes = {}
        if categorical_cols:
            for col in categorical_cols:
                categorical_attributes[col] = True
        
        # Describe the dataset
        describer = DataDescriber(
            category_threshold=categorical_threshold,
            histogram_bins=histogram_bins
        )
        
        if mode == 'independent':
            describer.describe_dataset_in_independent_attribute_mode(
                dataset_file=input_file,
                attribute_to_is_categorical=categorical_attributes,
                attribute_to_is_candidate_key={}
            )
        elif mode == 'correlated':
            describer.describe_dataset_in_correlated_attribute_mode(
                dataset_file=input_file,
                epsilon=0,
                k=max_parents,
                attribute_to_is_categorical=categorical_attributes
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'independent' or 'correlated'")
        
        describer.save_dataset_description_to_file(description_file)
        
        # Generate synthetic dataset
        generator = DataGenerator()
        
        if mode == 'independent':
            generator.generate_dataset_in_independent_mode(num_rows, description_file)
        else:  # correlated
            generator.generate_dataset_in_correlated_attribute_mode(num_rows, description_file)
        
        generator.save_synthetic_data(output_file)
        
    finally:
        # Clean up temp description file
        if os.path.exists(description_file):
            os.remove(description_file)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic tabular data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python synthetic_generator.py input.csv -o synthetic.csv -n 1000
  
  # Specify categorical columns
  python synthetic_generator.py data.csv -o synth.csv -n 500 -c gender smokes is_fit
  
  # Use correlated mode
  python synthetic_generator.py data.csv -o synth.csv -n 1000 -m correlated
        """
    )
    
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-n', '--num-rows', type=int, default=1000,
                       help='Number of synthetic rows to generate (default: 1000)')
    parser.add_argument('-m', '--mode', choices=['independent', 'correlated'],
                       default='independent',
                       help='Generation mode (default: independent)')
    parser.add_argument('-c', '--categorical', nargs='*', metavar='COL',
                       help='Column names to treat as categorical')
    parser.add_argument('-t', '--threshold', type=int, default=30,
                       help='Categorical threshold - columns with ≤N unique values '
                            'treated as categorical (default: 30)')
    parser.add_argument('-b', '--bins', type=int, default=10,
                       help='Number of histogram bins for numeric columns (default: 10)')
    parser.add_argument('-k', '--max-parents', type=int, default=2,
                       help='Max parents in Bayesian network for correlated mode (default: 2)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        parser.error(f"Input file not found: {args.input_file}")
    
    generate_synthetic_data(
        input_file=args.input_file,
        output_file=args.output,
        num_rows=args.num_rows,
        mode=args.mode,
        categorical_threshold=args.threshold,
        histogram_bins=args.bins,
        max_parents=args.max_parents,
        categorical_cols=args.categorical
    )


if __name__ == '__main__':
    main()