"""
Split dataset by minority class proportions, generate synthetic data, and combine.
"""
import argparse
import os
import pandas as pd
from utils.gen_datasynthesizer import generate_synthetic_data


def process_minority_proportion(input_file, target_col, target_proportion,
                                output_dir, categorical_cols=None, mode='independent'):
    """
    Process a dataset for a specific minority proportion:
    1. Downsample minority class to target proportion
    2. Generate synthetic data from minority samples
    3. Combine with majority class to create balanced dataset

    Args:
        input_file: Path to input CSV
        target_col: Name of target column
        target_proportion: Desired minority class proportion (0-1)
        output_dir: Directory to save output file
        categorical_cols: List of categorical column names
        mode: Synthesis mode ('independent' or 'correlated')
    """
    # Create temp file paths
    pid = os.getpid()
    minority_temp = f'.tmp_minority_{pid}.csv'
    synthetic_temp = f'.tmp_synthetic_{pid}.csv'

    try:
        # Load full dataset
        full_df = pd.read_csv(input_file)
        p = target_proportion

        # Infer minority class (less frequent value in binary classification)
        value_counts = full_df[target_col].value_counts()
        if len(value_counts) != 2:
            raise ValueError(f"Target column must be binary, found {len(value_counts)} unique values")
        minority_value = value_counts.idxmin()

        # Split minority and majority
        minority_rows = full_df[full_df[target_col] == minority_value]
        majority_rows = full_df[full_df[target_col] != minority_value]

        # Calculate number of minority rows to keep
        num_minority_rows_to_keep = int((p * len(majority_rows)) / (1 - p))
        minority_rows_to_keep = minority_rows.sample(n=num_minority_rows_to_keep, replace=False)

        # Save minority rows temporarily
        minority_rows_to_keep.to_csv(minority_temp, index=False)

        # Calculate how many synthetic rows needed to balance
        num_synthetic_needed = len(majority_rows) - num_minority_rows_to_keep

        # Generate synthetic data from minority samples
        generate_synthetic_data(
            input_file=minority_temp,
            output_file=synthetic_temp,
            num_rows=num_synthetic_needed,
            mode=mode,
            categorical_cols=categorical_cols
        )

        # Load synthetic data
        synthetic_df = pd.read_csv(synthetic_temp)

        # Drop 'Unnamed: 0' column if present (artifact from DataSynthesizer)
        if 'Unnamed: 0' in synthetic_df.columns:
            synthetic_df = synthetic_df.drop(columns=['Unnamed: 0'])

        # Combine: majority + kept minority + synthetic minority
        combined_df = pd.concat([majority_rows, minority_rows_to_keep, synthetic_df], axis=0)

        # Create output filename based on proportion
        proportion_str = f"{int(target_proportion * 100)}pct"
        output_file = os.path.join(output_dir, f"minority_{proportion_str}.csv")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save combined dataset
        combined_df.to_csv(output_file, index=False)
        print(f"Created {output_file} (minority proportion: {target_proportion:.1%})")

    finally:
        # Clean up temporary files
        for temp_file in [minority_temp, synthetic_temp]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def main():
    parser = argparse.ArgumentParser(
        description='Generate datasets with varying minority class proportions using synthetic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single proportion
  python split.py data/fitness_dataset.csv -t is_fit -p 0.2

  # Multiple proportions
  python split.py data/fitness_dataset.csv -t is_fit -p 0.1 0.15 0.2 0.25

  # Specify categorical columns and correlated mode
  python split.py data.csv -t target -p 0.2 0.3 -c gender smokes -m correlated
        """
    )

    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('-t', '--target', required=True,
                       help='Name of target column')
    parser.add_argument('-p', '--proportions', nargs='+', type=float, required=True,
                       help='Target minority proportions (e.g., 0.1 0.2 0.3)')
    parser.add_argument('-o', '--output-dir',
                       help='Output directory (default: same as input file)')
    parser.add_argument('-c', '--categorical', nargs='*', metavar='COL',
                       help='Column names to treat as categorical')
    parser.add_argument('--mode', choices=['independent', 'correlated'],
                       default='independent',
                       help='Synthesis mode (default: independent)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        parser.error(f"Input file not found: {args.input_file}")

    # Validate proportions
    for p in args.proportions:
        if not 0 < p < 1:
            parser.error(f"Proportion must be between 0 and 1, got: {p}")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.input_file)
        if not output_dir:
            output_dir = '.'

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Process each proportion
    for proportion in args.proportions:
        process_minority_proportion(
            input_file=args.input_file,
            target_col=args.target,
            target_proportion=proportion,
            output_dir=output_dir,
            categorical_cols=args.categorical,
            mode=args.mode
        )


if __name__ == '__main__':
    main()
