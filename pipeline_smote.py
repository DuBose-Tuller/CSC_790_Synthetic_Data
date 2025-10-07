"""
Downsample minority class to varying proportions and oversample with SMOTE to balance.
"""
import argparse
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import numpy as np


def process_minority_proportion(input_file, target_col, target_proportion,
                                output_dir, k_neighbors=5, random_state=None,
                                categorical_cols=None):
    """
    Process a dataset for a specific minority proportion:
    1. Downsample minority class to target proportion
    2. Use SMOTE to oversample minority to balance with majority class
    Args:
        input_file: Path to input CSV
        target_col: Name of target column
        target_proportion: Desired minority class proportion before SMOTE (0-1)
        output_dir: Directory to save output file
        k_neighbors: Number of nearest neighbors for SMOTE (default: 5)
        random_state: Random seed for reproducibility
        categorical_cols: List of categorical column names
    """
    # Load full dataset
    full_df = pd.read_csv(input_file).dropna()
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
    minority_rows_to_keep = minority_rows.sample(n=num_minority_rows_to_keep,
                                                 replace=False,
                                                 random_state=random_state)

    # Combine downsampled minority with majority
    downsampled_df = pd.concat([majority_rows, minority_rows_to_keep], axis=0)

    # Now apply SMOTE to balance the dataset
    X = downsampled_df.drop(columns=[target_col])
    y = downsampled_df[target_col]

    # Identify categorical columns
    if categorical_cols is None:
        categorical_cols = []

    # Also auto-detect object type columns
    auto_categorical = X.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = list(set(categorical_cols + auto_categorical))

    # Encode categorical variables
    encoders = {}
    X_encoded = X.copy()

    for col in categorical_cols:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # Encode target if it's categorical
    y_encoded = y
    target_encoder = None
    if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y.astype(str))

    # Calculate sampling strategy to balance classes
    y_series = pd.Series(y_encoded)
    class_counts = y_series.value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Balance by bringing minority up to majority count
    sampling_strategy = {minority_class: class_counts[majority_class]}

    # Apply SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(k_neighbors, class_counts[minority_class] - 1),
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X_encoded, y_encoded)

    # Decode categorical variables back
    X_decoded = pd.DataFrame(X_resampled, columns=X.columns)

    for col, encoder in encoders.items():
        # Round to nearest integer for categorical variables
        X_decoded[col] = np.round(X_decoded[col]).astype(int)
        # Clip to valid range
        X_decoded[col] = np.clip(X_decoded[col], 0, len(encoder.classes_) - 1)
        # Decode back to original labels
        X_decoded[col] = encoder.inverse_transform(X_decoded[col])

    # Decode target if it was encoded
    if target_encoder is not None:
        y_resampled = np.round(y_resampled).astype(int)
        y_resampled = np.clip(y_resampled, 0, len(target_encoder.classes_) - 1)
        y_resampled = target_encoder.inverse_transform(y_resampled)

    # Combine features and target back together
    balanced_df = X_decoded.copy()
    balanced_df[target_col] = y_resampled

    # Create output filename based on proportion
    proportion_str = f"{int(target_proportion * 100)}pct"
    output_file = os.path.join(output_dir, f"minority_{proportion_str}_smote.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save balanced dataset
    balanced_df.to_csv(output_file, index=False)
    print(f"Created {output_file} (initial minority proportion: {target_proportion:.1%}, "
          f"balanced to 50-50 with SMOTE)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate balanced datasets by downsampling minority then upsampling with SMOTE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single proportion
  python smote_downsample.py data/fitness_dataset.csv -t is_fit -p 0.2

  # Multiple proportions
  python smote_downsample.py data/fitness_dataset.csv -t is_fit -p 0.1 0.15 0.2 0.25

  # Specify categorical columns and random seed
  python smote_downsample.py data.csv -t target -p 0.2 0.3 -c gender smokes -r 42
        """
    )

    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('-t', '--target', required=True,
                       help='Name of target column')
    parser.add_argument('-p', '--proportions', nargs='+', type=float, required=True,
                       help='Target minority proportions before SMOTE (e.g., 0.1 0.2 0.3)')
    parser.add_argument('-o', '--output-dir',
                       help='Output directory (default: same as input file)')
    parser.add_argument('-c', '--categorical', nargs='*', metavar='COL',
                       help='Column names to treat as categorical')
    parser.add_argument('-k', '--k-neighbors', type=int, default=5,
                       help='Number of nearest neighbors for SMOTE (default: 5)')
    parser.add_argument('-r', '--random-state', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')

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
            k_neighbors=args.k_neighbors,
            random_state=args.random_state,
            categorical_cols=args.categorical
        )


if __name__ == '__main__':
    main()
