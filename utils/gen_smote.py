"""
Generate synthetic tabular data using SMOTE (Synthetic Minority Over-sampling Technique)
"""
import argparse
import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


def generate_synthetic_data(input_file, output_file, num_rows,
                           target_column, k_neighbors=5, random_state=None,
                           categorical_cols=None):
    """
    Generate synthetic data from input CSV file using SMOTE.

    Args:
        input_file: Path to input CSV file
        output_file: Path to save synthetic CSV
        num_rows: Number of synthetic rows to generate
        target_column: Name of the target/classification column
        k_neighbors: Number of nearest neighbors for SMOTE (default: 5)
        random_state: Random seed for reproducibility (default: None)
        categorical_cols: List of column names to treat as categorical
    """
    # Read the input data
    df = pd.read_csv(input_file).dropna()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

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

    # Determine sampling strategy to generate desired number of synthetic samples
    y_series = pd.Series(y_encoded)
    class_counts = y_series.value_counts()
    minority_class = class_counts.idxmin()

    # Calculate how many samples of minority class we need to reach num_rows total
    target_minority = class_counts[minority_class] + num_rows

    sampling_strategy = {minority_class: target_minority}

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
    df_resampled = X_decoded.copy()
    df_resampled[target_column] = y_resampled

    # Save synthetic dataset
    df_resampled.to_csv(output_file, index=False)
    print(f"Generated {len(df_resampled)} total rows ({len(df_resampled) - len(df)} synthetic) and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic tabular data using SMOTE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python smote_script.py input.csv -o synthetic.csv -n 1000 -t target_class

  # Specify k neighbors and random seed
  python smote_script.py data.csv -o synth.csv -n 500 -t label -k 3 -r 42
        """
    )

    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-n', '--num-rows', type=int, default=1000,
                       help='Number of synthetic rows to generate (default: 1000)')
    parser.add_argument('-t', '--target', required=True,
                       help='Name of the target/classification column')
    parser.add_argument('-c', '--categorical', nargs='*', metavar='COL',
                       help='Column names to treat as categorical (auto-detects text columns)')
    parser.add_argument('-k', '--k-neighbors', type=int, default=5,
                       help='Number of nearest neighbors for SMOTE (default: 5)')
    parser.add_argument('-r', '--random-state', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        parser.error(f"Input file not found: {args.input_file}")

    generate_synthetic_data(
        input_file=args.input_file,
        output_file=args.output,
        num_rows=args.num_rows,
        target_column=args.target,
        k_neighbors=args.k_neighbors,
        random_state=args.random_state,
        categorical_cols=args.categorical
    )


if __name__ == '__main__':
    main()
