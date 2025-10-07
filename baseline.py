from pathlib import Path
import numpy as np
from model_evaluation_pipeline import run_three_models_training, test_all_models

TARGET = 'is_fit'
N_RUNS = 20

# Original dataset paths
train_file = 'data/fitness/original/train.csv'
test_file = 'data/fitness/original/test.csv'

# Run multiple times to get statistics
all_runs = []
print(f"Running baseline evaluation on original dataset ({N_RUNS} runs)...")
for run_idx in range(N_RUNS):
    # Train models
    training_results = run_three_models_training(train_file, TARGET)
    # Test all models
    test_results = test_all_models(training_results, test_file, TARGET)
    all_runs.append(test_results)
    print(f"Run {run_idx + 1}/{N_RUNS} complete")

# Calculate statistics for each model
model_names = ['logistic', 'random_forest', 'hist_gradient']
model_display_names = {
    'logistic': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'hist_gradient': 'Histogram Gradient Boosting'
}

print("\n" + "="*70)
print("BASELINE RESULTS - Original Dataset")
print("="*70)
print(f"{'Model':<30} {'Mean ROC AUC':<15} {'Std Dev':<15}")
print("-"*70)

for model_name in model_names:
    roc_scores = [run[model_name]['test_roc'] for run in all_runs]
    mean_roc = np.mean(roc_scores)
    std_roc = np.std(roc_scores)

    display_name = model_display_names[model_name]
    print(f"{display_name:<30} {mean_roc:<15.4f} {std_roc:<15.4f}")

print("="*70)
