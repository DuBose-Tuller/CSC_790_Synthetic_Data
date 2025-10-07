from pathlib import Path
import numpy as np
from model_evaluation_pipeline import run_three_models_training, test_all_models

TARGET1 = 'sex'
N_RUNS1 = 1

# Original dataset paths
train_file1 = 'data/penguins/original/train.csv'
test_file1 = 'data/penguins/original/test.csv'

# Run multiple times to get statistics
all_runs = []
print(f"Running baseline evaluation on original dataset ({N_RUNS1} runs)...")
for run_idx in range(N_RUNS1):
    # Train models
    training_results = run_three_models_training(train_file1, TARGET1, random_state=None)
    # Test all models
    test_results = test_all_models(training_results, test_file1, TARGET1)
    all_runs.append(test_results)
    print(f"Run {run_idx + 1}/{N_RUNS1} complete")

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


TARGET2 = 'is_fit'
N_RUNS2 = 1

# Original dataset paths
train_file2 = 'data/fitness/original/train.csv'
test_file2 = 'data/fitness/original/test.csv'

# Run multiple times to get statistics
all_runs = []
print(f"Running baseline evaluation on original dataset ({N_RUNS2} runs)...")
for run_idx in range(N_RUNS2):
    # Train models
    training_results = run_three_models_training(train_file2, TARGET2, random_state=None)
    # Test all models
    test_results = test_all_models(training_results, test_file2, TARGET2)
    all_runs.append(test_results)
    print(f"Run {run_idx + 1}/{N_RUNS2} complete")

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
