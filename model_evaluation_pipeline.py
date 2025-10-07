import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold   # ← switched to StratifiedKFold
from sklearn.metrics import roc_auc_score


def _transform_with_fallback(le: LabelEncoder, s: pd.Series) -> np.ndarray:
    """Map unseen labels to the first known class to avoid transform errors."""
    known = set(le.classes_)
    if not known:
        return le.transform(s.astype(str))
    fallback = le.classes_[0]
    s2 = s.astype(str).map(lambda x: x if x in known else fallback)
    return le.transform(s2)

def _make_stratified_cv(y, max_splits=5, seed=42):
    """
    Build a StratifiedKFold whose n_splits ≤ min_class_count.
    If any class has <2 samples, return None (CV not meaningful).
    """
    y = np.asarray(y)
    # counts for present classes only
    unique, counts = np.unique(y, return_counts=True)
    min_class = counts.min() if counts.size else 0
    if min_class < 2:
        return None
    n_splits = min(max_splits, int(min_class))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# -----------------------
# Training
# -----------------------

def run_three_models_training(train_path, target_col, random_state=42):
    """Train three models with Stratified K-Fold CV (binary target) - SILENT."""
    # Load training data
    train_data = pd.read_csv(train_path).replace('.', np.nan).dropna()

    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]

    # Encode target (binary)
    if y_train.dtype == 'object' or str(y_train.dtype).startswith('category'):
        target_encoder = LabelEncoder()
        y_train_enc = target_encoder.fit_transform(y_train.astype(str))
    else:
        y_train_enc = y_train.values
        target_encoder = None

    # Preprocess X
    X_train_proc = X_train.copy()
    numeric_cols = X_train_proc.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    scaler = None
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X_train_proc[numeric_cols] = scaler.fit_transform(X_train_proc[numeric_cols])

    categorical_cols = X_train_proc.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_proc[col] = le.fit_transform(X_train_proc[col].astype(str))
        label_encoders[col] = le

    encoders = {
        'scaler': scaler,
        'target_encoder': target_encoder,
        'label_encoders': label_encoders
    }

    # Models (unchanged; LR a bit sturdier)
    models = {
        'logistic': LogisticRegression(
            random_state=random_state, max_iter=10000,
            solver='liblinear', class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            random_state=random_state, n_estimators=100, n_jobs=-1
        ),
        'hist_gradient': HistGradientBoostingClassifier(
            random_state=random_state
        )
    }

    # Stratified CV with safe n_splits
    cv_obj = _make_stratified_cv(y_train_enc, max_splits=5, seed=random_state)

    roc_scores = {}
    for model_name, model in models.items():
        if cv_obj is None:
            # Not enough minority samples to do stratified CV
            cv_mean, cv_std = np.nan, np.nan
            model.fit(X_train_proc, y_train_enc)
        else:
            # With stratified folds we can safely use built-in 'roc_auc'
            cv_scores = cross_val_score(
                model,
                X_train_proc,
                y_train_enc,
                cv=cv_obj,
                scoring='roc_auc',      # ← binary AUC
                n_jobs=-1,
                error_score=np.nan      # ← safety net
            )
            cv_mean = float(np.nanmean(cv_scores)) if not np.all(np.isnan(cv_scores)) else np.nan
            cv_std  = float(np.nanstd(cv_scores))  if not np.all(np.isnan(cv_scores)) else np.nan
            model.fit(X_train_proc, y_train_enc)

        roc_scores[model_name] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'trained_model': model,
            'encoders': encoders
        }

    return roc_scores


# -----------------------
# Testing
# -----------------------

def test_all_models(training_results, test_path, target_col):
    """Test ALL models on binary target - SILENT."""
    test_data = pd.read_csv(test_path).replace('.', np.nan).dropna()
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    first_model = list(training_results.keys())[0]
    encoders = training_results[first_model]['encoders']

    # Encode target
    if encoders['target_encoder']:
        y_test_enc = _transform_with_fallback(encoders['target_encoder'], y_test.astype(str))
    else:
        y_test_enc = y_test.values

    # Preprocess X
    X_test_proc = X_test.copy()
    if encoders['scaler']:
        numeric_cols = X_test_proc.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        if len(numeric_cols) > 0:
            X_test_proc[numeric_cols] = encoders['scaler'].transform(X_test_proc[numeric_cols])

    for col, le in encoders['label_encoders'].items():
        if col in X_test_proc.columns:
            X_test_proc[col] = _transform_with_fallback(le, X_test_proc[col].astype(str))

    all_test_scores = {}
    for model_name, model_data in training_results.items():
        trained_model = model_data['trained_model']

        if len(np.unique(y_test_enc)) < 2:
            test_roc = np.nan
        else:
            try:
                if hasattr(trained_model, "predict_proba"):
                    proba = trained_model.predict_proba(X_test_proc)
                    if proba.shape[1] == 1:   # trained as single-class (edge)
                        test_roc = np.nan
                    else:
                        test_roc = roc_auc_score(y_test_enc, proba[:, 1])
                elif hasattr(trained_model, "decision_function"):
                    dec = trained_model.decision_function(X_test_proc)
                    if dec.ndim > 1:
                        dec = dec.ravel()
                    p1 = 1.0 / (1.0 + np.exp(-dec))
                    test_roc = roc_auc_score(y_test_enc, p1)
                else:
                    p1 = trained_model.predict(X_test_proc).astype(float)
                    test_roc = roc_auc_score(y_test_enc, p1)
            except Exception:
                test_roc = np.nan

        all_test_scores[model_name] = {
            'cv_score': model_data['cv_mean'],
            'test_roc': test_roc
        }

    return all_test_scores

# -----------------------

def evaluate_all_datasets_with_testing(data_dir='data', target_cols=None):
    """Train and test all datasets - binary targets only."""
    if target_cols is None:
        target_cols = {
            'penguins': 'sex',
            'fitness': 'is_fit'
        }

    all_results = {}

    for dataset_name, target_col in target_cols.items():
        train_path = os.path.join(data_dir, dataset_name, 'original', 'train.csv')
        test_path = os.path.join(data_dir, dataset_name, 'original', 'test.csv')

        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            continue

        try:
            training_results = run_three_models_training(train_path, target_col)
            test_results = test_all_models(training_results, test_path, target_col)
            all_results[dataset_name] = {
                'training_results': training_results,
                'test_results': test_results
            }
        except Exception:
            all_results[dataset_name] = None

    # Final print
    print("TEST ROC SCORES (Binary):")
    print("="*50)
    for dataset_name, results in all_results.items():
        if results:
            print(f"\n{dataset_name.upper()}:")
            test_scores = results['test_results']
            for model_name, scores in test_scores.items():
                val = scores['test_roc']
                print(f"  {model_name:15}: {'NaN' if (val is None or (isinstance(val, float) and np.isnan(val))) else f'{val:.4f}'}")

    return all_results


if __name__ == '__main__':
    results = evaluate_all_datasets_with_testing(data_dir='data', target_cols=None)