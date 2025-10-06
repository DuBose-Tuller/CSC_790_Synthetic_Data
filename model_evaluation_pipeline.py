import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import os

def run_three_models_training(train_path, target_col, random_state=42):
    """Train three models with k-fold CV on training data only - SILENT."""
    # Load training data only
    train_data = pd.read_csv(train_path).replace('.', np.nan).dropna()
    
    # Separate features and targets
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    # Preprocess training data only
    if y_train.dtype == 'object':
        target_encoder = LabelEncoder()
        y_train_enc = target_encoder.fit_transform(y_train)
    else:
        y_train_enc = y_train.values
        target_encoder = None
    
    X_train_proc = X_train.copy()
    
    # Scale numeric features
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = None
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X_train_proc[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    # Encode categorical features
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_proc[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le
    
    # Store encoders for later use
    encoders = {
        'scaler': scaler,
        'target_encoder': target_encoder,
        'label_encoders': label_encoders
    }
    
    # Initialize models
    models = {
        'logistic': LogisticRegression(random_state=random_state, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
        'hist_gradient': HistGradientBoostingClassifier(random_state=random_state)
    }
    
    # Train models SILENTLY
    roc_scores = {}
    
    for model_name, model in models.items():
        # 5-Fold Cross-Validation on training data
        cv_scores = cross_val_score(model, X_train_proc, y_train_enc, 
                                  cv=5, scoring='roc_auc_ovr' if len(np.unique(y_train_enc)) > 2 else 'roc_auc')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Train final model on full training data
        model.fit(X_train_proc, y_train_enc)
        
        # Store results SILENTLY
        roc_scores[model_name] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'trained_model': model,
            'encoders': encoders
        }
    
    return roc_scores

def test_all_models(training_results, test_path, target_col):
    """Test ALL models from training on test.csv - SILENT."""
    # Load test data
    test_data = pd.read_csv(test_path).replace('.', np.nan).dropna()
    
    # Separate features and targets
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Get encoders from first model (they're all the same)
    first_model = list(training_results.keys())[0]
    encoders = training_results[first_model]['encoders']
    
    # Preprocess test data
    X_test_proc = X_test.copy()
    
    # Encode target
    if encoders['target_encoder']:
        y_test_enc = encoders['target_encoder'].transform(y_test)
    else:
        y_test_enc = y_test.values
    
    # Scale numeric features
    if encoders['scaler']:
        numeric_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
        X_test_proc[numeric_cols] = encoders['scaler'].transform(X_test[numeric_cols])
    
    # Encode categorical features
    for col, le in encoders['label_encoders'].items():
        X_test_proc[col] = le.transform(X_test[col].astype(str))
    
    # Test ALL models and collect ROC scores
    all_test_scores = {}
    
    for model_name, model_data in training_results.items():
        trained_model = model_data['trained_model']
        
        # Predict on test data SILENTLY
        try:
            y_pred_test = trained_model.predict_proba(X_test_proc)
            if y_pred_test.shape[1] == 2:
                test_roc = roc_auc_score(y_test_enc, y_pred_test[:, 1])
            else:
                test_roc = roc_auc_score(y_test_enc, y_pred_test, multi_class='ovr')
        except:
            y_pred_test = trained_model.predict(X_test_proc)
            test_roc = roc_auc_score(y_test_enc, y_pred_test, multi_class='ovr')
        
        all_test_scores[model_name] = {
            'cv_score': model_data['cv_mean'],
            'test_roc': test_roc
        }
    
    return all_test_scores

def evaluate_all_datasets_with_testing(data_dir='data', target_cols=None):
    """Train all datasets, then test best model from each dataset - SILENT until final results."""
    if target_cols is None:
        target_cols = {
            'penguins': 'sex',
            'fitness': 'is_fit'
        }
    
    all_results = {}
    
    for dataset_name in target_cols.keys():
        train_path = os.path.join(data_dir, dataset_name, 'original', 'train.csv')
        test_path = os.path.join(data_dir, dataset_name, 'original', 'test.csv')
        
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            continue
        
        try:
            target_col = target_cols[dataset_name]
            # Train and test ALL models SILENTLY
            training_results = run_three_models_training(train_path, target_col)
            test_results = test_all_models(training_results, test_path, target_col)
            
            all_results[dataset_name] = {
                'training_results': training_results,
                'test_results': test_results
            }
            
        except Exception as e:
            all_results[dataset_name] = None
    
    # ONLY print final test ROC scores for ALL models
    print("TEST ROC SCORES:")
    print("="*50)
    for dataset_name, results in all_results.items():
        if results:
            print(f"\n{dataset_name.upper()}:")
            test_scores = results['test_results']
            for model_name, scores in test_scores.items():
                print(f"  {model_name:15}: {scores['test_roc']:.4f}")
    
    return all_results

# Simple usage - just one function call to get test ROC scores
if __name__ == '__main__':
    # ONE FUNCTION CALL - Get test ROC scores for all datasets and should take train path as flexible arguments
    results = evaluate_all_datasets_with_testing(data_dir='data', target_cols=None)
