import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification 
from sklearn.ensemble import RandomForestClassifier   
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


    
def run_model( datapath, target_col, val_size =0.2, test_size=0.2, random_state=42):
    print(target_col)
    ROC_AUCs = []

    # Load the dataset
    data = pd.read_csv(datapath)
    
    # Separate features and target variable
    X = data.drop(columns=[target_col])
    y = data[target_col]
    # if target col has . values, remove those rows
    y = y.replace('.', np.nan)
    X = X.replace('.', np.nan)

    # remove all rows with missing values in features
    X = X.dropna()
    y = y[X.index]  

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # remove missing rows if target label has missing values
    train_data = pd.concat([X_train, y_train], axis=1).dropna(subset=[target_col])
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    test_data = pd.concat([X_test, y_test], axis=1).dropna(subset=[target_col])
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Encode target variable if it's categorical
    if y_train.dtype == 'object':
        target_encoder = LabelEncoder()
        y_train_encoded = target_encoder.fit_transform(y_train)
        y_test_encoded = target_encoder.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test  


    # preprocess the numeric features
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])   

    
    #preprocess categorical features using label encoding

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])


    # perform k -fold cross validation on logistic regression using my train and validation set
    model = LogisticRegression(random_state=random_state)
    # Fit the model on the training set and evaluate on the validation set
    cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", np.mean(cv_scores))

    #train final model on the whole training set
    model.fit(X_train, y_train_encoded)

    # test Roc AUC score
    y_pred = model.predict(X_test)
    # Evaluate the model by f1 score for logistic regression
    print("For Logistic Regression:")
    # Get target names for better readability
    if y_train.dtype == 'object':
        target_names = target_encoder.classes_
        report = classification_report(y_test_encoded, y_pred, target_names=target_names)
    else:
        report = classification_report(y_test_encoded, y_pred)
    print("Classification Report:")
    print(report)
    # area under ROC curve
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_test_encoded, y_pred, multi_class='ovr')
    print("ROC AUC:", roc_auc)
    ROC_AUCs.append(roc_auc)
    print("--------------------------------")
    

    # Random Forest Classifier with k - fold cross validation
    rf_model = RandomForestClassifier(random_state=random_state)
    cv_scores = cross_val_score(rf_model, X_train, y_train_encoded, cv=5)
    print("Random Forest Classifier Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", np.mean(cv_scores))

    # Evaluate the model
    rf_model.fit(X_train, y_train_encoded)
    # Get target names for better readability
    if y_train.dtype == 'object':
        target_names = target_encoder.classes_
        report = classification_report(y_test_encoded, y_pred, target_names=target_names)
    else:
        report = classification_report(y_test_encoded, y_pred)
    print("Classification Report:")
    print(report)
    # area under ROC curve
    roc_auc = roc_auc_score(y_test_encoded, y_pred, multi_class='ovr')
    print("ROC AUC:", roc_auc)
    ROC_AUCs.append(roc_auc)
    print("--------------------------------")


    # HistGradientBoostingClassifier with k - fold cross validation
    hgb_model = HistGradientBoostingClassifier(random_state=random_state)
    cv_scores = cross_val_score(hgb_model, X_train, y_train_encoded, cv=5)
    print("HistGradientBoostingClassifier Cross-validation scores:", cv_scores)
    # Evaluate the model
    # Train final model on the whole training set
    # Get target names for better readability
    if y_train.dtype == 'object':
        target_names = target_encoder.classes_
        report = classification_report(y_test_encoded, y_pred, target_names=target_names)
    else:
        report = classification_report(y_test_encoded, y_pred)
    print("Classification Report:")
    print(report)

    # area under ROC curve

    roc_auc = roc_auc_score(y_test_encoded, y_pred, multi_class='ovr')    

    print("ROC AUC:", roc_auc)
    ROC_AUCs.append(roc_auc)

    print("--------------------------------")   
    return ROC_AUCs
     