"""
Model Training Script for Precision Irrigation Prediction
Trains 4 classification models with hyperparameter tuning and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

from utils import (
    load_data, preprocess_data, evaluate_model, 
    print_evaluation_report, save_model, save_preprocessing_artifacts,
    get_feature_importance
)


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and tune Logistic Regression."""
    print("\n" + "=" * 100)
    print("TRAINING: LOGISTIC REGRESSION")
    print("=" * 100)
    
    # Hyperparameter tuning
    params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [200, 500]
    }
    
    lr = LogisticRegression(random_state=42)
    grid = GridSearchCV(lr, params, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"✓ Best Parameters: {grid.best_params_}")
    print(f"✓ Best CV Score: {grid.best_score_:.4f}")
    
    # Evaluation
    y_pred = grid.best_estimator_.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, 'Logistic Regression')
    
    return grid.best_estimator_, metrics


def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train and tune Decision Tree Classifier."""
    print("\n" + "=" * 100)
    print("TRAINING: DECISION TREE CLASSIFIER")
    print("=" * 100)
    
    # Hyperparameter tuning
    params = {
        'max_depth': [5, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(dt, params, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"✓ Best Parameters: {grid.best_params_}")
    print(f"✓ Best CV Score: {grid.best_score_:.4f}")
    
    # Evaluation
    y_pred = grid.best_estimator_.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, 'Decision Tree')
    
    return grid.best_estimator_, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and tune Random Forest Classifier."""
    print("\n" + "=" * 100)
    print("TRAINING: RANDOM FOREST CLASSIFIER")
    print("=" * 100)
    
    # Hyperparameter tuning
    params = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = RandomizedSearchCV(rf, params, n_iter=20, cv=5, n_jobs=-1, 
                             random_state=42, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"✓ Best Parameters: {grid.best_params_}")
    print(f"✓ Best CV Score: {grid.best_score_:.4f}")
    
    # Evaluation
    y_pred = grid.best_estimator_.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, 'Random Forest')
    
    return grid.best_estimator_, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and tune XGBoost Classifier."""
    print("\n" + "=" * 100)
    print("TRAINING: XGBOOST CLASSIFIER")
    print("=" * 100)
    
    # Hyperparameter tuning
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.5]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    grid = RandomizedSearchCV(xgb, params, n_iter=20, cv=5, n_jobs=-1, 
                             random_state=42, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"✓ Best Parameters: {grid.best_params_}")
    print(f"✓ Best CV Score: {grid.best_score_:.4f}")
    
    # Evaluation
    y_pred = grid.best_estimator_.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, 'XGBoost')
    
    return grid.best_estimator_, metrics


def main():
    """Main training pipeline."""
    print("\n" + "=" * 100)
    print("PRECISION IRRIGATION PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 100)
    
    # 1. Load Data
    print("\n1. LOADING DATA...")
    df = load_data('data/irrigation_prediction.csv')
    print(f"✓ Data loaded: {df.shape}")
    
    # 2. Preprocess Data
    print("\n2. PREPROCESSING DATA...")
    X, y, encoders, scaler = preprocess_data(df, target_col='Irrigation_Need', fit=True)
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Class distribution: {np.bincount(y)}")
    
    # 3. Train-Test Split
    print("\n3. SPLITTING DATA...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # 4. Train Models
    print("\n4. TRAINING MODELS WITH HYPERPARAMETER TUNING...")
    
    models = []
    metrics_list = []
    
    # Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    models.append(('Logistic Regression', lr_model))
    metrics_list.append(lr_metrics)
    
    # Decision Tree
    dt_model, dt_metrics = train_decision_tree(X_train, y_train, X_test, y_test)
    models.append(('Decision Tree', dt_model))
    metrics_list.append(dt_metrics)
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    models.append(('Random Forest', rf_model))
    metrics_list.append(rf_metrics)
    
    # XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    models.append(('XGBoost', xgb_model))
    metrics_list.append(xgb_metrics)
    
    # 5. Model Comparison
    print("\n5. MODEL COMPARISON...")
    print_evaluation_report(metrics_list)
    
    # 6. Identify Best Model
    best_idx = np.argmax([m['F1-Score'] for m in metrics_list])
    best_model_name, best_model = models[best_idx]
    print(f"\n✓ BEST MODEL: {best_model_name} (F1-Score: {metrics_list[best_idx]['F1-Score']:.4f})")
    
    # 7. Feature Importance (for tree-based models)
    print("\n6. FEATURE IMPORTANCE ANALYSIS...")
    feature_names = X.columns.tolist()
    
    for model_name, model in models:
        importance_df = get_feature_importance(model, feature_names, top_n=15)
        if importance_df is not None:
            print(f"\nTop Features - {model_name}:")
            print(importance_df)
    
    # 8. Save Models and Artifacts
    print("\n7. SAVING MODELS AND ARTIFACTS...")
    for model_name, model in models:
        filepath = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
        save_model(model, filepath)
    
    save_preprocessing_artifacts(encoders, scaler, "models/preprocessing_artifacts.pkl")
    
    # 9. Summary
    print("\n" + "=" * 100)
    print("TRAINING COMPLETE!")
    print("=" * 100)
    print(f"\n✓ {len(models)} models trained and saved")
    print(f"✓ Best performing model: {best_model_name}")
    print(f"✓ Preprocessing artifacts saved")
    print("\nModels ready for deployment in Streamlit app!")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
