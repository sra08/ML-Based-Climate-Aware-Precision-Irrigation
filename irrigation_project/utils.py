"""
Utility functions for irrigation prediction project
Includes preprocessing, encoding, evaluation, and model persistence
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import pickle
import os


def load_data(filepath):
    """Load irrigation dataset from CSV file."""
    return pd.read_csv(filepath)


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in dataset.
    
    Parameters:
    - df: DataFrame
    - strategy: 'mean' for numerical, 'mode' for categorical
    
    Returns:
    - DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    # Handle numerical columns with mean
    numerical_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        if df_copy[col].isnull().sum() > 0:
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
    
    # Handle categorical columns with mode
    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_copy[col].isnull().sum() > 0:
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    
    return df_copy


def encode_categorical_features(df, categorical_cols, encoder_dict=None, fit=False):
    """
    Encode categorical features using LabelEncoder.
    
    Parameters:
    - df: DataFrame
    - categorical_cols: List of categorical column names
    - encoder_dict: Dictionary of fitted encoders (for transform only)
    - fit: Boolean, whether to fit new encoders
    
    Returns:
    - Transformed DataFrame and encoder dictionary
    """
    df_copy = df.copy()
    encoders = encoder_dict if encoder_dict is not None else {}
    
    for col in categorical_cols:
        if fit:
            encoders[col] = LabelEncoder()
            df_copy[col] = encoders[col].fit_transform(df_copy[col].astype(str))
        else:
            if col in encoders:
                df_copy[col] = encoders[col].transform(df_copy[col].astype(str))
    
    return df_copy, encoders


def standardize_features(df, numerical_cols, scaler=None, fit=False):
    """
    Standardize numerical features using StandardScaler.
    
    Parameters:
    - df: DataFrame
    - numerical_cols: List of numerical column names
    - scaler: Fitted StandardScaler object
    - fit: Boolean, whether to fit new scaler
    
    Returns:
    - Transformed DataFrame and scaler object
    """
    df_copy = df.copy()
    
    if fit:
        scaler = StandardScaler()
        df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
    else:
        if scaler is not None:
            df_copy[numerical_cols] = scaler.transform(df_copy[numerical_cols])
    
    return df_copy, scaler


def preprocess_data(df, target_col='Irrigation_Need', encoders=None, scaler=None, fit=False):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    - df: DataFrame
    - target_col: Target column name
    - encoders: Dictionary of fitted encoders
    - scaler: Fitted scaler
    - fit: Boolean, whether to fit new encoders/scaler
    
    Returns:
    - X, y, encoders, scaler
    """
    # Handle missing values
    df_clean = handle_missing_values(df)
    
    # Identify feature and target columns
    y = df_clean[target_col].copy()
    X = df_clean.drop(columns=[target_col]).copy()
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X, encoders = encode_categorical_features(X, categorical_cols, encoders, fit=fit)
    
    # Encode target variable
    if fit:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        if encoders is None:
            encoders = {}
        encoders['target'] = target_encoder
    else:
        if encoders is not None and 'target' in encoders:
            y = encoders['target'].transform(y)
    
    # Identify numerical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Standardize numerical features
    X, scaler = standardize_features(X, numerical_cols, scaler, fit=fit)
    
    return X, y, encoders, scaler


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    Evaluate classification model and return metrics.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - model_name: Name of the model
    
    Returns:
    - Dictionary with evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion_Matrix': cm
    }
    
    return metrics


def print_evaluation_report(metrics_list):
    """
    Print comprehensive evaluation report for multiple models.
    
    Parameters:
    - metrics_list: List of metric dictionaries
    """
    print("\n" + "=" * 100)
    print("MODEL EVALUATION REPORT - PRECISION IRRIGATION PREDICTION")
    print("=" * 100)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': m['Model'],
            'Accuracy': f"{m['Accuracy']:.4f}",
            'Precision': f"{m['Precision']:.4f}",
            'Recall': f"{m['Recall']:.4f}",
            'F1-Score': f"{m['F1-Score']:.4f}"
        }
        for m in metrics_list
    ])
    
    print("\n" + comparison_df.to_string(index=False))
    print("\n" + "=" * 100)
    
    # Print detailed confusion matrices
    print("\nDETAILED CONFUSION MATRICES:\n")
    for metrics in metrics_list:
        print(f"{metrics['Model']}:")
        print(metrics['Confusion_Matrix'])
        print()


def save_model(model, filepath):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath):
    """Load trained model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def save_preprocessing_artifacts(encoders, scaler, filepath):
    """Save encoders and scaler for use in production."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    artifacts = {'encoders': encoders, 'scaler': scaler}
    with open(filepath, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"✓ Preprocessing artifacts saved to {filepath}")


def load_preprocessing_artifacts(filepath):
    """Load encoders and scaler from disk."""
    with open(filepath, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['encoders'], artifacts['scaler']


def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract feature importance from tree-based models.
    
    Parameters:
    - model: Trained model with feature_importances_ attribute
    - feature_names: List of feature names
    - top_n: Number of top features to return
    
    Returns:
    - DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)
    else:
        return None


def calculate_irrigation_recommendation(prediction_class, confidence):
    """
    Generate irrigation recommendation based on prediction.
    
    Parameters:
    - prediction_class: 'Low', 'Medium', or 'High'
    - confidence: Confidence score (0-100)
    
    Returns:
    - Recommendation message
    """
    recommendations = {
        'Low': 'Low irrigation required. Soil moisture is adequate. Monitor rainfall.',
        'Medium': 'Medium irrigation required. Apply regular watering schedule. Monitor soil moisture.',
        'High': 'High irrigation urgent! Severe drought conditions. Increase irrigation frequency immediately.'
    }
    
    return recommendations.get(prediction_class, 'Please consult agronomist')


if __name__ == '__main__':
    print("Utility module for Precision Irrigation Prediction Project")
