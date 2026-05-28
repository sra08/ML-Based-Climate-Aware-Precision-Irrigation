"""
ADVANCED_ANALYSIS.PY - Advanced analytical functions
Provides deeper insights into data patterns and model behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def analyze_class_imbalance(df, target_col='Irrigation_Need'):
    """
    Analyze class imbalance in target variable.
    
    Parameters:
    - df: DataFrame
    - target_col: Target column name
    
    Returns:
    - Dictionary with imbalance statistics
    """
    class_counts = df[target_col].value_counts()
    class_ratios = df[target_col].value_counts(normalize=True) * 100
    
    print("\n" + "="*80)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*80)
    
    imbalance_stats = {}
    for class_name in class_counts.index:
        imbalance_stats[class_name] = {
            'count': class_counts[class_name],
            'percentage': class_ratios[class_name],
            'ratio': class_counts[class_name] / len(df)
        }
        print(f"\n{class_name}:")
        print(f"  Count: {class_counts[class_name]:,}")
        print(f"  Percentage: {class_ratios[class_name]:.2f}%")
    
    # Imbalance ratio
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 3:
        print("⚠️  SIGNIFICANT CLASS IMBALANCE DETECTED")
        print("   Recommendation: Consider using class weights or SMOTE")
    else:
        print("✓ Class distribution is relatively balanced")
    
    return imbalance_stats


def analyze_feature_distributions(df, numerical_cols):
    """
    Analyze distribution characteristics of numerical features.
    
    Parameters:
    - df: DataFrame
    - numerical_cols: List of numerical column names
    
    Returns:
    - DataFrame with distribution statistics
    """
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    stats_list = []
    
    for col in numerical_cols:
        data = df[col].dropna()
        
        # Calculate statistics
        mean = data.mean()
        median = data.median()
        std = data.std()
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        stats_list.append({
            'Feature': col,
            'Mean': f'{mean:.2f}',
            'Median': f'{median:.2f}',
            'Std': f'{std:.2f}',
            'Skewness': f'{skewness:.2f}',
            'Kurtosis': f'{kurtosis:.2f}',
            'IQR': f'{iqr:.2f}'
        })
        
        # Interpretation
        print(f"\n{col}:")
        print(f"  Mean: {mean:.2f}, Median: {median:.2f}")
        print(f"  Std Dev: {std:.2f}")
        print(f"  Skewness: {skewness:.2f}", end="")
        if abs(skewness) < 0.5:
            print(" (Approximately symmetric)")
        elif skewness > 0:
            print(" (Right-skewed)")
        else:
            print(" (Left-skewed)")
        
        print(f"  Kurtosis: {kurtosis:.2f}", end="")
        if abs(kurtosis) < 3:
            print(" (Normal-like tails)")
        else:
            print(" (Heavy tails)")
    
    return pd.DataFrame(stats_list)


def analyze_outliers(df, numerical_cols, method='iqr'):
    """
    Detect and analyze outliers in numerical features.
    
    Parameters:
    - df: DataFrame
    - numerical_cols: List of numerical columns
    - method: 'iqr' or 'zscore'
    
    Returns:
    - Dictionary with outlier information
    """
    print("\n" + "="*80)
    print(f"OUTLIER ANALYSIS (Method: {method.upper()})")
    print("="*80)
    
    outlier_info = {}
    
    for col in numerical_cols:
        data = df[col]
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > 3
        
        outlier_count = outliers.sum()
        outlier_percentage = (outlier_count / len(data)) * 100
        
        if outlier_count > 0:
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'values': data[outliers].values[:5]  # First 5 outliers
            }
            
            print(f"\n{col}:")
            print(f"  Outliers found: {outlier_count} ({outlier_percentage:.2f}%)")
            if outlier_percentage > 5:
                print("  ⚠️  Consider investigating or treating outliers")
    
    return outlier_info


def analyze_feature_correlations(df, numerical_cols, target_encoded):
    """
    Deep dive into feature correlations.
    
    Parameters:
    - df: DataFrame
    - numerical_cols: List of numerical columns
    - target_encoded: Encoded target variable
    
    Returns:
    - Correlation analysis results
    """
    print("\n" + "="*80)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*80)
    
    df_temp = df[numerical_cols].copy()
    df_temp['Target'] = target_encoded
    
    correlation_matrix = df_temp.corr()
    
    # High correlations with target
    target_corr = correlation_matrix['Target'].sort_values(ascending=False)
    
    print("\nFeatures most correlated with target:")
    print(target_corr.head(11))
    
    # Multi-collinearity check
    print("\n\nMulti-collinearity Check (Features highly correlated with each other):")
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_val = correlation_matrix.iloc[i, j]
                
                if col1 != 'Target' and col2 != 'Target':
                    high_corr_pairs.append({
                        'Feature 1': col1,
                        'Feature 2': col2,
                        'Correlation': f'{corr_val:.3f}'
                    })
                    print(f"  {col1} <-> {col2}: {corr_val:.3f}")
    
    if len(high_corr_pairs) == 0:
        print("  ✓ No significant multi-collinearity detected")
    else:
        print(f"  ⚠️  {len(high_corr_pairs)} feature pairs show high correlation")
    
    return correlation_matrix


def analyze_feature_importance_stability(models_dict, feature_names):
    """
    Compare feature importance across tree-based models.
    
    Parameters:
    - models_dict: Dictionary of trained models
    - feature_names: List of feature names
    
    Returns:
    - Comparison of feature importances
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE STABILITY ANALYSIS")
    print("="*80)
    
    importance_comparison = pd.DataFrame({'Feature': feature_names})
    
    for model_name, model in models_dict.items():
        if hasattr(model, 'feature_importances_'):
            importance_comparison[model_name] = model.feature_importances_
    
    # Top features by each model
    print("\n\nTop 10 Features by Model:")
    for col in importance_comparison.columns[1:]:
        print(f"\n{col}:")
        top_features = importance_comparison.nlargest(10, col)[['Feature', col]]
        for idx, row in top_features.iterrows():
            print(f"  {row['Feature']}: {row[col]:.4f}")
    
    return importance_comparison


def analyze_prediction_patterns(y_true, y_pred, class_names):
    """
    Analyze prediction patterns and misclassification.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - class_names: List of class names
    
    Returns:
    - Pattern analysis results
    """
    print("\n" + "="*80)
    print("PREDICTION PATTERN ANALYSIS")
    print("="*80)
    
    correct = y_true == y_pred
    accuracy = correct.sum() / len(y_true)
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    print("\nPer-Class Performance:")
    for class_idx, class_name in enumerate(class_names):
        mask = y_true == class_idx
        class_accuracy = correct[mask].sum() / mask.sum() if mask.sum() > 0 else 0
        class_samples = mask.sum()
        print(f"  {class_name}: {class_accuracy:.4f} ({class_samples} samples)")
    
    # Confusion patterns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nMisclassification Patterns:")
    for i, class_name in enumerate(class_names):
        misclassified = cm[i].sum() - cm[i, i]
        if cm[i].sum() > 0:
            misclass_rate = misclassified / cm[i].sum()
            print(f"  {class_name}: {misclassified} misclassified ({misclass_rate:.2%})")


def generate_model_comparison_report(models_metrics_list):
    """
    Generate comprehensive model comparison report.
    
    Parameters:
    - models_metrics_list: List of model metrics dictionaries
    
    Returns:
    - Formatted comparison report
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("="*80)
    
    # Create comparison dataframe
    comparison_data = []
    for metrics in models_metrics_list:
        comparison_data.append({
            'Model': metrics['Model'],
            'Accuracy': f"{metrics['Accuracy']:.4f}",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall']:.4f}",
            'F1-Score': f"{metrics['F1-Score']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Rankings
    print("\n\nRanking Summary:")
    
    metrics_to_rank = {
        'Accuracy': [m['Accuracy'] for m in models_metrics_list],
        'Precision': [m['Precision'] for m in models_metrics_list],
        'Recall': [m['Recall'] for m in models_metrics_list],
        'F1-Score': [m['F1-Score'] for m in models_metrics_list]
    }
    
    for metric_name, values in metrics_to_rank.items():
        ranked_idx = np.argsort(values)[::-1]
        print(f"\n{metric_name}:")
        for rank, idx in enumerate(ranked_idx, 1):
            print(f"  {rank}. {models_metrics_list[idx]['Model']}: {values[idx]:.4f}")


def create_performance_visualization(models_metrics_list, output_path='outputs/'):
    """
    Create comprehensive performance visualization.
    
    Parameters:
    - models_metrics_list: List of metrics
    - output_path: Path to save visualizations
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = [m['Model'] for m in models_metrics_list]
    accuracies = [m['Accuracy'] for m in models_metrics_list]
    precisions = [m['Precision'] for m in models_metrics_list]
    recalls = [m['Recall'] for m in models_metrics_list]
    f1_scores = [m['F1-Score'] for m in models_metrics_list]
    
    x_pos = np.arange(len(model_names))
    
    # Accuracy
    axes[0, 0].bar(x_pos, accuracies, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].set_ylim([0.7, 1.0])
    
    # Precision
    axes[0, 1].bar(x_pos, precisions, color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Model Precision Comparison', fontweight='bold', fontsize=12)
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].set_ylim([0.7, 1.0])
    
    # Recall
    axes[1, 0].bar(x_pos, recalls, color='#f39c12', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Model Recall Comparison', fontweight='bold', fontsize=12)
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].set_ylim([0.7, 1.0])
    
    # F1-Score
    axes[1, 1].bar(x_pos, f1_scores, color='#e74c3c', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Model F1-Score Comparison', fontweight='bold', fontsize=12)
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 1].set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    plt.savefig(f'{output_path}model_comparison_detailed.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: model_comparison_detailed.png")
    plt.close()


def create_learning_curves(model, X_train, y_train, X_val, y_val, model_name):
    """
    Generate learning curves to diagnose bias/variance.
    
    Parameters:
    - model: Trained model
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - model_name: Model name for plotting
    """
    from sklearn.metrics import accuracy_score
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        n_samples = int(len(X_train) * size)
        X_train_subset = X_train[:n_samples]
        y_train_subset = y_train[:n_samples]
        
        model.fit(X_train_subset, y_train_subset)
        
        train_pred = model.predict(X_train_subset)
        val_pred = model.predict(X_val)
        
        train_scores.append(accuracy_score(y_train_subset, train_pred))
        val_scores.append(accuracy_score(y_val, val_pred))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes, train_scores, 'o-', color='#3498db', label='Training Score')
    ax.plot(train_sizes, val_scores, 'o-', color='#e74c3c', label='Validation Score')
    
    ax.set_xlabel('Training Set Size (Fraction)', fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontweight='bold')
    ax.set_title(f'Learning Curves - {model_name}', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/learning_curve_{model_name.lower().replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: learning_curve_{model_name}.png")
    plt.close()


if __name__ == '__main__':
    print("Advanced Analysis Module")
    print("Use functions from this module for detailed data and model analysis")
