"""
Exploratory Data Analysis (EDA) for Precision Irrigation Prediction
Generates comprehensive visualizations and statistical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configure visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def perform_eda(df, output_dir='outputs'):
    """
    Perform comprehensive exploratory data analysis.
    
    Parameters:
    - df: DataFrame to analyze
    - output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 100)
    print("EXPLORATORY DATA ANALYSIS - PRECISION IRRIGATION PREDICTION")
    print("=" * 100)
    
    # 1. Dataset Overview
    print("\n1. DATASET OVERVIEW")
    print("-" * 100)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    # 2. Target Variable Distribution
    print("\n2. TARGET VARIABLE DISTRIBUTION - IRRIGATION_NEED")
    print("-" * 100)
    target_dist = df['Irrigation_Need'].value_counts()
    target_pct = df['Irrigation_Need'].value_counts(normalize=True) * 100
    
    print("\nClass Distribution:")
    for class_name in target_dist.index:
        print(f"  {class_name}: {target_dist[class_name]:,} ({target_pct[class_name]:.2f}%)")
    
    # 3. Numerical Features Summary
    print("\n3. NUMERICAL FEATURES SUMMARY STATISTICS")
    print("-" * 100)
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(df[numerical_cols].describe())
    
    # 4. Categorical Features Summary
    print("\n4. CATEGORICAL FEATURES SUMMARY")
    print("-" * 100)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Irrigation_Need':
            print(f"\n{col}:")
            print(df[col].value_counts())
    
    # 5. Correlations
    print("\n5. CORRELATION ANALYSIS")
    print("-" * 100)
    correlation_matrix = df[numerical_cols].corr()
    print("\nHighest correlations with Irrigation_Need:")
    
    # Create numerical encoding for target
    target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df_temp = df.copy()
    df_temp['Irrigation_Need_Encoded'] = df_temp['Irrigation_Need'].map(target_mapping)
    
    target_corr = df_temp[list(numerical_cols) + ['Irrigation_Need_Encoded']].corr()['Irrigation_Need_Encoded'].sort_values(ascending=False)
    print(target_corr.head(10))
    
    # ==================== VISUALIZATIONS ====================
    
    # 1. Target Variable Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    target_dist.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0].set_title('Irrigation Need Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Irrigation Need Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    target_pct.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                    colors=['#2ecc71', '#f39c12', '#e74c3c'], startangle=90)
    axes[1].set_title('Irrigation Need Distribution (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_target_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 01_target_distribution.png")
    plt.close()
    
    # 2. Numerical Features Histograms
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols[:12]):
        axes[idx].hist(df[col], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_numerical_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 02_numerical_distributions.png")
    plt.close()
    
    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 03_correlation_heatmap.png")
    plt.close()
    
    # 4. Categorical Features Distribution
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    categorical_to_plot = [col for col in categorical_cols if col != 'Irrigation_Need'][:6]
    
    for idx, col in enumerate(categorical_to_plot):
        df[col].value_counts().plot(kind='bar', ax=axes[idx], color='#9b59b6', edgecolor='black')
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_categorical_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 04_categorical_distributions.png")
    plt.close()
    
    # 5. Key Features vs Target
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    key_features = ['Rainfall_mm', 'Soil_Moisture', 'Temperature_C', 'Humidity']
    target_order = ['Low', 'Medium', 'High']
    
    for idx, feature in enumerate(key_features):
        row, col = idx // 2, idx % 2
        sns.boxplot(data=df, x='Irrigation_Need', y=feature, ax=axes[row, col], 
                   order=target_order, palette=['#2ecc71', '#f39c12', '#e74c3c'])
        axes[row, col].set_title(f'{feature} vs Irrigation Need', fontweight='bold', fontsize=12)
        axes[row, col].set_xlabel('Irrigation Need')
        axes[row, col].set_ylabel(feature)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_key_features_vs_target.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 05_key_features_vs_target.png")
    plt.close()
    
    # 6. Soil Type vs Irrigation Need
    fig, ax = plt.subplots(figsize=(12, 6))
    soil_irrigation = pd.crosstab(df['Soil_Type'], df['Irrigation_Need'], normalize='index') * 100
    soil_irrigation.plot(kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax.set_title('Soil Type vs Irrigation Need (Percentage)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Soil Type')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Irrigation Need')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_soil_type_vs_target.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 06_soil_type_vs_target.png")
    plt.close()
    
    # 7. Crop Type vs Irrigation Need
    fig, ax = plt.subplots(figsize=(12, 6))
    crop_irrigation = pd.crosstab(df['Crop_Type'], df['Irrigation_Need'], normalize='index') * 100
    crop_irrigation.plot(kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax.set_title('Crop Type vs Irrigation Need (Percentage)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Crop Type')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Irrigation Need')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_crop_type_vs_target.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 07_crop_type_vs_target.png")
    plt.close()
    
    # 8. Statistical Summary Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Rainfall statistics
    rainfall_by_target = [df[df['Irrigation_Need'] == cls]['Rainfall_mm'].values for cls in target_order]
    axes[0].boxplot(rainfall_by_target, labels=target_order, patch_artist=True,
                   boxprops=dict(facecolor='#3498db', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
    axes[0].set_title('Rainfall Distribution by Irrigation Need', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Rainfall (mm)')
    axes[0].set_xlabel('Irrigation Need')
    
    # Soil Moisture statistics
    moisture_by_target = [df[df['Irrigation_Need'] == cls]['Soil_Moisture'].values for cls in target_order]
    axes[1].boxplot(moisture_by_target, labels=target_order, patch_artist=True,
                   boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
    axes[1].set_title('Soil Moisture Distribution by Irrigation Need', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Soil Moisture (%)')
    axes[1].set_xlabel('Irrigation Need')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_statistics_by_target.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 08_statistics_by_target.png")
    plt.close()
    
    # 9. Region vs Irrigation Need
    fig, ax = plt.subplots(figsize=(12, 6))
    region_irrigation = pd.crosstab(df['Region'], df['Irrigation_Need'], normalize='index') * 100
    region_irrigation.plot(kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax.set_title('Region vs Irrigation Need (Percentage)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Region')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Irrigation Need')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_region_vs_target.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 09_region_vs_target.png")
    plt.close()
    
    print("\n" + "=" * 100)
    print(f"EDA Complete! All visualizations saved to '{output_dir}' directory")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    # Load data and perform EDA
    df = pd.read_csv('data/irrigation_prediction.csv')
    perform_eda(df, 'outputs')
