"""
CONFIG.PY - Configuration file for hyperparameter tuning
Centralized configuration for all model parameters and settings
"""

# ============================================================================
# MODEL HYPERPARAMETER CONFIGURATION
# ============================================================================

# LOGISTIC REGRESSION PARAMETERS
LOGISTIC_REGRESSION_PARAMS = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [200, 500],
    'random_state': 42,
    'cv_folds': 5
}

LOGISTIC_REGRESSION_CONFIG = {
    'name': 'Logistic Regression',
    'type': 'linear',
    'search_type': 'GridSearchCV',
    'n_jobs': -1,
    'verbose': 1
}

# DECISION TREE PARAMETERS
DECISION_TREE_PARAMS = {
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'random_state': 42,
    'cv_folds': 5
}

DECISION_TREE_CONFIG = {
    'name': 'Decision Tree Classifier',
    'type': 'tree',
    'search_type': 'GridSearchCV',
    'n_jobs': -1,
    'verbose': 1
}

# RANDOM FOREST PARAMETERS
RANDOM_FOREST_PARAMS = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'random_state': 42,
    'cv_folds': 5
}

RANDOM_FOREST_CONFIG = {
    'name': 'Random Forest Classifier',
    'type': 'ensemble',
    'search_type': 'RandomizedSearchCV',
    'n_iter': 20,
    'n_jobs': -1,
    'verbose': 1,
    'random_state': 42
}

# XGBOOST PARAMETERS
XGBOOST_PARAMS = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.5],
    'random_state': 42,
    'cv_folds': 5
}

XGBOOST_CONFIG = {
    'name': 'XGBoost Classifier',
    'type': 'gradient_boosting',
    'search_type': 'RandomizedSearchCV',
    'n_iter': 20,
    'n_jobs': -1,
    'verbose': 1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    'data_path': 'data/irrigation_prediction.csv',
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'target_column': 'Irrigation_Need'
}

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURE_CONFIG = {
    'numerical_features': [
        'Soil_pH',
        'Soil_Moisture',
        'Organic_Carbon',
        'Electrical_Conductivity',
        'Temperature_C',
        'Humidity',
        'Rainfall_mm',
        'Sunlight_Hours',
        'Wind_Speed_kmh',
        'Field_Area_hectare',
        'Previous_Irrigation_mm'
    ],
    'categorical_features': [
        'Soil_Type',
        'Crop_Type',
        'Crop_Growth_Stage',
        'Season',
        'Irrigation_Type',
        'Water_Source',
        'Mulching_Used',
        'Region'
    ]
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

PREPROCESSING_CONFIG = {
    'missing_value_strategy': 'mean',  # 'mean', 'median', 'mode', 'drop'
    'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
    'encoding_method': 'label',  # 'label', 'onehot'
    'handle_outliers': False,
    'outlier_method': 'iqr'  # 'iqr', 'zscore'
}

# ============================================================================
# IRRIGATION NEED MAPPING
# ============================================================================

IRRIGATION_CLASSES = {
    'Low': {
        'code': 0,
        'description': 'Low irrigation required',
        'color': '#2ecc71',
        'emoji': '✅'
    },
    'Medium': {
        'code': 1,
        'description': 'Medium irrigation required',
        'color': '#f39c12',
        'emoji': '⚠️'
    },
    'High': {
        'code': 2,
        'description': 'High irrigation required',
        'color': '#e74c3c',
        'emoji': '🚨'
    }
}

# ============================================================================
# RECOMMENDATION RULES
# ============================================================================

IRRIGATION_RECOMMENDATIONS = {
    'Low': [
        'Low irrigation required. Soil moisture is adequate. Monitor rainfall.',
        'Rainfall is sufficient. Reduce irrigation frequency.',
        'Favorable soil moisture conditions. Water only when needed.',
        'Excellent weather conditions. Minimize irrigation.',
    ],
    'Medium': [
        'Medium irrigation required. Apply regular watering schedule.',
        'Moderate water demand. Schedule irrigation accordingly.',
        'Monitor soil moisture closely and adjust irrigation as needed.',
        'Balanced water requirements. Follow standard irrigation calendar.',
    ],
    'High': [
        'High irrigation urgent! Severe drought conditions. Increase frequency immediately.',
        'Critical water deficit. Implement intensive irrigation schedule.',
        'Emergency irrigation required due to drought. Act immediately.',
        'Severe water stress. Maximize irrigation to prevent crop loss.',
    ]
}

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

PERFORMANCE_THRESHOLDS = {
    'excellent': {'min': 0.90, 'label': 'Excellent'},
    'very_good': {'min': 0.85, 'label': 'Very Good'},
    'good': {'min': 0.80, 'label': 'Good'},
    'acceptable': {'min': 0.70, 'label': 'Acceptable'},
    'poor': {'min': 0.0, 'label': 'Poor'}
}

# ============================================================================
# CONFIDENCE SCORE INTERPRETATION
# ============================================================================

CONFIDENCE_INTERPRETATION = {
    'very_high': {'min': 0.90, 'label': 'Very High', 'color': '#2ecc71'},
    'high': {'min': 0.80, 'label': 'High', 'color': '#3498db'},
    'moderate': {'min': 0.60, 'label': 'Moderate', 'color': '#f39c12'},
    'low': {'min': 0.0, 'label': 'Low', 'color': '#e74c3c'}
}

# ============================================================================
# FILE PATHS
# ============================================================================

FILE_PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'outputs_dir': 'outputs',
    'logs_dir': 'logs',
    'data_file': 'data/irrigation_prediction.csv',
    'preprocessing_artifacts': 'models/preprocessing_artifacts.pkl',
    'models': {
        'logistic_regression': 'models/logistic_regression_model.pkl',
        'decision_tree': 'models/decision_tree_model.pkl',
        'random_forest': 'models/random_forest_model.pkl',
        'xgboost': 'models/xgboost_model.pkl'
    }
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'log_file': 'logs/irrigation_prediction.log'
}

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

STREAMLIT_CONFIG = {
    'page_title': 'Precision Irrigation AI',
    'page_icon': '🌱',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'theme': {
        'primaryColor': '#1e7e34',
        'backgroundColor': '#f0f2f6',
        'secondaryBackgroundColor': '#d4edda',
        'textColor': '#262730'
    }
}

# ============================================================================
# FEATURE RANGES FOR VALIDATION
# ============================================================================

FEATURE_RANGES = {
    'Soil_pH': {'min': 4.0, 'max': 9.0, 'optimal': (6.0, 7.5)},
    'Soil_Moisture': {'min': 0, 'max': 100, 'optimal': (30, 60)},
    'Organic_Carbon': {'min': 0, 'max': 5.0, 'optimal': (1.0, 3.0)},
    'Electrical_Conductivity': {'min': 0, 'max': 10.0, 'optimal': (0.5, 2.0)},
    'Temperature_C': {'min': -10, 'max': 50, 'optimal': (15, 35)},
    'Humidity': {'min': 0, 'max': 100, 'optimal': (50, 80)},
    'Rainfall_mm': {'min': 0, 'max': 3000, 'optimal': (500, 1500)},
    'Sunlight_Hours': {'min': 0, 'max': 24, 'optimal': (6, 12)},
    'Wind_Speed_kmh': {'min': 0, 'max': 50, 'optimal': (5, 15)},
    'Field_Area_hectare': {'min': 0.1, 'max': 100, 'optimal': (2, 10)},
    'Previous_Irrigation_mm': {'min': 0, 'max': 200, 'optimal': (30, 80)}
}

# ============================================================================
# CLIMATE IMPACT FACTORS
# ============================================================================

CLIMATE_FACTORS = {
    'high_temperature': {'threshold': 35, 'impact': 'Increases water demand'},
    'low_rainfall': {'threshold': 300, 'impact': 'Drought conditions'},
    'high_humidity': {'threshold': 80, 'impact': 'Reduces evaporation'},
    'low_humidity': {'threshold': 40, 'impact': 'Increases evaporation'},
    'low_soil_moisture': {'threshold': 25, 'impact': 'Critical water stress'},
    'high_wind_speed': {'threshold': 15, 'impact': 'Increases evaporation'}
}

# ============================================================================
# SEASONAL ADJUSTMENTS
# ============================================================================

SEASONAL_ADJUSTMENTS = {
    'Kharif': {
        'description': 'Monsoon season (June-October)',
        'rainfall_factor': 1.0,
        'evapotranspiration_factor': 0.9
    },
    'Rabi': {
        'description': 'Winter season (October-March)',
        'rainfall_factor': 0.6,
        'evapotranspiration_factor': 0.7
    },
    'Zaid': {
        'description': 'Summer season (March-June)',
        'rainfall_factor': 0.3,
        'evapotranspiration_factor': 1.2
    }
}

# ============================================================================
# CROP-SPECIFIC WATER REQUIREMENTS
# ============================================================================

CROP_WATER_REQUIREMENTS = {
    'Wheat': {
        'low_threshold': 400,
        'high_threshold': 600,
        'optimal': 500,
        'season': 'Rabi'
    },
    'Maize': {
        'low_threshold': 500,
        'high_threshold': 800,
        'optimal': 600,
        'season': 'Kharif'
    },
    'Cotton': {
        'low_threshold': 600,
        'high_threshold': 1000,
        'optimal': 800,
        'season': 'Kharif'
    },
    'Rice': {
        'low_threshold': 800,
        'high_threshold': 1200,
        'optimal': 1000,
        'season': 'Kharif'
    },
    'Sugarcane': {
        'low_threshold': 1200,
        'high_threshold': 1800,
        'optimal': 1500,
        'season': 'Zaid'
    },
    'Potato': {
        'low_threshold': 300,
        'high_threshold': 500,
        'optimal': 400,
        'season': 'Rabi'
    }
}

# ============================================================================
# SOIL-SPECIFIC ADJUSTMENTS
# ============================================================================

SOIL_WATER_HOLDING_CAPACITY = {
    'Clay': {
        'water_holding_capacity': 'High',
        'drainage': 'Poor',
        'irrigation_frequency': 'Low',
        'irrigation_depth': 'Medium'
    },
    'Silt': {
        'water_holding_capacity': 'Medium',
        'drainage': 'Moderate',
        'irrigation_frequency': 'Medium',
        'irrigation_depth': 'Medium'
    },
    'Sandy': {
        'water_holding_capacity': 'Low',
        'drainage': 'Excellent',
        'irrigation_frequency': 'High',
        'irrigation_depth': 'Shallow'
    },
    'Loamy': {
        'water_holding_capacity': 'Medium',
        'drainage': 'Good',
        'irrigation_frequency': 'Medium',
        'irrigation_depth': 'Medium'
    }
}

# ============================================================================
# MODEL SELECTION RULES
# ============================================================================

MODEL_SELECTION_RULES = {
    'logistic_regression': {
        'pros': ['Fast training', 'Interpretable', 'Good for linear relationships'],
        'cons': ['May underfit', 'Assumes linearity'],
        'use_case': 'Quick baseline model'
    },
    'decision_tree': {
        'pros': ['Easy to interpret', 'Handles non-linear', 'Feature importance'],
        'cons': ['Can overfit', 'Biased to dominant features'],
        'use_case': 'Explainability important'
    },
    'random_forest': {
        'pros': ['Robust', 'High accuracy', 'Reduces overfitting'],
        'cons': ['Less interpretable', 'Slower prediction'],
        'use_case': 'Good balance of performance'
    },
    'xgboost': {
        'pros': ['Highest accuracy', 'Handles imbalance', 'Fast training'],
        'cons': ['More complex', 'Hyperparameter tuning needed'],
        'use_case': 'Best overall performance'
    }
}

# ============================================================================
# VALIDATION RULES
# ============================================================================

VALIDATION_RULES = {
    'min_samples': 100,
    'max_missing_percentage': 30,
    'min_feature_count': 5,
    'max_feature_count': 100,
    'min_classes': 2,
    'max_classes': 20
}

# ============================================================================
# ALERT THRESHOLDS
# ============================================================================

ALERT_THRESHOLDS = {
    'critical_drought': {
        'soil_moisture': 15,
        'rainfall': 100,
        'action': 'Immediate irrigation required'
    },
    'moderate_drought': {
        'soil_moisture': 25,
        'rainfall': 250,
        'action': 'Schedule irrigation within 24 hours'
    },
    'watch_conditions': {
        'soil_moisture': 35,
        'rainfall': 400,
        'action': 'Monitor conditions closely'
    }
}

# ============================================================================
# VERSION AND METADATA
# ============================================================================

PROJECT_METADATA = {
    'name': 'Precision Irrigation Prediction',
    'version': '1.0.0',
    'author': 'Data Science Team',
    'description': 'ML system for predicting irrigation needs considering climate change',
    'created_date': '2024',
    'last_updated': '2024',
    'license': 'MIT',
    'python_version': '3.8+',
    'tensorflow_version': 'N/A',
    'sklearn_version': '1.3.0',
    'xgboost_version': '2.0.0'
}

if __name__ == '__main__':
    print("Configuration file loaded successfully")
    print(f"Project: {PROJECT_METADATA['name']}")
    print(f"Version: {PROJECT_METADATA['version']}")
