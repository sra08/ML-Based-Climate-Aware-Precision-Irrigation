
# 🌱 Precision Irrigation AI — Smart Water Management in a Changing Climate

Precision Irrigation AI is an end-to-end Machine Learning system designed to optimize agricultural water usage. By analyzing real-time soil, weather, crop, and geographic factors, the system predicts irrigation needs (**Low**, **Medium**, or **High**) and provides expert water management recommendations. The system is designed to help farmers adapt to changing climate conditions, minimize water waste, and prevent crop stress.

---

## 🚀 Key Features

* **Interactive Streamlit Dashboard**: User-friendly web interface allowing farmers and agronomists to input field metrics and get instant predictions, model confidence, and actionable recommendations.
* **Multiple ML Classifiers**: Implementation of four machine learning models with automatic hyperparameter tuning:
  * **XGBoost** (Gradient Boosting)
  * **Random Forest** (Ensemble Bagging)
  * **Decision Tree** (Explainable Tree Classifier)
  * **Logistic Regression** (Linear Baseline)
* **Automated Data Preprocessing**: Custom pipelines for handling missing values, encoding categorical variables, and scaling numerical features.
* **Exploratory Data Analysis (EDA)**: Automated script generating statistical summaries and 9 key visual plots (correlations, class distributions, crop/soil type interactions).
* **Advanced Diagnostics**: In-depth analysis of class imbalance, feature importance stability, multicollinearity, outlier detection, and learning curves for diagnosing bias vs. variance.
* **Climate Change Adaptive Rules**: Integrated logic assessing climate stress factors (drought alerts, heatwave evapotranspiration impacts, low humidity warnings).

---

## 📂 Repository Structure

```directory
irrigation_project/
│
├── app.py                      # Streamlit interactive dashboard web application
├── train_models.py             # Model training, hyperparameter tuning & evaluation pipeline
├── eda_analysis.py             # Exploratory Data Analysis (EDA) & visualization generator
├── advanced_analysis.py        # Advanced data diagnostics, learning curves & model validation
├── utils.py                    # Core utility functions (preprocessing, scaling, evaluation, I/O)
├── config.py                   # Centralized configuration (hyperparameters, rules, metadata)
├── smart_irrigation_project.ipynb # Jupyter Notebook for development and prototyping
│
├── data/
│   └── irrigation_prediction.csv # Agricultural dataset (features: soil, weather, crop, region)
│
├── models/                     # Saved preprocessors and trained model pickle files
│   ├── preprocessing_artifacts.pkl # StandardScaler & LabelEncoders
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
└── outputs/                    # Output directory for generated EDA plots and reports
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/irrigation_project.git
cd irrigation_project
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required packages using pip:
```bash
pip install streamlit pandas numpy scikit-learn xgboost scipy matplotlib seaborn
```

---

## 💻 Workflow & Usage

### Step 1: Run Exploratory Data Analysis
Generate descriptive statistics and plots to understand the dataset.
```bash
python eda_analysis.py
```
This saves visualization plots (e.g., target distributions, correlation matrices, crop/soil type breakdowns) in the `outputs/` directory.

### Step 2: Train and Tune Models
Train all four models, perform hyperparameter tuning, compare performance, and save the best-performing models.
```bash
python train_models.py
```
The script outputs a comprehensive model comparison report, plots feature importances, and saves the trained `.pkl` models to the `models/` directory.

### Step 3: Run the Dashboard
Launch the interactive web application to get real-time irrigation recommendations.
```bash
streamlit run app.py
```

---

## 📊 Dataset & Features

The model trains on a dataset containing the following features:

### Soil Properties
* **Soil Type**: Clay, Silt, Sandy, Loamy
* **Soil pH**: Acidity/alkalinity of the soil (4.5 - 8.5)
* **Soil Moisture (%)**: Water content in the soil (5% - 65%)
* **Organic Carbon**: Soil organic carbon content indicator
* **Electrical Conductivity**: Indicates soil salt concentration

### Weather & Climate Factors
* **Temperature (°C)**: Evapotranspiration driver (10°C - 45°C)
* **Humidity (%)**: Air humidity percentage (20% - 95%)
* **Rainfall (mm)**: Expected or historical precipitation (0mm - 2500mm)
* **Sunlight Hours**: Daily duration of sunlight
* **Wind Speed (km/h)**: Wind speed affecting evaporation rate

### Crop & Management
* **Crop Type**: Wheat, Maize, Cotton, Rice, Sugarcane, Potato
* **Growth Stage**: Sowing, Vegetative, Flowering, Harvest
* **Season**: Kharif (monsoon), Rabi (winter), Zaid (summer)
* **Irrigation Type**: Canal, Drip, Sprinkler, Rainfed
* **Water Source**: Groundwater, Reservoir, River, Rainwater
* **Mulching Used**: Yes / No
* **Field Area (hectares)**: Size of the field

### Target Output
* **Irrigation Need**: **Low**, **Medium**, or **High**

---

## 📈 Model Performance & Evaluation

The training pipeline evaluates models using the following metrics:
* **Accuracy**: Overall correct prediction rate.
* **Precision**: Weighted precision to avoid false positives (over-irrigation).
* **Recall (Sensitivity)**: Weighted recall to prevent false negatives (under-irrigation/crop stress).
* **F1-Score**: Harmonic mean of precision and recall.

*Tree-based models (Random Forest and XGBoost) also output feature importances, indicating that factors like **Soil Moisture** and **Rainfall** have the highest predictive weight in deciding irrigation requirements.*

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.
