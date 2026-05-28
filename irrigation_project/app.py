"""
Streamlit Application for Precision Irrigation Prediction
Interactive dashboard for predicting irrigation needs based on environmental factors
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from utils import (
    load_preprocessing_artifacts, load_model, 
    calculate_irrigation_recommendation, preprocess_data
)

# Configure Streamlit
st.set_page_config(
    page_title="Precision Irrigation AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for agriculture theme
st.markdown("""
    <style>
    .main-header {
        color: #1e7e34;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        color: #2d5016;
        font-size: 1.3em;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid;
        margin: 20px 0;
        font-size: 1.1em;
        font-weight: bold;
    }
    .low-irrigation {
        background-color: #d4edda;
        border-color: #2ecc71;
        color: #1e7e34;
    }
    .medium-irrigation {
        background-color: #fff3cd;
        border-color: #f39c12;
        color: #856404;
    }
    .high-irrigation {
        background-color: #f8d7da;
        border-color: #e74c3c;
        color: #721c24;
    }
    .recommendation-box {
        padding: 15px;
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        border-radius: 5px;
        margin: 15px 0;
        font-size: 1.05em;
    }
    .metrics-container {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .confidence-high {
        color: #2ecc71;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_artifacts():
    """Load all trained models and preprocessing artifacts."""
    try:
        encoders, scaler = load_preprocessing_artifacts("models/preprocessing_artifacts.pkl")
        
        models = {
            'Logistic Regression': load_model("models/logistic_regression_model.pkl"),
            'Decision Tree': load_model("models/decision_tree_model.pkl"),
            'Random Forest': load_model("models/random_forest_model.pkl"),
            'XGBoost': load_model("models/xgboost_model.pkl")
        }
        
        return models, encoders, scaler
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please run: python train_models.py")
        return None, None, None


def prepare_input_data(input_dict, encoders, scaler):
    """Convert user input to properly formatted feature array."""
    # Create DataFrame from input
    df_input = pd.DataFrame([input_dict])
    
    # Get feature order (same as training)
    feature_cols = list(encoders.keys())
    feature_cols.remove('target')
    
    # Encode categorical features
    for col in feature_cols:
        if col in encoders and col != 'target':
            try:
                df_input[col] = encoders[col].transform(df_input[col].astype(str))
            except:
                st.warning(f"Could not encode {col}. Using first category.")
                df_input[col] = 0
    
    # Select only numerical columns for scaling
    numerical_cols = df_input.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Scale features
    if scaler is not None:
        df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
    
    return df_input[numerical_cols].values


def get_prediction_confidence(model, input_features):
    """Calculate confidence score for prediction."""
    try:
        probabilities = model.predict_proba(input_features)[0]
        confidence = np.max(probabilities) * 100
        return confidence
    except:
        return 0


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🌱 Precision Irrigation AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Smart Water Management in a Changing Climate</div>', unsafe_allow_html=True)
    
    # Load models
    models, encoders, scaler = load_models_and_artifacts()
    
    if models is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Prediction Settings")
    selected_model = st.sidebar.selectbox(
        "Select Model for Prediction:",
        list(models.keys()),
        help="Choose the ML model to use for irrigation prediction"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 About This Application")
    st.sidebar.markdown("""
    This AI-powered tool predicts irrigation needs based on:
    - **Soil Properties**: Moisture, pH, conductivity
    - **Climate Factors**: Temperature, humidity, rainfall
    - **Crop Information**: Type, growth stage, season
    - **Farm Management**: Water source, irrigation type
    
    The model considers climate change impacts including:
    - Drought conditions
    - Extreme temperatures
    - Rainfall variations
    - Seasonal patterns
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📋 Field Information")
        
        # Soil Properties
        st.markdown("#### 🌍 Soil Properties")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            soil_type = st.selectbox(
                "Soil Type",
                ["Clay", "Silt", "Sandy", "Loamy"],
                help="Type of soil in the field"
            )
        
        with col_b:
            soil_ph = st.slider(
                "Soil pH",
                min_value=4.5, max_value=8.5, value=6.5, step=0.1,
                help="Acidity/alkalinity of soil (4.5-8.5 normal range)"
            )
        
        with col_c:
            soil_moisture = st.slider(
                "Soil Moisture (%)",
                min_value=5, max_value=65, value=35, step=1,
                help="Percentage of water in soil"
            )
        
        col_d, col_e, col_f = st.columns(3)
        
        with col_d:
            organic_carbon = st.slider(
                "Organic Carbon",
                min_value=0.3, max_value=1.6, value=0.8, step=0.1,
                help="Soil organic carbon content"
            )
        
        with col_e:
            electrical_conductivity = st.slider(
                "Electrical Conductivity",
                min_value=0.2, max_value=3.5, value=1.5, step=0.1,
                help="Soil salt content indicator"
            )
        
        with col_f:
            field_area = st.slider(
                "Field Area (hectares)",
                min_value=0.3, max_value=15.0, value=5.0, step=0.5,
                help="Size of agricultural field"
            )
        
        # Climate Factors
        st.markdown("#### 🌤️ Climate Factors")
        col_g, col_h, col_i = st.columns(3)
        
        with col_g:
            temperature = st.slider(
                "Temperature (°C)",
                min_value=10, max_value=45, value=28, step=1,
                help="Current temperature affecting evapotranspiration"
            )
        
        with col_h:
            humidity = st.slider(
                "Humidity (%)",
                min_value=20, max_value=95, value=60, step=1,
                help="Air humidity percentage"
            )
        
        with col_i:
            rainfall = st.slider(
                "Rainfall (mm)",
                min_value=0, max_value=2500, value=500, step=50,
                help="Recent rainfall or expected precipitation"
            )
       
        col_j, col_k, col_l = st.columns(3)

        with col_j:
            sunlight = st.slider(
                "Sunlight Hours",
                min_value=0.0, max_value=12.0, value=6.0, step=0.5,
                help="Daily sunlight hours"
            )

        with col_k:
            wind_speed = st.slider(
                "Wind Speed (km/h)",
                min_value=0, max_value=20, value=8, step=1,
                help="Wind speed (affects evaporation)"
            )

        with col_l:
            previous_irrigation = st.slider(
                "Previous Irrigation (mm)",
                min_value=0, max_value=120, value=50, step=5,
                help="Last irrigation amount"
            )

        
        # Crop Information
        st.markdown("#### 🌾 Crop Information")
        col_m, col_n, col_o = st.columns(3)
        
        with col_m:
            crop_type = st.selectbox(
                "Crop Type",
                ["Wheat", "Maize", "Cotton", "Rice", "Sugarcane", "Potato"],
                help="Type of crop being grown"
            )
        
        with col_n:
            growth_stage = st.selectbox(
                "Crop Growth Stage",
                ["Sowing", "Vegetative", "Flowering", "Harvest"],
                help="Current growth stage of crop"
            )
        
        with col_o:
            season = st.selectbox(
                "Season",
                ["Kharif", "Rabi", "Zaid"],
                help="Agricultural season in the region"
            )
        
        # Irrigation Management
        st.markdown("#### 💧 Irrigation Management")
        col_p, col_q, col_r = st.columns(3)
        
        with col_p:
            irrigation_type = st.selectbox(
                "Irrigation Type",
                ["Canal", "Drip", "Sprinkler", "Rainfed"],
                help="Method of irrigation used"
            )
        
        with col_q:
            water_source = st.selectbox(
                "Water Source",
                ["Groundwater", "Reservoir", "River", "Rainwater"],
                help="Primary water source for irrigation"
            )
        
        with col_r:
            mulching = st.selectbox(
                "Mulching Used",
                ["Yes", "No"],
                help="Whether mulching is practiced in the field"
            )
        
        # Geographic Information
        st.markdown("#### 📍 Geographic Information")
        col_s, col_t = st.columns(2)
        
        with col_s:
            region = st.selectbox(
                "Region",
                ["North", "South", "East", "West", "Central"],
                help="Geographic region of the field"
            )
        
        with col_t:
            st.empty()
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        # Prepare input for prediction
        input_dict = {
            'Soil_Type': soil_type,
            'Soil_pH': soil_ph,
            'Soil_Moisture': soil_moisture,
            'Organic_Carbon': organic_carbon,
            'Electrical_Conductivity': electrical_conductivity,
            'Temperature_C': temperature,
            'Humidity': humidity,
            'Rainfall_mm': rainfall,
            'Sunlight_Hours': sunlight,
            'Wind_Speed_kmh': wind_speed,
            'Crop_Type': crop_type,
            'Crop_Growth_Stage': growth_stage,
            'Season': season,
            'Irrigation_Type': irrigation_type,
            'Water_Source': water_source,
            'Field_Area_hectare': field_area,
            'Mulching_Used': mulching,
            'Previous_Irrigation_mm': previous_irrigation,
            'Region': region
        }
        
        # Make prediction
        try:
            input_features = prepare_input_data(input_dict, encoders, scaler)
            model = models[selected_model]
            
            # Get prediction and probability
            prediction_encoded = model.predict(input_features)[0]
            confidence = get_prediction_confidence(model, input_features)
            
            # Decode prediction
            target_encoder = encoders['target']
            prediction_class = target_encoder.inverse_transform([prediction_encoded])[0]
            
            # Display prediction
            st.markdown("#### 🎯 Irrigation Prediction")
            
            # Determine styling class
            if prediction_class == 'Low':
                style_class = 'low-irrigation'
                emoji = '✅'
            elif prediction_class == 'Medium':
                style_class = 'medium-irrigation'
                emoji = '⚠️'
            else:
                style_class = 'high-irrigation'
                emoji = '🚨'
            
            st.markdown(f"""
            <div class="prediction-box {style_class}">
                {emoji} <strong>Irrigation Need: {prediction_class}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Score
            st.markdown("#### 📈 Model Confidence")
            
            if confidence >= 80:
                conf_class = 'confidence-high'
                conf_text = 'Very High'
            elif confidence >= 60:
                conf_class = 'confidence-medium'
                conf_text = 'Good'
            else:
                conf_class = 'confidence-low'
                conf_text = 'Moderate'
            
            st.markdown(f"""
            <div class="metrics-container">
                <strong>Prediction Confidence:</strong><br>
                <span class="{conf_class}">{confidence:.1f}% ({conf_text})</span><br>
                <strong>Selected Model:</strong> {selected_model}
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation
            st.markdown("#### 💡 Expert Recommendation")
            recommendation = calculate_irrigation_recommendation(prediction_class, confidence)
            
            st.markdown(f"""
            <div class="recommendation-box">
                {recommendation}
            </div>
            """, unsafe_allow_html=True)
            
            # Additional Insights
            st.markdown("#### 📌 Key Factors Analysis")
            
            col_x, col_y = st.columns(2)
            
            with col_x:
                st.metric(
                    "💧 Soil Moisture Level",
                    f"{soil_moisture}%",
                    delta=f"{'Adequate' if soil_moisture > 30 else 'Low'}"
                )
            
            with col_y:
                st.metric(
                    "🌧️ Recent Rainfall",
                    f"{rainfall} mm",
                    delta=f"{'Good' if rainfall > 500 else 'Low'}"
                )
            
            col_z1, col_z2 = st.columns(2)
            
            with col_z1:
                st.metric(
                    "🌡️ Temperature",
                    f"{temperature}°C",
                    delta=f"{'High' if temperature > 35 else 'Normal'}"
                )
            
            with col_z2:
                st.metric(
                    "💨 Humidity",
                    f"{humidity}%",
                    delta=f"{'High' if humidity > 70 else 'Low'}"
                )
            
            # Climate Impact Analysis
            st.markdown("#### 🌍 Climate Impact Analysis")
            
            climate_factors = []
            if temperature > 35:
                climate_factors.append("🔴 High temperature increases water demand")
            if humidity < 40:
                climate_factors.append("🔴 Low humidity increases evaporation")
            if rainfall < 300:
                climate_factors.append("🔴 Low rainfall - drought conditions")
            if soil_moisture < 25:
                climate_factors.append("🔴 Low soil moisture - critical condition")
            
            if not climate_factors:
                st.success("✅ Favorable climate conditions for moderate irrigation")
            else:
                for factor in climate_factors:
                    st.warning(factor)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 30px;">
        <strong>Precision Irrigation AI</strong> | Leveraging Machine Learning for Sustainable Agriculture<br>
        Developed to optimize water usage and adapt to climate change impacts
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
