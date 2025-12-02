import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle
import os


# DEBUG: Show what files Streamlit Cloud sees
st.write("### 🔍 Debug Info")
st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir('.'))
pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
st.write(f"Found {len(pkl_files)} .pkl files:", pkl_files)
st.write("---")

# Set page config
st.set_page_config(
    page_title="Pesticide & Respiratory Health ML Demo",
    page_icon="🌾",
    layout="wide"
)

# Title and description
st.title("🌾 Pesticide Exposure & Respiratory Health Predictor")
st.markdown("""
This demo showcases machine learning models predicting respiratory health outcomes 
based on agricultural pesticide exposure data.
""")

# Load or train models FIRST (before using available_models)
@st.cache_resource
def load_models():
    """
    Load pre-trained models from pickle files.
    Each model may have been trained on different features.
    """
    models = {}
    model_info = {}
    
    model_files = {
        'Linear Regression': {
            'model': 'linear_regression.pkl',
            'info': 'linear_regression_info.pkl'
        },
        'Random Forest': {
            'model': 'random_forest.pkl',
            'info': 'random_forest_info.pkl'
        },
        'XGBoost': {
            'model': 'xgboost.pkl',
            'info': 'xgboost_info.pkl'
        }
    }
    
    for model_name, files in model_files.items():
        try:
            # Load the model
            with open(files['model'], 'rb') as f:
                models[model_name] = pickle.load(f)
            
            # Load model info (features, metrics, etc.)
            with open(files['info'], 'rb') as f:
                model_info[model_name] = pickle.load(f)
            
            print(f"✓ Loaded {model_name}")
        except FileNotFoundError:
            st.warning(f"⚠️ {model_name} files not found. Model will not be available.")
            continue
    
    return models, model_info

# Try to load models
try:
    models, model_info = load_models()
    available_models = list(models.keys())
    
    if not available_models:
        st.error("❌ No models found! Please train and save your models first.")
        st.info("""
        Run your analysis script with the model saving code to generate:
        - linear_regression.pkl & linear_regression_info.pkl
        - random_forest.pkl & random_forest_info.pkl  
        - xgboost.pkl & xgboost_info.pkl
        """)
        st.stop()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Creating demo models for testing...")
    
    # Create placeholder models for demo
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    model_info = {
        'Linear Regression': {
            'feature_names': [
                'Total_Pest_Intensity_Lag_5', 'Farm_Density_Lag_5',
                'Median AQI', 'Poverty_AllAges_Percent_Est', 'pct_65_plus',
                'pct_Latino', 'pct_Black', 'pct_Asian', 'pct_AI/AN'
            ],
            'description': 'Demo model - 5-year lag with full demographic controls',
            'metrics': {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
        },
        'Random Forest': {
            'feature_names': [
                'Total_Pest_Intensity_Lag_5', 'Farm_Density_Lag_5',
                'Median AQI', 'Poverty_AllAges_Percent_Est', 'pct_65_plus',
                'pct_Latino'
            ],
            'description': 'Demo model - customize with your actual features',
            'metrics': {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
        },
        'XGBoost': {
            'feature_names': [
                'Total_Pest_Intensity_Lag_5', 'Farm_Density_Lag_5',
                'Median AQI', 'Poverty_AllAges_Percent_Est'
            ],
            'description': 'Demo model - customize with your actual features',
            'metrics': {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
        }
    }
    available_models = list(models.keys())

# Sidebar for model selection and parameters
st.sidebar.header("Model Configuration")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    available_models
)

# Display model info
if model_choice in model_info:
    st.sidebar.markdown(f"**Features used:** {len(model_info[model_choice]['feature_names'])}")
    st.sidebar.markdown(f"*{model_info[model_choice].get('description', '')}*")
    
    # Show metrics if available
    if 'metrics' in model_info[model_choice]:
        metrics = model_info[model_choice]['metrics']
        if metrics.get('r2', 0) > 0:  # Only show if real metrics exist
            st.sidebar.metric("R² Score", f"{metrics['r2']:.3f}")
            st.sidebar.metric("RMSE", f"{metrics['rmse']:.2f}")
            st.sidebar.metric("MAE", f"{metrics['mae']:.2f}")

# Feature input section
st.header("📊 Input Features")

# Get features for the selected model
selected_features = model_info[model_choice]['feature_names']

st.info(f"**{model_choice}** uses {len(selected_features)} features. Enter values for each below:")

# Dynamic feature collection
feature_values = {}

# Create columns for organized input
col1, col2 = st.columns(2)

# Define input widgets for each possible feature
feature_inputs = {
    # ===== PESTICIDE FEATURES (5-year lag) =====
    'Total_Pest_Intensity_Lag_5': {
        'label': 'Total Pesticide Intensity (lbs/acre) - 5 years ago',
        'type': 'number',
        'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.1,
        'help': 'Total pounds of pesticides applied per farmed acre'
    },
    'Total_Pest_Intensity_Lag_3': {
        'label': 'Total Pesticide Intensity (lbs/acre) - 3 years ago',
        'type': 'number',
        'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.1
    },
    'Total_Pest_Intensity_Lag_10': {
        'label': 'Total Pesticide Intensity (lbs/acre) - 10 years ago',
        'type': 'number',
        'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.1
    },
    
    # ===== FARM DENSITY =====
    'Farm_Density_Lag_5': {
        'label': 'Farm Density (% of county) - 5 years ago',
        'type': 'slider',
        'min': 0.0, 'max': 1.0, 'value': 0.3, 'step': 0.01,
        'help': 'Proportion of county area used for agriculture'
    },
    'Farm_Density_Lag_3': {
        'label': 'Farm Density (% of county) - 3 years ago',
        'type': 'slider',
        'min': 0.0, 'max': 1.0, 'value': 0.3, 'step': 0.01
    },
    
    # ===== XGBOOST FEATURES =====
    'total_population': {
        'label': 'Total County Population',
        'type': 'number',
        'min': 0, 'max': 10000000, 'value': 500000, 'step': 10000,
        'help': 'Total population of the county'
    },
    'total_pesticide_lbs_per_100k': {
        'label': 'Pesticide (lbs per 100k residents) - Current',
        'type': 'number',
        'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0,
        'help': 'Pounds of pesticides per 100k population'
    },
    'total_acres_treated_per_100k': {
        'label': 'Acres Treated (per 100k residents) - Current',
        'type': 'number',
        'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0,
        'help': 'Acres treated with pesticides per 100k population'
    },
    'total_pesticide_lbs_per_100k_lag1': {
        'label': 'Pesticide (lbs per 100k) - 1 year ago',
        'type': 'number',
        'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0
    },
    'total_pesticide_lbs_per_100k_lag2': {
        'label': 'Pesticide (lbs per 100k) - 2 years ago',
        'type': 'number',
        'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0
    },
    'total_acres_treated_per_100k_lag1': {
        'label': 'Acres Treated (per 100k) - 1 year ago',
        'type': 'number',
        'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0
    },
    'total_acres_treated_per_100k_lag2': {
        'label': 'Acres Treated (per 100k) - 2 years ago',
        'type': 'number',
        'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0
    },
    'total_pesticide_lbs_per_100k_cumulative_mean5year': {
        'label': 'Pesticide 5-Year Average (lbs per 100k)',
        'type': 'number',
        'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0,
        'help': 'Rolling 5-year mean of pesticide use'
    },
    'total_pesticide_lbs_per_100k_cumulative_mean20year': {
        'label': 'Pesticide 20-Year Average (lbs per 100k)',
        'type': 'number',
        'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0,
        'help': 'Rolling 20-year mean of pesticide use'
    },
    'total_acres_treated_per_100k_cumulative_mean5year': {
        'label': 'Acres Treated 5-Year Average (per 100k)',
        'type': 'number',
        'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0
    },
    'total_acres_treated_per_100k_cumulative_mean20year': {
        'label': 'Acres Treated 20-Year Average (per 100k)',
        'type': 'number',
        'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0
    },
    
    # ===== AIR QUALITY =====
    'Median AQI': {
        'label': 'Median Air Quality Index (AQI)',
        'type': 'number',
        'min': 0.0, 'max': 200.0, 'value': 50.0, 'step': 1.0,
        'help': 'Lower is better air quality'
    },
    'median aqi': {
        'label': 'Median Air Quality Index (AQI)',
        'type': 'number',
        'min': 0.0, 'max': 200.0, 'value': 50.0, 'step': 1.0,
        'help': 'Lower is better air quality'
    },
    
    # ===== POVERTY & INCOME =====
    'Poverty_AllAges_Percent_Est': {
        'label': 'Poverty Rate (%)',
        'type': 'slider',
        'min': 0.0, 'max': 50.0, 'value': 15.0, 'step': 0.5
    },
    'poverty_allages_percent_est': {
        'label': 'Poverty Rate (%)',
        'type': 'slider',
        'min': 0.0, 'max': 50.0, 'value': 15.0, 'step': 0.5
    },
    'median_household_income_est': {
        'label': 'Median Household Income ($)',
        'type': 'number',
        'min': 20000, 'max': 150000, 'value': 65000, 'step': 1000
    },
    
    # ===== AGE DEMOGRAPHICS =====
    'pct_65_plus': {
        'label': 'Population 65+ (%)',
        'type': 'slider',
        'min': 0.0, 'max': 40.0, 'value': 15.0, 'step': 0.5
    },
    'pct_under_18': {
        'label': 'Population Under 18 (%)',
        'type': 'slider',
        'min': 0.0, 'max': 40.0, 'value': 23.0, 'step': 0.5
    },
    'pct_18_64': {
        'label': 'Population 18-64 (%)',
        'type': 'slider',
        'min': 0.0, 'max': 100.0, 'value': 62.0, 'step': 0.5
    },
    'median_age': {
        'label': 'Median Age (years)',
        'type': 'slider',
        'min': 20.0, 'max': 70.0, 'value': 38.0, 'step': 0.5
    },
    
    # ===== RACE/ETHNICITY DEMOGRAPHICS =====
    'pct_Latino': {
        'label': 'Latino Population (%)',
        'type': 'slider',
        'min': 0.0, 'max': 100.0, 'value': 40.0, 'step': 1.0
    },
    'pct_latino': {
        'label': 'Latino Population (%)',
        'type': 'slider',
        'min': 0.0, 'max': 100.0, 'value': 40.0, 'step': 1.0
    },
    'pct_Black': {
        'label': 'Black Population (%)',
        'type': 'slider',
        'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.5
    },
    'pct_black': {
        'label': 'Black Population (%)',
        'type': 'slider',
        'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.5
    },
    'pct_Asian': {
        'label': 'Asian Population (%)',
        'type': 'slider',
        'min': 0.0, 'max': 50.0, 'value': 10.0, 'step': 0.5
    },
    'pct_asian': {
        'label': 'Asian Population (%)',
        'type': 'slider',
        'min': 0.0, 'max': 50.0, 'value': 10.0, 'step': 0.5
    },
    'pct_AI/AN': {
        'label': 'American Indian/Alaska Native (%)',
        'type': 'slider',
        'min': 0.0, 'max': 20.0, 'value': 1.0, 'step': 0.5
    },
    'pct_ai/an': {
        'label': 'American Indian/Alaska Native (%)',
        'type': 'slider',
        'min': 0.0, 'max': 20.0, 'value': 1.0, 'step': 0.5
    },
    'pct_White': {
        'label': 'White Population (%)',
        'type': 'slider',
        'min': 0.0, 'max': 100.0, 'value': 50.0, 'step': 1.0
    },
    'pct_white': {
        'label': 'White Population (%)',
        'type': 'slider',
        'min': 0.0, 'max': 100.0, 'value': 50.0, 'step': 1.0
    },
    'pct_multi_race': {
        'label': 'Multi-racial Population (%)',
        'type': 'slider',
        'min': 0.0, 'max': 20.0, 'value': 3.0, 'step': 0.5
    },
    'pct_nh/pi': {
        'label': 'Native Hawaiian/Pacific Islander (%)',
        'type': 'slider',
        'min': 0.0, 'max': 10.0, 'value': 0.5, 'step': 0.1
    },
    
    # ===== RANDOM FOREST SPECIFIC FEATURES =====
    'num_chemicals': {
        'label': 'Number of Different Chemicals Used',
        'type': 'number',
        'min': 0, 'max': 200, 'value': 50, 'step': 1,
        'help': 'Count of unique pesticide chemicals applied'
    },
    'total_pesticide_lbs': {
        'label': 'Total Pesticide (lbs)',
        'type': 'number',
        'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000,
        'help': 'Total pounds of pesticides applied'
    },
    'total_acres_treated': {
        'label': 'Total Acres Treated',
        'type': 'number',
        'min': 0, 'max': 10000000, 'value': 500000, 'step': 10000,
        'help': 'Total acres treated with pesticides'
    },
    'pesticide_intensity': {
        'label': 'Pesticide Intensity (lbs/acre)',
        'type': 'number',
        'min': 0.0, 'max': 100.0, 'value': 10.0, 'step': 0.5,
        'help': 'Pounds of pesticides per acre'
    },
    'pesticide_lbs_lag_1': {
        'label': 'Pesticide (lbs) - 1 year ago',
        'type': 'number',
        'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000
    },
    'pesticide_intensity_lag_1': {
        'label': 'Pesticide Intensity - 1 year ago',
        'type': 'number',
        'min': 0.0, 'max': 100.0, 'value': 10.0, 'step': 0.5
    },
    'pesticide_lbs_lag_3': {
        'label': 'Pesticide (lbs) - 3 years ago',
        'type': 'number',
        'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000
    },
    'pesticide_intensity_lag_3': {
        'label': 'Pesticide Intensity - 3 years ago',
        'type': 'number',
        'min': 0.0, 'max': 100.0, 'value': 10.0, 'step': 0.5
    },
    'pesticide_lbs_lag_5': {
        'label': 'Pesticide (lbs) - 5 years ago',
        'type': 'number',
        'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000
    },
    'pesticide_intensity_lag_5': {
        'label': 'Pesticide Intensity - 5 years ago',
        'type': 'number',
        'min': 0.0, 'max': 100.0, 'value': 10.0, 'step': 0.5
    },
    'pesticide_lbs_lag_10': {
        'label': 'Pesticide (lbs) - 10 years ago',
        'type': 'number',
        'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000
    },
    'pesticide_intensity_lag_10': {
        'label': 'Pesticide Intensity - 10 years ago',
        'type': 'number',
        'min': 0.0, 'max': 100.0, 'value': 10.0, 'step': 0.5
    },
    'pesticide_cumsum_5yr': {
        'label': 'Pesticide 5-Year Cumulative Sum (lbs)',
        'type': 'number',
        'min': 0, 'max': 500000000, 'value': 25000000, 'step': 1000000
    },
    'pesticide_avg_5yr': {
        'label': 'Pesticide 5-Year Average (lbs/year)',
        'type': 'number',
        'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000
    },
    'pesticide_cumsum_10yr': {
        'label': 'Pesticide 10-Year Cumulative Sum (lbs)',
        'type': 'number',
        'min': 0, 'max': 1000000000, 'value': 50000000, 'step': 1000000
    },
    'pesticide_avg_10yr': {
        'label': 'Pesticide 10-Year Average (lbs/year)',
        'type': 'number',
        'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000
    },
    'pesticide_cumsum_15yr': {
        'label': 'Pesticide 15-Year Cumulative Sum (lbs)',
        'type': 'number',
        'min': 0, 'max': 1500000000, 'value': 75000000, 'step': 1000000
    },
    'pesticide_avg_15yr': {
        'label': 'Pesticide 15-Year Average (lbs/year)',
        'type': 'number',
        'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000
    },
    'pesticide_cumsum_20yr': {
        'label': 'Pesticide 20-Year Cumulative Sum (lbs)',
        'type': 'number',
        'min': 0, 'max': 2000000000, 'value': 100000000, 'step': 1000000
    },
    'pesticide_avg_20yr': {
        'label': 'Pesticide 20-Year Average (lbs/year)',
        'type': 'number',
        'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000
    },
    
    # ===== SPECIFIC CHEMICALS =====
    'Sulfur_Intensity_Lag_5': {
        'label': 'Sulfur Intensity (lbs/acre) - 5 years ago',
        'type': 'number',
        'min': 0.0, 'max': 20.0, 'value': 2.0, 'step': 0.1
    },
    'Petroleum Oil_Intensity_Lag_5': {
        'label': 'Petroleum Oil Intensity - 5 years ago',
        'type': 'number',
        'min': 0.0, 'max': 5.0, 'value': 0.5, 'step': 0.1
    }
}

# Distribute features across columns
features_col1 = selected_features[:len(selected_features)//2 + len(selected_features)%2]
features_col2 = selected_features[len(selected_features)//2 + len(selected_features)%2:]

with col1:
    st.subheader("Input Variables (Part 1)")
    for feature in features_col1:
        if feature in feature_inputs:
            config = feature_inputs[feature]
            if config['type'] == 'number':
                feature_values[feature] = st.number_input(
                    config['label'],
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['value'],
                    step=config['step'],
                    help=config.get('help', None),
                    key=f"{model_choice}_{feature}"
                )
            elif config['type'] == 'slider':
                feature_values[feature] = st.slider(
                    config['label'],
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['value'],
                    step=config['step'],
                    help=config.get('help', None),
                    key=f"{model_choice}_{feature}"
                )
        else:
            # Generic fallback for unknown features
            feature_values[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                key=f"{model_choice}_{feature}"
            )

with col2:
    st.subheader("Input Variables (Part 2)")
    for feature in features_col2:
        if feature in feature_inputs:
            config = feature_inputs[feature]
            if config['type'] == 'number':
                feature_values[feature] = st.number_input(
                    config['label'],
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['value'],
                    step=config['step'],
                    help=config.get('help', None),
                    key=f"{model_choice}_{feature}"
                )
            elif config['type'] == 'slider':
                feature_values[feature] = st.slider(
                    config['label'],
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['value'],
                    step=config['step'],
                    help=config.get('help', None),
                    key=f"{model_choice}_{feature}"
                )
        else:
            # Generic fallback for unknown features
            feature_values[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                key=f"{model_choice}_{feature}"
            )

# Create feature array in the correct order
features = np.array([[feature_values[f] for f in selected_features]])

# Make prediction
if st.button("🔮 Make Prediction", type="primary"):
    st.header("📈 Prediction Results")
    
    # Get selected model
    selected_model = models[model_choice]
    
    # Make prediction (models are already trained)
    try:
        prediction = selected_model.predict(features)[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Make sure your model is properly trained and saved.")
        st.stop()
    
    # Display prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted COPD Hospitalization Rate",
            value=f"{prediction:.2f}",
            delta="per 10,000 residents",
            help="Number of COPD hospitalizations per 10,000 residents"
        )
    
    with col2:
        risk_level = "Low" if prediction < 30 else "Medium" if prediction < 60 else "High"
        risk_color = "🟢" if prediction < 30 else "🟡" if prediction < 60 else "🔴"
        st.metric(
            label="Risk Level",
            value=f"{risk_color} {risk_level}"
        )
    
    with col3:
        st.metric(
            label="Model Used",
            value=model_choice
        )
    
    # Show which features were used
    with st.expander("📋 Features Used in This Prediction"):
        feature_df = pd.DataFrame({
            'Feature': selected_features,
            'Value': [feature_values[f] for f in selected_features]
        })
        st.dataframe(feature_df, use_container_width=True)
    
    # Feature importance (for tree-based models)
    if model_choice in ["Random Forest", "XGBoost"]:
        st.subheader("🎯 Feature Importance")
        
        try:
            importances = selected_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance_df, x='Importance', y='Feature', ax=ax)
            ax.set_title(f'Feature Importance - {model_choice}')
            st.pyplot(fig)
        except Exception as e:
            st.info(f"Feature importance not available: {str(e)}")

# Model Comparison Section
st.header("⚖️ Model Comparison")

if st.checkbox("Show Model Comparison"):
    st.subheader("Performance Metrics Across All Models")
    
    # Gather metrics from all available models
    comparison_data = {
        'Model': [],
        'R² Score': [],
        'RMSE': [],
        'MAE': [],
        'Features Used': []
    }
    
    for model_name in available_models:
        if 'metrics' in model_info[model_name]:
            metrics = model_info[model_name]['metrics']
            comparison_data['Model'].append(model_name)
            comparison_data['R² Score'].append(metrics['r2'])
            comparison_data['RMSE'].append(metrics['rmse'])
            comparison_data['MAE'].append(metrics['mae'])
            comparison_data['Features Used'].append(len(model_info[model_name]['feature_names']))
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(comparison_df, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(comparison_df))
            width = 0.25
            
            ax.bar(x - width, comparison_df['R² Score'], width, label='R² Score', alpha=0.8)
            # Scale RMSE and MAE to fit on same plot
            max_rmse = comparison_df['RMSE'].max() if comparison_df['RMSE'].max() > 0 else 1
            max_mae = comparison_df['MAE'].max() if comparison_df['MAE'].max() > 0 else 1
            ax.bar(x, comparison_df['RMSE']/max_rmse, width, label='RMSE (normalized)', alpha=0.8)
            ax.bar(x + width, comparison_df['MAE']/max_mae, width, label='MAE (normalized)', alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
            ax.legend()
            
            st.pyplot(fig)
        
        # Show which features each model uses
        st.subheader("Features by Model")
        for model_name in available_models:
            with st.expander(f"{model_name} - {len(model_info[model_name]['feature_names'])} features"):
                st.write(f"**Description:** {model_info[model_name].get('description', 'N/A')}")
                st.write("**Features:**")
                for feat in model_info[model_name]['feature_names']:
                    st.write(f"- {feat}")
    else:
        st.warning("No model metrics available. Train your models and save metrics to see comparison.")

# Information section
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### AI4ALL Ignite Project
    
    This project analyzes the relationship between agricultural pesticide use and 
    respiratory health outcomes in California using machine learning.
    
    **Data Sources:**
    - California pesticide usage data (1974-2022)
    - Respiratory health outcome data
    - Environmental and demographic factors
    
    **Models:**
    - **Linear Regression**: Baseline model for understanding linear relationships
    - **Random Forest**: Ensemble method capturing non-linear patterns
    - **XGBoost**: Gradient boosting for optimal predictive performance
    
    **Note**: This is a demonstration. Replace dummy data with actual trained models 
    and real features from your dataset.
    """)

# Footer
st.markdown("---")
st.markdown("*Developed for AI4ALL Ignite Program | Oregon State University*")
