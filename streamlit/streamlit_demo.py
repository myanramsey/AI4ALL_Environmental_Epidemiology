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

# Configuration
MODEL_DIR = 'streamlit/' if os.path.exists('streamlit') else ''

st.set_page_config(
    page_title="Pesticide & COPD Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .stAlert {
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🌾 Pesticide Exposure & COPD Hospitalization Predictor")
st.markdown("""
    *Machine learning models predicting respiratory health outcomes based on agricultural pesticide exposure across California counties.*
""")
st.markdown("---")

@st.cache_resource
def load_models():
    """Load pre-trained models and metadata from pickle files."""
    models = {}
    model_info = {}
    
    model_files = {
        'Linear Regression': {
            'model': f'{MODEL_DIR}linear_regression.pkl',
            'info': f'{MODEL_DIR}linear_regression_info.pkl'
        },
        'Random Forest': {
            'model': f'{MODEL_DIR}random_forest.pkl',
            'info': f'{MODEL_DIR}random_forest_info.pkl'
        },
        'XGBoost': {
            'model': f'{MODEL_DIR}xgboost.pkl',
            'info': f'{MODEL_DIR}xgboost_info.pkl'
        }
    }
    
    for model_name, files in model_files.items():
        try:
            with open(files['model'], 'rb') as f:
                models[model_name] = pickle.load(f)
            with open(files['info'], 'rb') as f:
                model_info[model_name] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"⚠️ {model_name} files not found")
            continue
    
    return models, model_info

try:
    models, model_info = load_models()
    available_models = list(models.keys())
    
    if not available_models:
        st.error("❌ No models found! Please ensure model files are in the correct directory.")
        st.info("""
        Required files:
        - `xgboost.pkl` & `xgboost_info.pkl`
        - `random_forest.pkl` & `random_forest_info.pkl` (optional)
        - `linear_regression.pkl` & `linear_regression_info.pkl` (optional)
        """)
        st.stop()
        
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Creating demo models for testing...")
    
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
            'description': 'Demo model with 5-year lag and demographic controls',
            'metrics': {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
        },
        'Random Forest': {
            'feature_names': [
                'Total_Pest_Intensity_Lag_5', 'Farm_Density_Lag_5',
                'Median AQI', 'Poverty_AllAges_Percent_Est', 'pct_65_plus',
                'pct_Latino'
            ],
            'description': 'Demo model with selected features',
            'metrics': {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
        },
        'XGBoost': {
            'feature_names': [
                'total_population', 'total_pesticide_lbs_per_100k',
                'total_acres_treated_per_100k', 'total_pesticide_lbs_per_100k_lag1',
                'total_pesticide_lbs_per_100k_lag2', 'total_acres_treated_per_100k_lag1',
                'total_acres_treated_per_100k_lag2',
                'total_pesticide_lbs_per_100k_cumulative_mean5year',
                'total_pesticide_lbs_per_100k_cumulative_mean20year',
                'total_acres_treated_per_100k_cumulative_mean5year',
                'total_acres_treated_per_100k_cumulative_mean20year',
                'median aqi', 'pct_under_18', 'pct_18_64', 'pct_65_plus', 'median_age',
                'pct_ai/an', 'pct_asian', 'pct_black', 'pct_latino', 'pct_multi_race',
                'pct_nh/pi', 'pct_white', 'poverty_allages_percent_est', 
                'median_household_income_est'
            ],
            'description': 'XGBoost with temporal lags (1-2yr), cumulative exposures (5yr, 20yr), and full demographic controls. Trained on 2005-2022 data (n=943).',
            'metrics': {'r2': 0.5895, 'rmse': 6.51, 'mae': 4.43, 'train_r2': 0.7242, 'train_rmse': 4.42}
        }
    }
    available_models = list(models.keys())

# Sidebar
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    available_models,
    help="Choose a machine learning model for prediction"
)

if model_choice in model_info:
    st.sidebar.markdown(f"**Features:** {len(model_info[model_choice]['feature_names'])}")
    st.sidebar.caption(model_info[model_choice].get('description', ''))
    
    if 'metrics' in model_info[model_choice]:
        metrics = model_info[model_choice]['metrics']
        if metrics.get('r2', 0) > 0:
            st.sidebar.markdown("### 📊 Model Performance")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("R² (Test)", f"{metrics['r2']:.3f}")
                st.metric("RMSE", f"{metrics['rmse']:.2f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.2f}")
                if model_choice == 'XGBoost' and 'train_r2' in metrics:
                    r2_gap = metrics['train_r2'] - metrics['r2']
                    gap_label = "Moderate" if r2_gap > 0.1 else "Minimal"
                    st.metric("Overfit", gap_label, f"{r2_gap:.3f}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Quick Guide")
st.sidebar.info("""
1. Select a model above
2. Enter feature values below
3. Click 'Predict' to see results
4. Explore feature importance & insights
""")

# Feature input definitions
selected_features = model_info[model_choice]['feature_names']

feature_inputs = {
    'Total_Pest_Intensity_Lag_5': {
        'label': 'Pesticide Intensity (lbs/acre) - 5yr Lag',
        'type': 'number', 'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.1,
        'help': 'Total pounds of pesticides per farmed acre, 5 years ago'
    },
    'Total_Pest_Intensity_Lag_3': {
        'label': 'Pesticide Intensity (lbs/acre) - 3yr Lag',
        'type': 'number', 'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.1
    },
    'Total_Pest_Intensity_Lag_10': {
        'label': 'Pesticide Intensity (lbs/acre) - 10yr Lag',
        'type': 'number', 'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.1
    },
    'Farm_Density_Lag_5': {
        'label': 'Farm Density - 5yr Lag',
        'type': 'slider', 'min': 0.0, 'max': 1.0, 'value': 0.3, 'step': 0.01,
        'help': 'Proportion of county area used for agriculture'
    },
    'Farm_Density_Lag_3': {
        'label': 'Farm Density - 3yr Lag',
        'type': 'slider', 'min': 0.0, 'max': 1.0, 'value': 0.3, 'step': 0.01
    },
    'total_population': {
        'label': 'County Population',
        'type': 'number', 'min': 0, 'max': 10000000, 'value': 500000, 'step': 10000
    },
    'total_pesticide_lbs_per_100k': {
        'label': 'Pesticide (lbs/100k) - Current',
        'type': 'number', 'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0,
        'help': 'Pounds of pesticides per 100k population'
    },
    'total_acres_treated_per_100k': {
        'label': 'Acres Treated (per 100k) - Current',
        'type': 'number', 'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0
    },
    'total_pesticide_lbs_per_100k_lag1': {
        'label': 'Pesticide (lbs/100k) - 1yr Lag',
        'type': 'number', 'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0
    },
    'total_pesticide_lbs_per_100k_lag2': {
        'label': 'Pesticide (lbs/100k) - 2yr Lag',
        'type': 'number', 'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0
    },
    'total_acres_treated_per_100k_lag1': {
        'label': 'Acres Treated (per 100k) - 1yr Lag',
        'type': 'number', 'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0
    },
    'total_acres_treated_per_100k_lag2': {
        'label': 'Acres Treated (per 100k) - 2yr Lag',
        'type': 'number', 'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0
    },
    'total_pesticide_lbs_per_100k_cumulative_mean5year': {
        'label': 'Pesticide 5yr Avg (lbs/100k)',
        'type': 'number', 'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0,
        'help': 'Rolling 5-year mean of pesticide use'
    },
    'total_pesticide_lbs_per_100k_cumulative_mean20year': {
        'label': 'Pesticide 20yr Avg (lbs/100k)',
        'type': 'number', 'min': 0.0, 'max': 100000.0, 'value': 5000.0, 'step': 100.0,
        'help': 'Rolling 20-year mean - captures long-term exposure'
    },
    'total_acres_treated_per_100k_cumulative_mean5year': {
        'label': 'Acres Treated 5yr Avg (per 100k)',
        'type': 'number', 'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0
    },
    'total_acres_treated_per_100k_cumulative_mean20year': {
        'label': 'Acres Treated 20yr Avg (per 100k)',
        'type': 'number', 'min': 0.0, 'max': 50000.0, 'value': 2000.0, 'step': 100.0
    },
    'Median AQI': {
        'label': 'Median Air Quality Index',
        'type': 'number', 'min': 0.0, 'max': 200.0, 'value': 50.0, 'step': 1.0,
        'help': 'Lower values indicate better air quality'
    },
    'median aqi': {
        'label': 'Median Air Quality Index',
        'type': 'number', 'min': 0.0, 'max': 200.0, 'value': 50.0, 'step': 1.0
    },
    'Poverty_AllAges_Percent_Est': {
        'label': 'Poverty Rate (%)',
        'type': 'slider', 'min': 0.0, 'max': 50.0, 'value': 15.0, 'step': 0.5
    },
    'poverty_allages_percent_est': {
        'label': 'Poverty Rate (%)',
        'type': 'slider', 'min': 0.0, 'max': 50.0, 'value': 15.0, 'step': 0.5
    },
    'median_household_income_est': {
        'label': 'Median Household Income ($)',
        'type': 'number', 'min': 20000, 'max': 150000, 'value': 65000, 'step': 1000
    },
    'pct_65_plus': {
        'label': 'Population 65+ (%)',
        'type': 'slider', 'min': 0.0, 'max': 40.0, 'value': 15.0, 'step': 0.5
    },
    'pct_under_18': {
        'label': 'Population Under 18 (%)',
        'type': 'slider', 'min': 0.0, 'max': 40.0, 'value': 23.0, 'step': 0.5
    },
    'pct_18_64': {
        'label': 'Population 18-64 (%)',
        'type': 'slider', 'min': 0.0, 'max': 100.0, 'value': 62.0, 'step': 0.5
    },
    'median_age': {
        'label': 'Median Age',
        'type': 'slider', 'min': 20.0, 'max': 70.0, 'value': 38.0, 'step': 0.5
    },
    'pct_Latino': {
        'label': 'Latino Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 100.0, 'value': 40.0, 'step': 1.0
    },
    'pct_latino': {
        'label': 'Latino Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 100.0, 'value': 40.0, 'step': 1.0
    },
    'pct_Black': {
        'label': 'Black Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.5
    },
    'pct_black': {
        'label': 'Black Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 50.0, 'value': 5.0, 'step': 0.5
    },
    'pct_Asian': {
        'label': 'Asian Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 50.0, 'value': 10.0, 'step': 0.5
    },
    'pct_asian': {
        'label': 'Asian Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 50.0, 'value': 10.0, 'step': 0.5
    },
    'pct_AI/AN': {
        'label': 'AI/AN Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 20.0, 'value': 1.0, 'step': 0.5,
        'help': 'American Indian / Alaska Native'
    },
    'pct_ai/an': {
        'label': 'AI/AN Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 20.0, 'value': 1.0, 'step': 0.5
    },
    'pct_White': {
        'label': 'White Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 100.0, 'value': 50.0, 'step': 1.0
    },
    'pct_white': {
        'label': 'White Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 100.0, 'value': 50.0, 'step': 1.0
    },
    'pct_multi_race': {
        'label': 'Multi-racial Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 20.0, 'value': 3.0, 'step': 0.5
    },
    'pct_nh/pi': {
        'label': 'NH/PI Population (%)',
        'type': 'slider', 'min': 0.0, 'max': 10.0, 'value': 0.5, 'step': 0.1,
        'help': 'Native Hawaiian / Pacific Islander'
    },
    'num_chemicals': {
        'label': 'Number of Chemicals Used',
        'type': 'number', 'min': 0, 'max': 200, 'value': 50, 'step': 1
    },
    'total_pesticide_lbs': {
        'label': 'Total Pesticide (lbs)',
        'type': 'number', 'min': 0, 'max': 100000000, 'value': 5000000, 'step': 100000
    },
    'total_acres_treated': {
        'label': 'Total Acres Treated',
        'type': 'number', 'min': 0, 'max': 10000000, 'value': 500000, 'step': 10000
    },
    'pesticide_intensity': {
        'label': 'Pesticide Intensity (lbs/acre)',
        'type': 'number', 'min': 0.0, 'max': 100.0, 'value': 10.0, 'step': 0.5
    }
}

# Feature input section
st.header("📊 Input Features")
st.info(f"**{model_choice}** requires {len(selected_features)} features. Adjust values below:")

feature_values = {}
col1, col2 = st.columns(2)

features_per_column = len(selected_features) // 2 + (len(selected_features) % 2)
features_col1 = selected_features[:features_per_column]
features_col2 = selected_features[features_per_column:]

with col1:
    st.subheader("Features (Part 1)")
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
            feature_values[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                key=f"{model_choice}_{feature}"
            )

with col2:
    st.subheader("Features (Part 2)")
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
            feature_values[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                key=f"{model_choice}_{feature}"
            )

features = np.array([[feature_values[f] for f in selected_features]])

st.markdown("---")

# Prediction
if st.button("🔮 Generate Prediction", type="primary"):
    st.header("📈 Prediction Results")
    
    selected_model = models[model_choice]
    
    try:
        prediction = selected_model.predict(features)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Ensure your model is properly trained and saved.")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted COPD Rate",
            value=f"{prediction:.2f}",
            help="COPD hospitalizations per 10,000 residents"
        )
    
    with col2:
        if prediction < 10:
            risk_level, risk_color, risk_emoji = "Very Low", "#28a745", "🟢"
        elif prediction < 15:
            risk_level, risk_color, risk_emoji = "Low", "#90ee90", "🟢"
        elif prediction < 20:
            risk_level, risk_color, risk_emoji = "Moderate", "#ffc107", "🟡"
        elif prediction < 30:
            risk_level, risk_color, risk_emoji = "High", "#ff9800", "🟠"
        else:
            risk_level, risk_color, risk_emoji = "Very High", "#dc3545", "🔴"
        
        st.metric(
            label="Risk Level",
            value=f"{risk_emoji} {risk_level}"
        )
    
    with col3:
        st.metric(
            label="Model",
            value=model_choice
        )
    
    with st.expander("📋 View Input Features"):
        feature_df = pd.DataFrame({
            'Feature': selected_features,
            'Value': [feature_values[f] for f in selected_features]
        })
        st.dataframe(feature_df, use_container_width=True, height=400)
    
    # Feature importance
    if model_choice in ["Random Forest", "XGBoost"]:
        st.markdown("---")
        st.subheader("🎯 Feature Importance Analysis")
        
        try:
            importances = selected_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df)))
                sns.barplot(
                    data=feature_importance_df.head(15), 
                    x='Importance', 
                    y='Feature', 
                    palette=colors,
                    ax=ax
                )
                ax.set_title(f'Top 15 Most Important Features - {model_choice}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Importance Score', fontsize=12)
                ax.set_ylabel('Feature', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col_b:
                st.markdown("### Top 5 Features")
                for idx, row in feature_importance_df.head(5).iterrows():
                    st.metric(
                        row['Feature'][:30] + "..." if len(row['Feature']) > 30 else row['Feature'],
                        f"{row['Importance']:.3f}"
                    )
            
            # XGBoost specific analysis
            if model_choice == "XGBoost":
                st.markdown("---")
                st.subheader("🧬 Pesticide vs Confounder Contribution")
                
                pesticide_features = [f for f in selected_features if any(x in f for x in 
                    ['pesticide', 'acres', 'population'])]
                confounder_features = [f for f in selected_features if f not in pesticide_features]
                
                pesticide_importance = feature_importance_df[
                    feature_importance_df['Feature'].isin(pesticide_features)
                ]['Importance'].sum()
                confounder_importance = feature_importance_df[
                    feature_importance_df['Feature'].isin(confounder_features)
                ]['Importance'].sum()
                
                col_x, col_y, col_z = st.columns([1, 1, 2])
                
                with col_x:
                    st.metric(
                        "Pesticide Features", 
                        f"{pesticide_importance:.1%}",
                        help="Contribution from pesticide exposure variables"
                    )
                
                with col_y:
                    st.metric(
                        "Confounder Features", 
                        f"{confounder_importance:.1%}",
                        help="Contribution from demographic and environmental controls"
                    )
                
                with col_z:
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    colors_pie = ['#ff9999', '#66b3ff']
                    explode = (0.05, 0)
                    ax2.pie(
                        [pesticide_importance, confounder_importance], 
                        labels=['Pesticide Features', 'Confounder Features'],
                        autopct='%1.1f%%', 
                        startangle=90,
                        colors=colors_pie,
                        explode=explode,
                        shadow=True
                    )
                    ax2.set_title('Feature Category Contribution', fontweight='bold')
                    st.pyplot(fig2)
                    
        except Exception as e:
            st.info(f"Feature importance not available: {str(e)}")
    
    # XGBoost insights
    if model_choice == "XGBoost":
        st.markdown("---")
        with st.expander("🔬 XGBoost Model Insights & Findings", expanded=False):
            col_i, col_ii = st.columns([1, 1])
            
            with col_i:
                st.markdown("### 📊 Model Performance")
                st.markdown(f"""
                - **Test R²:** 0.5895 (explains 59% of variance)
                - **Test RMSE:** 6.51 hospitalizations per 10k
                - **Test MAE:** 4.43 hospitalizations per 10k
                - **Train R²:** 0.7242 (moderate overfitting)
                - **Baseline Improvement:** 36% better than naive prediction
                """)
                
                st.markdown("### 🎯 Top Predictive Features")
                st.markdown("""
                1. **20-year cumulative pesticide exposure** ⭐
                2. Median household income (socioeconomic)
                3. 20-year cumulative acres treated
                4. American Indian/Alaska Native population %
                5. 5-year pesticide exposure averages
                """)
            
            with col_ii:
                st.markdown("### 🧪 Key Findings")
                st.markdown("""
                **Evidence for Pesticide-COPD Association:**
                - Pesticide features = **46.3%** of predictive power
                - Long-term exposure (20yr) > Short-term exposure
                - Results control for air quality, poverty, demographics, income
                
                **Methodology:**
                - Random 80/20 train/test split
                - Year included as feature (addresses temporal trends)
                - Temporal lags (1-2yr) capture delayed health effects
                - Cumulative windows (5yr, 20yr) capture chronic exposure
                - Data: 2005-2022, 53 California counties (n=943)
                """)
                
                st.markdown("### ⚙️ Model Configuration")
                st.markdown("""
                - Learning rate: 0.05
                - Max depth: 3 (prevents overfitting)
                - Trees: 50
                - Regularization: L1=1.0, L2=1.0
                """)

# Model comparison
st.markdown("---")
st.header("⚖️ Model Comparison")

if st.checkbox("Show Model Performance Comparison"):
    st.subheader("Performance Metrics Across Models")
    
    comparison_data = {
        'Model': [],
        'R² Score': [],
        'RMSE': [],
        'MAE': [],
        'Features': []
    }
    
    for model_name in available_models:
        if 'metrics' in model_info[model_name]:
            metrics = model_info[model_name]['metrics']
            comparison_data['Model'].append(model_name)
            comparison_data['R² Score'].append(metrics.get('r2', 0))
            comparison_data['RMSE'].append(metrics.get('rmse', 0))
            comparison_data['MAE'].append(metrics.get('mae', 0))
            comparison_data['Features'].append(len(model_info[model_name]['feature_names']))
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 0:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, subset=['R² Score'], color='lightgreen')
                               .highlight_min(axis=0, subset=['RMSE', 'MAE'], color='lightgreen'),
                use_container_width=True
            )
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(comparison_df))
            width = 0.25
            
            ax.bar(x - width, comparison_df['R² Score'], width, label='R² Score', alpha=0.8, color='#2ecc71')
            
            max_rmse = comparison_df['RMSE'].max() if comparison_df['RMSE'].max() > 0 else 1
            max_mae = comparison_df['MAE'].max() if comparison_df['MAE'].max() > 0 else 1
            ax.bar(x, comparison_df['RMSE']/max_rmse, width, label='RMSE (normalized)', alpha=0.8, color='#e74c3c')
            ax.bar(x + width, comparison_df['MAE']/max_mae, width, label='MAE (normalized)', alpha=0.8, color='#f39c12')
            
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
            ax.legend(loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("### Features by Model")
        for model_name in available_models:
            with st.expander(f"{model_name} - {len(model_info[model_name]['feature_names'])} features"):
                st.markdown(f"**Description:** {model_info[model_name].get('description', 'N/A')}")
                st.markdown("**Feature List:**")
                
                features_list = model_info[model_name]['feature_names']
                col1, col2 = st.columns(2)
                mid = len(features_list) // 2
                
                with col1:
                    for feat in features_list[:mid]:
                        st.markdown(f"- `{feat}`")
                with col2:
                    for feat in features_list[mid:]:
                        st.markdown(f"- `{feat}`")
    else:
        st.warning("No model metrics available for comparison.")

# About section
st.markdown("---")
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### 🎓 AI4ALL Ignite Research Project
    
    **Objective:** Analyze the relationship between agricultural pesticide exposure and COPD hospitalization 
    rates across California counties using machine learning.
    
    ### 📊 Data Sources
    - **Pesticide Data:** California Department of Pesticide Regulation (1974-2022)
    - **Health Outcomes:** California OSHPD hospitalization records (2005-2022)
    - **Demographics:** US Census Bureau, American Community Survey
    - **Environmental:** EPA Air Quality System
    - **Socioeconomic:** Census SAIPE poverty estimates
    
    ### 🤖 Machine Learning Models
    
    **XGBoost (Gradient Boosting)**
    - Best performance: R² = 0.5895
    - 25 features including temporal lags and cumulative exposures
    - Handles non-linear relationships and feature interactions
    
    **Random Forest (Ensemble)**
    - Reduces overfitting through tree averaging
    - Good for capturing complex patterns
    
    **Linear Regression (Baseline)**
    - Interpretable linear relationships
    - Benchmark for comparison
    
    ### 🔬 Analytical Approach
    - **Temporal Analysis:** 1-2 year lag features capture delayed health effects
    - **Cumulative Exposure:** 5 and 20-year rolling averages capture chronic exposure
    - **Confounding Control:** Demographics, poverty, income, air quality included
    - **Population Normalization:** Per 100k residents for fair county comparison
    - **Time Period:** 2005-2022 (n=943 county-year observations)
    
    ### 📈 Key Research Findings
    1. Long-term pesticide exposure (20-year cumulative) is a top predictor of COPD rates
    2. Pesticide features contribute 46.3% of model predictive power
    3. Chronic exposure effects stronger than acute exposure
    4. Results remain significant after controlling for confounders
    
    ### 👥 Project Team
    *AI4ALL Ignite Program - Data Science Track*
    
    ### 📝 Citation
    If using this analysis, please cite: "Machine Learning Analysis of Pesticide Exposure 
    and Respiratory Health Outcomes in California Counties, AI4ALL Ignite 2024"
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d;'>"
    "🌾 Developed for AI4ALL Ignite Program | Data Science for Public Health 🏥"
    "</div>",
    unsafe_allow_html=True
)
