# ==============================================================================
# STREAMLIT: Final Interactive Analytics Dashboard (Converted from Jupyter Notebook)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.dates as mdates
import warnings

# --- Setup ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

st.set_page_config(layout="wide")
st.title("Ocean Specialist Team Forecast Dashboard")

# ==============================================================================
# LOAD ALL MODELS AND DATA
# ==============================================================================

@st.cache_resource
def load_resources():
    # --- Surface Model (GRU) ---
    surface_model = tf.keras.models.load_model('surface_model_v2.h5', compile=False)
    with open('surface_model_results_v2.pkl', 'rb') as f:
        X_test_surf, y_test_surf, scaler_X_surf, scaler_y_surf = pickle.load(f)

    # --- Transition Model (LGBM) ---
    with open('transition_model_lgbm_v2.pkl', 'rb') as f:
        transition_model = pickle.load(f)
    with open('transition_model_results_lgbm_v2.pkl', 'rb') as f:
        X_test_trans, y_test_trans = pickle.load(f)

    # --- Deep Model (LGBM) ---
    with open('deep_model_lgbm_v2.pkl', 'rb') as f:
        deep_model = pickle.load(f)
    with open('deep_model_results_lgbm_v2.pkl', 'rb') as f:
        X_test_deep, y_test_deep = pickle.load(f)
        
    # --- Full fused dataset ---
    df_fused = pd.read_pickle('fused_master_dataset.pkl')
    return (surface_model, X_test_surf, y_test_surf, scaler_X_surf, scaler_y_surf,
            transition_model, X_test_trans, y_test_trans,
            deep_model, X_test_deep, y_test_deep,
            df_fused)

(surface_model, X_test_surf, y_test_surf, scaler_X_surf, scaler_y_surf,
 transition_model, X_test_trans, y_test_trans,
 deep_model, X_test_deep, y_test_deep,
 df_fused) = load_resources()

# ==============================================================================
# GENERATE & COMBINE PREDICTIONS (RUN ONCE)
# ==============================================================================

def get_predictions_gru(model, X_test, y_test, scaler_X, scaler_y):
    X_test_scaled = scaler_X.transform(X_test.fillna(0))
    X_test_gru = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    y_pred_scaled = model.predict(X_test_gru, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

def get_predictions_lgbm(model, X_test, y_test):
    y_pred = model.predict(X_test.fillna(0))
    return pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

y_pred_df_surf = get_predictions_gru(surface_model, X_test_surf, y_test_surf, scaler_X_surf, scaler_y_surf)
y_pred_df_trans = get_predictions_lgbm(transition_model, X_test_trans, y_test_trans)
y_pred_df_deep = get_predictions_lgbm(deep_model, X_test_deep, y_test_deep)

# Combine
y_test_combined = y_test_surf.copy()
y_pred_df_combined = y_pred_df_surf.copy()
trans_cols_to_add = y_test_trans.columns.difference(y_test_combined.columns)
y_test_combined = y_test_combined.join(y_test_trans[trans_cols_to_add], how='outer')
y_pred_df_combined = y_pred_df_combined.join(y_pred_df_trans[trans_cols_to_add], how='outer')
deep_cols_to_add = y_test_deep.columns.difference(y_test_combined.columns)
y_test_combined = y_test_combined.join(y_test_deep[deep_cols_to_add], how='outer')
y_pred_df_combined = y_pred_df_combined.join(y_pred_df_deep[deep_cols_to_add], how='outer')

# ==============================================================================
# OCEAN FORECASTER CLASS
# ==============================================================================

class OceanForecaster:
    def __init__(self, models, scalers, full_feature_lists, y_test_lists):
        self.surface_model = models['surface']
        self.transition_model = models['transition']
        self.deep_model = models['deep']
        self.scaler_X_surf, self.scaler_y_surf = scalers['surface']
        self.X_cols_surf = full_feature_lists['surface']
        self.X_cols_trans = full_feature_lists['transition']
        self.X_cols_deep = full_feature_lists['deep']
        self.y_cols_surf = y_test_lists['surface'].columns
        self.y_cols_trans = y_test_lists['transition'].columns
        self.y_cols_deep = y_test_lists['deep'].columns

    def _engineer_features(self, recent_data):
        df = recent_data.copy()
        new_features = {}
        new_features['day_of_year'] = df.index.dayofyear
        new_features['month'] = df.index.month
        new_features['is_winter_season'] = new_features['month'].isin([12, 1, 2]).astype(int)
        new_features['is_summer_season'] = new_features['month'].isin([3, 4, 5]).astype(int)
        new_features['is_monsoon_season'] = new_features['month'].isin([6, 7, 8, 9]).astype(int)
        new_features['is_post_monsoon_season'] = new_features['month'].isin([10, 11]).astype(int)
        
        depth_layers_trans = ['5m', '10m', '15m', '20m', '30m', '50m', '75m', '100m']
        depth_layers_deep = ['75m', '100m', '200m', '500m']
        parameters = ['temp', 'sal', 'density']
        def find_col(df, param, depth): return next((c for c in df.columns if param in c and depth in c), None)
        for i in range(len(depth_layers_trans) - 1):
            for param in parameters:
                upper_col = find_col(df, param, depth_layers_trans[i])
                lower_col = find_col(df, param, depth_layers_trans[i+1])
                if upper_col and lower_col: new_features[f'{param}_gradient_{depth_layers_trans[i]}_to_{depth_layers_trans[i+1]}'] = df[upper_col] - df[lower_col]
        for i in range(len(depth_layers_deep) - 1):
            for param in parameters:
                upper_col = find_col(df, param, depth_layers_deep[i])
                lower_col = find_col(df, param, depth_layers_deep[i+1])
                if upper_col and lower_col: new_features[f'{param}_gradient_{depth_layers_deep[i]}_to_{depth_layers_deep[i+1]}'] = df[upper_col] - df[lower_col]
        
        df = df.assign(**new_features)
        temporal_features = {}
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
        lags = [1, 24, 168]; window = 24
        for col in feature_cols:
            for lag in lags: temporal_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
            temporal_features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        df = df.assign(**temporal_features)
        latest_row = df.tail(1).fillna(0)
        latest_row['buoy_id_BD08'] = False
        latest_row['buoy_id_BD14'] = False
        latest_row['buoy_id_AD07'] = False
        current_buoy = recent_data['buoy_id'].iloc[0]
        buoy_col_name = f'buoy_id_{current_buoy}'
        if buoy_col_name in latest_row.columns: latest_row[buoy_col_name] = True
        if 'buoy_id' in latest_row.columns: latest_row = latest_row.drop(columns=['buoy_id'])
        return latest_row

    def _post_process(self, forecast_df):
        sal_cols = [col for col in forecast_df.columns if 'sal_' in col]
        forecast_df[sal_cols] = forecast_df[sal_cols].clip(lower=0)
        return forecast_df

    def predict(self, recent_data_history):
        latest_features = self._engineer_features(recent_data_history)
        all_expected_cols = set(self.X_cols_surf) | set(self.X_cols_trans) | set(self.X_cols_deep)
        for col in all_expected_cols:
            if col not in latest_features.columns: latest_features[col] = 0
        X_surf = latest_features[self.X_cols_surf]
        X_trans = latest_features[self.X_cols_trans]
        X_deep = latest_features[self.X_cols_deep]
        X_surf_scaled = self.scaler_X_surf.transform(X_surf.fillna(0)).reshape((1, 1, -1))
        X_trans_scaled = X_trans.fillna(0)
        X_deep_scaled = X_deep.fillna(0)
        pred_surf_scaled = self.surface_model.predict(X_surf_scaled, verbose=0)
        pred_trans = self.transition_model.predict(X_trans_scaled)
        pred_deep = self.deep_model.predict(X_deep_scaled)
        pred_surf = self.scaler_y_surf.inverse_transform(pred_surf_scaled)
        df_pred_surf = pd.DataFrame(pred_surf, columns=self.y_cols_surf)
        df_pred_trans = pd.DataFrame(pred_trans, columns=self.y_cols_trans)
        df_pred_deep = pd.DataFrame(pred_deep, columns=self.y_cols_deep)
        final_forecast_raw = df_pred_surf
        final_forecast_raw = final_forecast_raw.join(df_pred_trans.drop(columns=final_forecast_raw.columns, errors='ignore'))
        final_forecast_raw = final_forecast_raw.join(df_pred_deep.drop(columns=final_forecast_raw.columns, errors='ignore'))
        final_forecast_processed = self._post_process(final_forecast_raw)
        return final_forecast_processed

models = {'surface': surface_model, 'transition': transition_model, 'deep': deep_model}
scalers = {'surface': (scaler_X_surf, scaler_y_surf)}
full_feature_lists = {
    'surface': scaler_X_surf.feature_names_in_,
    'transition': X_test_trans.columns,
    'deep': X_test_deep.columns
}
y_test_lists = {'surface': y_test_surf, 'transition': y_test_trans, 'deep': y_test_deep}
forecaster = OceanForecaster(models, scalers, full_feature_lists, y_test_lists)

# ==============================================================================
# HELPERS, COMMON VALUES
# ==============================================================================

def get_depth_from_col(col_name):
    parts = col_name.split('_')
    for part in parts:
        if part.endswith('m') or part.endswith('mself') or part.endswith('msmp'):
            numeric_part = part.replace('mself', '').replace('msmp', '').replace('m', '')
            if numeric_part.isdigit():
                return part
    return None

all_target_cols = [col for col in y_test_combined.columns if '_target_' in col]
unique_depths = sorted(list(set(get_depth_from_col(c) for c in all_target_cols if get_depth_from_col(c))),
                   key=lambda d: int(re.search(r'\d+', d).group()))
horizons_map = {'1hr': 1, '1day': 24, '3day': 72, '5day': 120, '7day': 168, '10day': 240, '15day': 360}

param_options = ['temp', 'sal']
depth_options = unique_depths
horizon_options = ['1hr', '1day', '3day', '5day', '7day', '10day', '15day']
duration_options = ['24 Hours', '7 Days', '30 Days']

# ==============================================================================
# --- STREAMLIT SIDEBAR SECTION SELECTOR ---
# ==============================================================================

section = st.sidebar.selectbox(
    "Dashboard Section", 
    ("System Performance by Depth", "Interactive Forecast Explorer", "Live Forecast Simulator")
)

# ==============================================================================
# --- PART 1: SYSTEM PERFORMANCE BY DEPTH ---
# ==============================================================================
if section == "System Performance by Depth":
    st.header("PART 1: System Performance by Depth")
    st.info(f"Analyzing system forecast accuracy for each depth profile over 1-day horizons.")

    depth_results = []
    horizon_to_check = '1day'
    for depth in unique_depths:
        for param in ['sal', 'temp']:
            target = f'{param}_{depth}_target_{horizon_to_check}'
            if target in y_test_combined.columns:
                temp_df = pd.DataFrame({
                    'actual': y_test_combined[target],
                    'predicted': y_pred_df_combined[target]
                }).dropna()
                if not temp_df.empty:
                    r2 = r2_score(temp_df['actual'], temp_df['predicted'])
                    label = 'Salinity' if param == 'sal' else 'Temperature'
                    depth_results.append({'Parameter': label, 'Depth (m)': int(re.search(r'\d+', depth).group()), 'R-squared': r2})

    results_df = pd.DataFrame(depth_results)
    aggregated_results_df = results_df.groupby(['Depth (m)', 'Parameter']).mean().reset_index()

    st.subheader(f"Accuracy Report (1-day horizon):")
    pivot = aggregated_results_df.pivot(index='Depth (m)', columns='Parameter', values='R-squared')
    st.dataframe(pivot, use_container_width=True)
    avg_r2 = aggregated_results_df['R-squared'].mean()
    st.success(f"Overall Average R²: {avg_r2:.2%}")

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    sns.barplot(data=aggregated_results_df[aggregated_results_df['Parameter'] == 'Salinity'], 
                x='Depth (m)', y='R-squared', ax=ax1, palette='coolwarm')
    ax1.set_title('Salinity Forecast Accuracy (R²)'); ax1.axhline(0.85, color='green', linestyle='--'); ax1.legend(['85% Target'])
    sns.barplot(data=aggregated_results_df[aggregated_results_df['Parameter'] == 'Temperature'], 
                x='Depth (m)', y='R-squared', ax=ax2, palette='coolwarm')
    ax2.set_title('Temperature Forecast Accuracy (R²)'); ax2.set_xlabel('Depth (m)'); ax2.legend(['85% Target'])
    st.pyplot(fig)

# ==============================================================================
# --- PART 2: INTERACTIVE FORECAST EXPLORER ---
# ==============================================================================

elif section == "Interactive Forecast Explorer":
    st.header("PART 2: Interactive Forecast Explorer")
    st.info("Select options below to visualize how well the model performed for any given depth/parameter/horizon.")
    param = st.selectbox("Parameter", param_options, index=0)
    depth = st.selectbox("Depth", depth_options, index=depth_options.index('50m') if '50m' in depth_options else 0)
    horizon = st.selectbox("Forecast Horizon", horizon_options, index=1)
    duration = st.selectbox("Plot Duration", duration_options, index=1)

    # Interactive plot code
    target_col = f"{param}_{depth}_target_{horizon}"
    if target_col not in y_test_combined.columns:
        st.error(f"ERROR: The target '{target_col}' does not exist. Please check your inputs.")
    else:
        start_date = y_test_combined.index.min()
        # Duration handling
        if duration == '24 Hours':
            end_date = start_date + pd.Timedelta(hours=24)
            datelabel_fmt = '%Y-%m-%d %H:%M'
        elif duration == '7 Days':
            end_date = start_date + pd.Timedelta(days=7)
            datelabel_fmt = '%Y-%m-%d'
        else:
            end_date = start_date + pd.Timedelta(days=30)
            datelabel_fmt = '%Y-%m-%d'

        actual_series = y_test_combined.loc[start_date:end_date, target_col]
        predicted_series = y_pred_df_combined.loc[start_date:end_date, target_col]
        temp_df = pd.DataFrame({'actual': actual_series, 'predicted': predicted_series}).dropna()
        unit = "PSU" if param == 'sal' else "°C"
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(actual_series.index, actual_series, 'o-', ms=4, color='darkblue', label='Actual')
        ax.plot(predicted_series.index, predicted_series, 'x--', ms=4, color='red', label=f'Predicted ({horizon})')
        ax.set_title(f"{duration} Forecast Validation for {target_col}")
        ax.set_ylabel(f"{param.capitalize()} ({unit})")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        if not temp_df.empty:
            r2 = r2_score(temp_df['actual'], temp_df['predicted'])
            st.success(f"Performance for this plot: R² = {r2:.2f}")

# ==============================================================================
# --- PART 3: LIVE FORECAST SIMULATOR ---
# ==============================================================================

else:
    st.header("PART 3: 'Live' Forecast Simulator")
    st.info("Simulate a fresh forecast using the latest week of real data for any buoy.")
    buoy = st.selectbox("Buoy", ['AD07', 'BD08', 'BD14'])
    parameter = st.selectbox("Parameter", param_options, index=1)
    depth = st.selectbox("Depth", depth_options, index=0)
    horizon = st.selectbox("Horizon", horizon_options, index=1)
    
    if st.button("Generate Live Forecast"):
        try:
            # Simulate live data
            end_time = df_fused[df_fused['buoy_id'] == buoy].index.max()
            start_time = end_time - pd.Timedelta(days=7)
            live_data_simulation = df_fused[
                (df_fused['buoy_id'] == buoy) &
                (df_fused.index >= start_time) & (df_fused.index <= end_time)
            ]
            complete_forecast = forecaster.predict(live_data_simulation)
            target_col = f"{parameter}_{depth}_target_{horizon}"
            predicted_value = complete_forecast[target_col].iloc[0]
            unit = "PSU" if parameter == 'sal' else "°C"
            forecast_date = end_time + pd.Timedelta(hours=horizons_map[horizon])
            st.success(f"Prediction for {horizon} ({forecast_date.date()}):\n"
                       f"- Parameter: {parameter.capitalize()}\n"
                       f"- Depth: {depth}\n"
                       f"- Value: {predicted_value:.4f} {unit}")
        except KeyError:
            st.error(f"Combination of parameter '{parameter}', depth '{depth}', and horizon '{horizon}' not found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
