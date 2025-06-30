import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced forecasting libraries (with fallbacks)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import itertools
from scipy import stats
from scipy.optimize import minimize

def create_sales_forecast(df):
    """
    Create enhanced advanced sales forecasting analysis with multiple models
    """
    st.header("🚀 Enhanced Advanced Sales Forecasting")
    st.markdown("*All amounts in Saudi Riyal (SAR)*")
    
    if df.empty:
        st.error("No data available for forecasting")
        return
    
    # Display available models
    available_models = get_enhanced_available_models()
    st.info(f"🎯 **Available Enhanced Models**: {', '.join(available_models)}")
    
    # Enhanced forecast options
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_type = st.selectbox(
            "Select Forecast Type",
            ["Overall Sales Forecast", "Business Unit Forecast", "Customer Forecast", 
             "Product Category Forecast", "Multi-level Hierarchical Forecast"]
        )
        
        forecast_period = st.selectbox(
            "Forecast Period",
            ["Next 3 Months", "Next 6 Months", "Next 12 Months", "Next 18 Months"],
            index=1
        )
    
    with col2:
        confidence_level = st.selectbox(
            "Confidence Level",
            ["80%", "90%", "95%", "99%"],
            index=2
        )
        
        validation_method = st.selectbox(
            "Validation Method",
            ["Walk-Forward", "Time Series CV", "Expanding Window"],
            index=0
        )
    
    # Enhanced model selection with categories
    st.subheader("🎛️ Model Selection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Statistical Models**")
        statistical_models = ["Enhanced SARIMA", "Advanced ETS", "Theta Method", "Seasonal Naive"]
        selected_statistical = st.multiselect("Statistical", statistical_models, default=statistical_models[:2])
    
    with col2:
        st.write("**Machine Learning Models**")
        ml_models = ["Auto XGBoost", "LightGBM", "Neural Network", "Random Forest Pro", "Gradient Boosting"]
        selected_ml = st.multiselect("Machine Learning", ml_models, default=ml_models[:3])
    
    with col3:
        st.write("**Ensemble Methods**")
        ensemble_models = ["Dynamic Ensemble", "Stacked Ensemble", "Bayesian Model Averaging", "AutoML Ensemble"]
        selected_ensemble = st.multiselect("Ensemble", ensemble_models, default=ensemble_models[:2])
    
    model_selection = selected_statistical + selected_ml + selected_ensemble
    
    if not model_selection:
        st.warning("Please select at least one forecasting model")
        return
    
    # Advanced options
    with st.expander("🔧 Advanced Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            enable_feature_engineering = st.checkbox("Enhanced Feature Engineering", value=True)
            enable_outlier_detection = st.checkbox("Outlier Detection & Treatment", value=True)
            enable_hyperparameter_tuning = st.checkbox("Auto Hyperparameter Tuning", value=False)
        
        with col2:
            enable_uncertainty_quantification = st.checkbox("Uncertainty Quantification", value=True)
            enable_model_explanability = st.checkbox("Model Explainability", value=False)
            enable_real_time_updates = st.checkbox("Real-time Model Updates", value=False)
    
    # Execute forecasting based on type
    if forecast_type == "Overall Sales Forecast":
        create_enhanced_overall_forecast(df, forecast_period, model_selection, 
                                       confidence_level, validation_method, {
                                           'feature_engineering': enable_feature_engineering,
                                           'outlier_detection': enable_outlier_detection,
                                           'hyperparameter_tuning': enable_hyperparameter_tuning,
                                           'uncertainty_quantification': enable_uncertainty_quantification,
                                           'model_explanability': enable_model_explanability
                                       })
    elif forecast_type == "Multi-level Hierarchical Forecast":
        create_hierarchical_forecast(df, forecast_period, model_selection)
    # ... other forecast types implementation

def get_enhanced_available_models():
    """
    Get enhanced list of available forecasting models
    """
    models = ["Enhanced Linear Trend", "Advanced Moving Average", "Random Forest Pro"]
    
    if STATSMODELS_AVAILABLE:
        models.extend(["Enhanced SARIMA", "Advanced ETS", "Theta Method", "Seasonal Naive"])
    
    if PROPHET_AVAILABLE:
        models.extend(["Prophet Pro", "Prophet with Regressors"])
    
    if XGBOOST_AVAILABLE:
        models.extend(["Auto XGBoost", "XGBoost with Feature Selection"])
    
    if LIGHTGBM_AVAILABLE:
        models.append("LightGBM")
    
    models.extend([
        "Neural Network", "Gradient Boosting", "Dynamic Ensemble", 
        "Stacked Ensemble", "Bayesian Model Averaging", "AutoML Ensemble"
    ])
    
    return models

def create_enhanced_overall_forecast(df, forecast_period, model_selection, confidence_level, validation_method, advanced_options):
    """
    Create enhanced overall sales forecast with advanced techniques
    """
    st.subheader("🎯 Enhanced Overall Sales Forecast")
    
    try:
        # Enhanced data preparation
        monthly_data = prepare_enhanced_monthly_data(df, advanced_options)
        
        if monthly_data.empty or len(monthly_data) < 12:
            st.error("Insufficient data for enhanced forecasting (minimum 12 months required)")
            return
        
        periods = get_forecast_periods(forecast_period)
        confidence_interval = float(confidence_level.strip('%')) / 100
        
        # Data quality assessment
        quality_report = assess_data_quality(monthly_data)
        display_data_quality_report(quality_report)
        
        # Enhanced forecasting pipeline
        forecasts = {}
        model_performance = {}
        model_diagnostics = {}
        
        with st.spinner("🚀 Generating enhanced forecasts..."):
            progress_bar = st.progress(0)
            
            for i, model in enumerate(model_selection):
                try:
                    # Generate forecast with enhanced methods
                    forecast_result = generate_enhanced_forecast(
                        monthly_data, model, periods, advanced_options
                    )
                    
                    if forecast_result is not None:
                        forecasts[model] = forecast_result
                        
                        # Enhanced model validation
                        performance = calculate_enhanced_model_performance(
                            monthly_data, model, validation_method, advanced_options
                        )
                        model_performance[model] = performance
                        
                        # Model diagnostics
                        diagnostics = generate_model_diagnostics(monthly_data, model, forecast_result)
                        model_diagnostics[model] = diagnostics
                        
                except Exception as e:
                    st.warning(f"⚠️ Error with {model}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(model_selection))
        
        if not forecasts:
            st.error("❌ No forecasts could be generated with the selected models")
            return
        
        # Enhanced results display
        display_enhanced_forecast_results(monthly_data, forecasts, model_performance, 
                                        forecast_period, confidence_interval)
        
        # Advanced model comparison
        create_advanced_model_comparison(forecasts, model_performance, model_diagnostics)
        
        # Enhanced ensemble methods
        if len(forecasts) > 1:
            create_advanced_ensemble_forecast(monthly_data, forecasts, model_performance, 
                                            forecast_period, advanced_options)
        
        # Comprehensive forecast analysis
        create_comprehensive_forecast_analysis(monthly_data, forecasts, model_performance, 
                                             model_diagnostics, advanced_options)
        
        # Business insights and recommendations
        generate_business_insights(monthly_data, forecasts, model_performance)
        
    except Exception as e:
        st.error(f"❌ Error creating enhanced forecast: {e}")

def prepare_enhanced_monthly_data(df, advanced_options):
    """
    Enhanced data preparation with advanced feature engineering
    """
    try:
        # Basic aggregation
        monthly_data = df.groupby('YearMonth').agg({
            'Total Line Amount': ['sum', 'mean', 'std', 'count'],
            'Invoice No.': 'nunique',
            'Cust Name': 'nunique',
            'QTY': ['sum', 'mean'],
            'BU Name': 'nunique'
        }).reset_index()
        
        # Flatten column names
        monthly_data.columns = [
            'YearMonth', 'Revenue', 'Avg_Order_Value', 'Revenue_Std', 'Order_Count',
            'Invoice_Count', 'Customer_Count', 'Total_Quantity', 'Avg_Quantity', 'BU_Count'
        ]
        
        monthly_data['YearMonth'] = pd.to_datetime(monthly_data['YearMonth'])
        monthly_data = monthly_data.sort_values('YearMonth').reset_index(drop=True)
        
        if advanced_options.get('feature_engineering', True):
            monthly_data = add_enhanced_features(monthly_data)
        
        if advanced_options.get('outlier_detection', True):
            monthly_data = detect_and_treat_outliers(monthly_data)
        
        return monthly_data
        
    except Exception as e:
        st.error(f"❌ Error preparing enhanced monthly data: {e}")
        return pd.DataFrame()

def add_enhanced_features(monthly_data):
    """
    Add comprehensive feature engineering
    """
    # Time-based features
    monthly_data['Month'] = monthly_data['YearMonth'].dt.month
    monthly_data['Quarter'] = monthly_data['YearMonth'].dt.quarter
    monthly_data['Year'] = monthly_data['YearMonth'].dt.year
    monthly_data['Days_in_Month'] = monthly_data['YearMonth'].dt.days_in_month
    monthly_data['Is_Quarter_End'] = monthly_data['Month'].isin([3, 6, 9, 12]).astype(int)
    monthly_data['Is_Year_End'] = (monthly_data['Month'] == 12).astype(int)
    
    # Cyclical encoding
    monthly_data['Month_Sin'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
    monthly_data['Month_Cos'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)
    monthly_data['Quarter_Sin'] = np.sin(2 * np.pi * monthly_data['Quarter'] / 4)
    monthly_data['Quarter_Cos'] = np.cos(2 * np.pi * monthly_data['Quarter'] / 4)
    
    # Lag features (multiple lags)
    for lag in [1, 2, 3, 6, 12]:
        if lag < len(monthly_data):
            monthly_data[f'Revenue_Lag_{lag}'] = monthly_data['Revenue'].shift(lag)
            monthly_data[f'Customer_Count_Lag_{lag}'] = monthly_data['Customer_Count'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        if window < len(monthly_data):
            monthly_data[f'Revenue_MA_{window}'] = monthly_data['Revenue'].rolling(window=window).mean()
            monthly_data[f'Revenue_Std_{window}'] = monthly_data['Revenue'].rolling(window=window).std()
            monthly_data[f'Revenue_Min_{window}'] = monthly_data['Revenue'].rolling(window=window).min()
            monthly_data[f'Revenue_Max_{window}'] = monthly_data['Revenue'].rolling(window=window).max()
    
    # Growth rates
    monthly_data['Revenue_Growth'] = monthly_data['Revenue'].pct_change()
    monthly_data['Revenue_Growth_3M'] = monthly_data['Revenue'].pct_change(periods=3)
    monthly_data['Customer_Growth'] = monthly_data['Customer_Count'].pct_change()
    
    # Derived metrics
    monthly_data['Revenue_per_Customer'] = monthly_data['Revenue'] / monthly_data['Customer_Count']
    monthly_data['Revenue_per_Invoice'] = monthly_data['Revenue'] / monthly_data['Invoice_Count']
    monthly_data['Quantity_per_Customer'] = monthly_data['Total_Quantity'] / monthly_data['Customer_Count']
    
    # Trend and seasonal components
    if len(monthly_data) >= 24:  # Need at least 2 years for seasonal decomposition
        try:
            decomposition = seasonal_decompose(monthly_data['Revenue'], model='additive', period=12)
            monthly_data['Trend'] = decomposition.trend
            monthly_data['Seasonal'] = decomposition.seasonal
            monthly_data['Residual'] = decomposition.resid
        except:
            pass
    
    return monthly_data

def detect_and_treat_outliers(monthly_data):
    """
    Advanced outlier detection and treatment
    """
    # IQR method for outlier detection
    Q1 = monthly_data['Revenue'].quantile(0.25)
    Q3 = monthly_data['Revenue'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Mark outliers
    monthly_data['Is_Outlier'] = (
        (monthly_data['Revenue'] < lower_bound) | 
        (monthly_data['Revenue'] > upper_bound)
    ).astype(int)
    
    # Z-score method
    z_scores = np.abs(stats.zscore(monthly_data['Revenue']))
    monthly_data['Z_Score'] = z_scores
    monthly_data['Is_Outlier_ZScore'] = (z_scores > 3).astype(int)
    
    # Treatment: winsorization instead of removal
    monthly_data['Revenue_Treated'] = monthly_data['Revenue'].clip(
        lower=monthly_data['Revenue'].quantile(0.05),
        upper=monthly_data['Revenue'].quantile(0.95)
    )
    
    return monthly_data

def assess_data_quality(monthly_data):
    """
    Comprehensive data quality assessment
    """
    quality_report = {
        'total_periods': len(monthly_data),
        'missing_values': monthly_data.isnull().sum().sum(),
        'duplicate_periods': monthly_data['YearMonth'].duplicated().sum(),
        'outlier_count': monthly_data.get('Is_Outlier', pd.Series([])).sum(),
        'data_completeness': (1 - monthly_data.isnull().sum().sum() / (len(monthly_data) * len(monthly_data.columns))) * 100,
        'revenue_cv': monthly_data['Revenue'].std() / monthly_data['Revenue'].mean(),
        'stationarity_test': None
    }
    
    # Stationarity test
    try:
        if STATSMODELS_AVAILABLE:
            adf_result = adfuller(monthly_data['Revenue'].dropna())
            quality_report['stationarity_test'] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
    except:
        pass
    
    return quality_report

def display_data_quality_report(quality_report):
    """
    Display comprehensive data quality report
    """
    st.subheader("📊 Data Quality Assessment")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Periods", quality_report['total_periods'])
        
    with col2:
        completeness = quality_report['data_completeness']
        st.metric("Data Completeness", f"{completeness:.1f}%")
        
    with col3:
        cv = quality_report['revenue_cv']
        st.metric("Revenue Variability", f"{cv:.2f}")
        
    with col4:
        outliers = quality_report['outlier_count']
        st.metric("Outliers Detected", outliers)
    
    # Stationarity test results
    if quality_report['stationarity_test']:
        st_test = quality_report['stationarity_test']
        is_stationary = "✅ Stationary" if st_test['is_stationary'] else "⚠️ Non-Stationary"
        st.info(f"**Stationarity Test**: {is_stationary} (p-value: {st_test['p_value']:.4f})")

def generate_enhanced_forecast(monthly_data, model_name, periods, advanced_options):
    """
    Generate forecast using enhanced models with advanced techniques
    """
    try:
        if model_name == "Enhanced SARIMA":
            return enhanced_sarima_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Auto XGBoost":
            return auto_xgboost_forecast(monthly_data, periods, advanced_options)
        elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
            return lightgbm_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Neural Network":
            return neural_network_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Prophet Pro" and PROPHET_AVAILABLE:
            return prophet_pro_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Advanced ETS":
            return advanced_ets_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Theta Method":
            return theta_method_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Dynamic Ensemble":
            return dynamic_ensemble_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Stacked Ensemble":
            return stacked_ensemble_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Bayesian Model Averaging":
            return bayesian_model_averaging_forecast(monthly_data, periods, advanced_options)
        elif model_name == "AutoML Ensemble":
            return automl_ensemble_forecast(monthly_data, periods, advanced_options)
        else:
            # Fallback to basic models
            return generate_forecast(monthly_data, model_name, periods)
            
    except Exception as e:
        st.warning(f"⚠️ Error generating {model_name} forecast: {e}")
        return None

def enhanced_sarima_forecast(monthly_data, periods, advanced_options):
    """
    Enhanced SARIMA with automatic parameter selection
    """
    try:
        ts = monthly_data['Revenue'].dropna()
        
        if advanced_options.get('hyperparameter_tuning', False):
            # Grid search for optimal parameters
            best_aic = np.inf
            best_params = None
            
            # Parameter ranges
            p_range = range(0, 3)
            d_range = range(0, 2)
            q_range = range(0, 3)
            seasonal_range = [(0, 0, 0, 0), (1, 1, 1, 12), (2, 1, 2, 12)]
            
            for p, d, q in itertools.product(p_range, d_range, q_range):
                for seasonal in seasonal_range:
                    try:
                        model = SARIMAX(ts, order=(p, d, q), seasonal_order=seasonal)
                        fitted_model = model.fit(disp=False)
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = ((p, d, q), seasonal)
                    except:
                        continue
            
            if best_params:
                order, seasonal_order = best_params
                model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
                fitted_model = model.fit(disp=False)
            else:
                # Fallback to default
                model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                fitted_model = model.fit(disp=False)
        else:
            # Auto SARIMA selection (simplified)
            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fitted_model = model.fit(disp=False)
        
        # Generate forecast with confidence intervals
        forecast_result = fitted_model.get_forecast(steps=periods)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_mean.values,
            'Lower_CI': forecast_ci.iloc[:, 0].values,
            'Upper_CI': forecast_ci.iloc[:, 1].values,
            'Model': 'Enhanced SARIMA'
        })
        
    except Exception as e:
        return None

def auto_xgboost_forecast(monthly_data, periods, advanced_options):
    """
    Auto XGBoost with feature selection and hyperparameter tuning
    """
    try:
        # Feature selection
        feature_columns = [col for col in monthly_data.columns 
                          if col not in ['YearMonth', 'Revenue', 'Is_Outlier', 'Z_Score']]
        
        # Remove features with too many NaN values
        feature_columns = [col for col in feature_columns 
                          if monthly_data[col].notna().sum() / len(monthly_data) > 0.7]
        
        # Prepare training data
        train_data = monthly_data.dropna(subset=['Revenue'] + feature_columns)
        
        if len(train_data) < 12:
            return None
        
        X = train_data[feature_columns]
        y = train_data['Revenue']
        
        if advanced_options.get('hyperparameter_tuning', False):
            # Hyperparameter tuning with time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=tscv, 
                scoring='neg_mean_absolute_error', n_jobs=-1
            )
            grid_search.fit(X, y)
            model = grid_search.best_estimator_
        else:
            # Default parameters
            model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X, y)
        
        # Generate future predictions
        predictions = []
        prediction_intervals = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            # Create future features
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            future_row = create_future_features(last_row, future_date, monthly_data)
            
            # Predict
            X_future = future_row[feature_columns].values.reshape(1, -1)
            pred = model.predict(X_future)[0]
            predictions.append(pred)
            
            # Prediction intervals using quantile regression (simplified)
            # In practice, you might use more sophisticated methods
            historical_residuals = y - model.predict(X)
            residual_std = historical_residuals.std()
            lower_ci = pred - 1.96 * residual_std
            upper_ci = pred + 1.96 * residual_std
            prediction_intervals.append((lower_ci, upper_ci))
            
            # Update for next iteration
            last_row = future_row
        
        # Create forecast dataframe
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        lower_cis, upper_cis = zip(*prediction_intervals)
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Lower_CI': lower_cis,
            'Upper_CI': upper_cis,
            'Model': 'Auto XGBoost'
        })
        
    except Exception as e:
        return None

def lightgbm_forecast(monthly_data, periods, advanced_options):
    """
    LightGBM forecast with advanced configuration
    """
    try:
        # Feature preparation
        feature_columns = [col for col in monthly_data.columns 
                          if col not in ['YearMonth', 'Revenue', 'Is_Outlier', 'Z_Score']]
        
        train_data = monthly_data.dropna(subset=['Revenue'] + feature_columns)
        
        if len(train_data) < 12:
            return None
        
        X = train_data[feature_columns]
        y = train_data['Revenue']
        
        # LightGBM model
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X, y)
        
        # Generate predictions (similar to XGBoost)
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            future_row = create_future_features(last_row, future_date, monthly_data)
            
            X_future = future_row[feature_columns].values.reshape(1, -1)
            pred = model.predict(X_future)[0]
            predictions.append(pred)
            
            last_row = future_row
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'LightGBM'
        })
        
    except Exception as e:
        return None

def neural_network_forecast(monthly_data, periods, advanced_options):
    """
    Neural Network forecast using MLPRegressor
    """
    try:
        # Feature preparation
        feature_columns = [col for col in monthly_data.columns 
                          if col not in ['YearMonth', 'Revenue', 'Is_Outlier', 'Z_Score']]
        
        train_data = monthly_data.dropna(subset=['Revenue'] + feature_columns)
        
        if len(train_data) < 12:
            return None
        
        X = train_data[feature_columns]
        y = train_data['Revenue']
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Neural network model
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        
        model.fit(X_scaled, y)
        
        # Generate predictions
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            future_row = create_future_features(last_row, future_date, monthly_data)
            
            X_future = future_row[feature_columns].values.reshape(1, -1)
            X_future_scaled = scaler.transform(X_future)
            pred = model.predict(X_future_scaled)[0]
            predictions.append(pred)
            
            last_row = future_row
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Neural Network'
        })
        
    except Exception as e:
        return None

def create_future_features(last_row, future_date, monthly_data):
    """
    Create features for future periods
    """
    future_row = last_row.copy()
    
    # Update time-based features
    future_row['YearMonth'] = future_date
    future_row['Month'] = future_date.month
    future_row['Quarter'] = future_date.quarter
    future_row['Year'] = future_date.year
    future_row['Days_in_Month'] = future_date.days_in_month
    future_row['Is_Quarter_End'] = int(future_date.month in [3, 6, 9, 12])
    future_row['Is_Year_End'] = int(future_date.month == 12)
    
    # Update cyclical features
    future_row['Month_Sin'] = np.sin(2 * np.pi * future_date.month / 12)
    future_row['Month_Cos'] = np.cos(2 * np.pi * future_date.month / 12)
    future_row['Quarter_Sin'] = np.sin(2 * np.pi * future_date.quarter / 4)
    future_row['Quarter_Cos'] = np.cos(2 * np.pi * future_date.quarter / 4)
    
    # Update lag features (use last known values)
    if 'Revenue_Lag_1' in future_row:
        future_row['Revenue_Lag_1'] = last_row['Revenue']
    
    return future_row

def dynamic_ensemble_forecast(monthly_data, periods, advanced_options):
    """
    Dynamic ensemble that adapts weights based on recent performance
    """
    try:
        # Generate base forecasts
        base_forecasts = {}
        base_models = ['Enhanced Linear Trend', 'Advanced Moving Average']
        
        if STATSMODELS_AVAILABLE:
            base_models.append('Enhanced SARIMA')
        
        for model in base_models:
            forecast_result = generate_enhanced_forecast(monthly_data, model, periods, advanced_options)
            if forecast_result is not None:
                base_forecasts[model] = forecast_result
        
        if len(base_forecasts) < 2:
            return None
        
        # Calculate dynamic weights based on recent performance
        weights = calculate_dynamic_weights(monthly_data, base_forecasts.keys(), advanced_options)
        
        # Weighted ensemble
        ensemble_forecast = None
        first_model = True
        
        for model_name, forecast_df in base_forecasts.items():
            weight = weights.get(model_name, 1/len(base_forecasts))
            
            if first_model:
                ensemble_forecast = forecast_df.copy()
                ensemble_forecast['Forecast'] = forecast_df['Forecast'] * weight
                first_model = False
            else:
                ensemble_forecast['Forecast'] += forecast_df['Forecast'] * weight
        
        ensemble_forecast['Model'] = 'Dynamic Ensemble'
        
        return ensemble_forecast
        
    except Exception as e:
        return None

def calculate_dynamic_weights(monthly_data, model_names, advanced_options):
    """
    Calculate dynamic weights based on recent model performance
    """
    weights = {}
    
    # Use last 6 months for weight calculation
    if len(monthly_data) >= 12:
        train_size = len(monthly_data) - 6
        train_data = monthly_data[:train_size]
        test_data = monthly_data[train_size:]
        
        performance_scores = {}
        
        for model_name in model_names:
            try:
                forecast_result = generate_enhanced_forecast(train_data, model_name, len(test_data), advanced_options)
                if forecast_result is not None:
                    actual = test_data['Revenue'].values
                    predicted = forecast_result['Forecast'].values
                    mae = mean_absolute_error(actual, predicted)
                    performance_scores[model_name] = 1 / (1 + mae)  # Higher score for lower error
            except:
                performance_scores[model_name] = 0
        
        # Normalize weights
        total_score = sum(performance_scores.values())
        if total_score > 0:
            weights = {model: score/total_score for model, score in performance_scores.items()}
        else:
            weights = {model: 1/len(model_names) for model in model_names}
    else:
        # Equal weights if insufficient data
        weights = {model: 1/len(model_names) for model in model_names}
    
    return weights

def stacked_ensemble_forecast(monthly_data, periods, advanced_options):
    """
    Stacked ensemble using meta-learning
    """
    try:
        # Generate base forecasts
        base_models = ['Enhanced Linear Trend', 'Advanced Moving Average']
        if STATSMODELS_AVAILABLE:
            base_models.append('Enhanced SARIMA')
        
        base_forecasts = {}
        base_predictions = []
        
        # Generate base model forecasts for training
        if len(monthly_data) >= 18:
            train_size = len(monthly_data) - 6
            train_data = monthly_data[:train_size]
            val_data = monthly_data[train_size:]
            
            val_predictions = []
            for model in base_models:
                forecast_result = generate_enhanced_forecast(train_data, model, len(val_data), advanced_options)
                if forecast_result is not None:
                    val_predictions.append(forecast_result['Forecast'].values)
            
            if len(val_predictions) >= 2:
                # Train meta-learner
                X_meta = np.column_stack(val_predictions)
                y_meta = val_data['Revenue'].values
                
                meta_model = Ridge(alpha=1.0)
                meta_model.fit(X_meta, y_meta)
                
                # Generate final forecasts
                future_predictions = []
                for model in base_models:
                    forecast_result = generate_enhanced_forecast(monthly_data, model, periods, advanced_options)
                    if forecast_result is not None:
                        future_predictions.append(forecast_result['Forecast'].values)
                
                if len(future_predictions) >= 2:
                    X_future = np.column_stack(future_predictions)
                    stacked_forecast = meta_model.predict(X_future)
                    
                    last_date = monthly_data['YearMonth'].iloc[-1]
                    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
                    
                    return pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': stacked_forecast,
                        'Model': 'Stacked Ensemble'
                    })
        
        return None
        
    except Exception as e:
        return None

def bayesian_model_averaging_forecast(monthly_data, periods, advanced_options):
    """
    Bayesian Model Averaging for forecast combination
    """
    try:
        # Generate base forecasts
        base_models = ['Enhanced Linear Trend', 'Advanced Moving Average']
        if STATSMODELS_AVAILABLE:
            base_models.append('Enhanced SARIMA')
        
        forecasts = []
        weights = []
        
        for model in base_models:
            forecast_result = generate_enhanced_forecast(monthly_data, model, periods, advanced_options)
            if forecast_result is not None:
                forecasts.append(forecast_result['Forecast'].values)
                
                # Calculate model likelihood (simplified BIC-based weight)
                if len(monthly_data) >= 12:
                    train_size = len(monthly_data) - 3
                    train_data = monthly_data[:train_size]
                    test_data = monthly_data[train_size:]
                    
                    test_forecast = generate_enhanced_forecast(train_data, model, len(test_data), advanced_options)
                    if test_forecast is not None:
                        residuals = test_data['Revenue'].values - test_forecast['Forecast'].values
                        mse = np.mean(residuals**2)
                        bic = len(test_data) * np.log(mse) + 2 * np.log(len(test_data))
                        weight = np.exp(-0.5 * bic)
                        weights.append(weight)
                    else:
                        weights.append(1.0)
                else:
                    weights.append(1.0)
        
        if len(forecasts) >= 2:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Weighted average
            bma_forecast = np.average(forecasts, axis=0, weights=weights)
            
            last_date = monthly_data['YearMonth'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
            
            return pd.DataFrame({
                'Date': future_dates,
                'Forecast': bma_forecast,
                'Model': 'Bayesian Model Averaging'
            })
        
        return None
        
    except Exception as e:
        return None

def theta_method_forecast(monthly_data, periods, advanced_options):
    """
    Theta method forecast (simplified implementation)
    """
    try:
        ts = monthly_data['Revenue'].values
        
        # Theta method parameters
        theta = 2  # Standard theta value
        
        # Linear regression on time
        n = len(ts)
        x = np.arange(n)
        coeffs = np.polyfit(x, ts, 1)
        linear_trend = coeffs[0] * x + coeffs[1]
        
        # Theta line
        theta_line = theta * linear_trend + (1 - theta) * ts
        
        # Simple exponential smoothing on theta line
        alpha = 0.3  # Smoothing parameter
        smoothed = [theta_line[0]]
        
        for i in range(1, len(theta_line)):
            smoothed.append(alpha * theta_line[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast
        last_smoothed = smoothed[-1]
        trend = coeffs[0]
        
        predictions = []
        for i in range(1, periods + 1):
            pred = last_smoothed + trend * i
            predictions.append(pred)
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Theta Method'
        })
        
    except Exception as e:
        return None

def prophet_pro_forecast(monthly_data, periods, advanced_options):
    """
    Enhanced Prophet with additional regressors and components
    """
    try:
        # Prepare data for Prophet
        prophet_data = monthly_data[['YearMonth', 'Revenue']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Initialize Prophet with enhanced configuration
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            n_changepoints=25
        )
        
        # Add additional regressors if available
        if 'Customer_Count' in monthly_data.columns:
            prophet_data['customer_count'] = monthly_data['Customer_Count']
            model.add_regressor('customer_count')
        
        if 'Order_Count' in monthly_data.columns:
            prophet_data['order_count'] = monthly_data['Order_Count']
            model.add_regressor('order_count')
        
        # Add custom seasonalities
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        
        # Fit model
        model.fit(prophet_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='M')
        
        # Add future regressor values (simple extrapolation)
        if 'customer_count' in prophet_data.columns:
            last_customers = monthly_data['Customer_Count'].iloc[-1]
            future['customer_count'] = last_customers  # Simplified
        
        if 'order_count' in prophet_data.columns:
            last_orders = monthly_data['Order_Count'].iloc[-1]
            future['order_count'] = last_orders  # Simplified
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Return only future predictions
        future_forecast = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        future_forecast.columns = ['Date', 'Forecast', 'Lower_CI', 'Upper_CI']
        future_forecast['Model'] = 'Prophet Pro'
        
        return future_forecast
        
    except Exception as e:
        return None

def advanced_ets_forecast(monthly_data, periods, advanced_options):
    """
    Advanced ETS with automatic model selection
    """
    try:
        ts = monthly_data['Revenue'].values
        
        # Try different ETS configurations
        ets_configs = [
            {'error': 'add', 'trend': 'add', 'seasonal': 'add'},
            {'error': 'add', 'trend': 'add', 'seasonal': 'mul'},
            {'error': 'add', 'trend': 'mul', 'seasonal': 'add'},
            {'error': 'mul', 'trend': 'add', 'seasonal': 'add'},
        ]
        
        best_aic = np.inf
        best_model = None
        
        for config in ets_configs:
            try:
                model = ExponentialSmoothing(
                    ts,
                    trend=config['trend'],
                    seasonal=config['seasonal'],
                    seasonal_periods=12
                )
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
            except:
                continue
        
        if best_model is not None:
            # Generate forecast
            forecast = best_model.forecast(steps=periods)
            
            last_date = monthly_data['YearMonth'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
            
            return pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast,
                'Model': 'Advanced ETS'
            })
        
        return None
        
    except Exception as e:
        return None

def automl_ensemble_forecast(monthly_data, periods, advanced_options):
    """
    AutoML-style ensemble with automatic model selection
    """
    try:
        # Generate forecasts from multiple models
        model_forecasts = {}
        model_scores = {}
        
        # Define model candidates
        candidate_models = [
            'Enhanced Linear Trend', 'Advanced Moving Average', 'Random Forest Pro'
        ]
        
        if STATSMODELS_AVAILABLE:
            candidate_models.extend(['Enhanced SARIMA', 'Advanced ETS'])
        
        if XGBOOST_AVAILABLE:
            candidate_models.append('Auto XGBoost')
        
        # Evaluate each model
        for model in candidate_models:
            try:
                forecast_result = generate_enhanced_forecast(monthly_data, model, periods, advanced_options)
                if forecast_result is not None:
                    model_forecasts[model] = forecast_result
                    
                    # Calculate validation score
                    score = calculate_model_validation_score(monthly_data, model, advanced_options)
                    model_scores[model] = score
            except:
                continue
        
        if len(model_forecasts) >= 2:
            # Select top models based on scores
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            top_models = [model for model, score in sorted_models[:min(3, len(sorted_models))]]
            
            # Create weighted ensemble
            weights = []
            forecasts = []
            
            for model in top_models:
                if model in model_forecasts:
                    weight = model_scores[model]
                    weights.append(weight)
                    forecasts.append(model_forecasts[model]['Forecast'].values)
            
            if len(forecasts) >= 2:
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Weighted ensemble
                ensemble_forecast = np.average(forecasts, axis=0, weights=weights)
                
                last_date = monthly_data['YearMonth'].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
                
                return pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': ensemble_forecast,
                    'Model': 'AutoML Ensemble'
                })
        
        return None
        
    except Exception as e:
        return None

def calculate_model_validation_score(monthly_data, model_name, advanced_options):
    """
    Calculate validation score for model selection
    """
    try:
        if len(monthly_data) < 12:
            return 0
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, test_idx in tscv.split(monthly_data):
            train_data = monthly_data.iloc[train_idx]
            test_data = monthly_data.iloc[test_idx]
            
            forecast_result = generate_enhanced_forecast(train_data, model_name, len(test_data), advanced_options)
            
            if forecast_result is not None:
                actual = test_data['Revenue'].values
                predicted = forecast_result['Forecast'].values
                
                # Calculate score (1 / (1 + MAPE))
                mape = mean_absolute_percentage_error(actual, predicted)
                score = 1 / (1 + mape)
                scores.append(score)
        
        return np.mean(scores) if scores else 0
        
    except Exception as e:
        return 0

def display_enhanced_forecast_results(monthly_data, forecasts, model_performance, forecast_period, confidence_interval):
    """
    Enhanced display of forecast results with confidence intervals
    """
    st.subheader("📊 Enhanced Forecast Results")
    
    # Create advanced forecast chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Forecast Comparison', 'Model Performance'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        row_heights=[0.7, 0.3]
    )
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=monthly_data['YearMonth'],
            y=monthly_data['Revenue'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='blue', width=3)
        ),
        row=1, col=1
    )
    
    # Add forecasts with confidence intervals
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    for i, (model_name, forecast_df) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        
        # Main forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Forecast'],
                mode='lines+markers',
                name=f'{model_name}',
                line=dict(color=color, dash='dash', width=2)
            ),
            row=1, col=1
        )
        
        # Confidence intervals if available
        if 'Lower_CI' in forecast_df.columns and 'Upper_CI' in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Upper_CI'],
                    mode='lines',
                    line=dict(color=color, width=0),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Lower_CI'],
                    mode='lines',
                    line=dict(color=color, width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({color}, 0.2)',
                    name=f'{model_name} CI',
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Model performance chart
    if model_performance:
        models = list(model_performance.keys())
        mape_values = [model_performance[model].get('MAPE', 0) for model in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=mape_values,
                name='MAPE (%)',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title='Enhanced Sales Forecast Analysis',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="Revenue (SAR)", row=1, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced summary table
    st.subheader("📋 Detailed Forecast Summary")
    
    summary_data = []
    for model_name, forecast_df in forecasts.items():
        total_forecast = forecast_df['Forecast'].sum()
        avg_monthly = forecast_df['Forecast'].mean()
        forecast_std = forecast_df['Forecast'].std()
        
        performance = model_performance.get(model_name, {})
        mae = performance.get('MAE', 0)
        mape = performance.get('MAPE', 0)
        rmse = performance.get('RMSE', 0)
        
        summary_data.append({
            'Model': model_name,
            f'Total Forecast ({forecast_period})': f"{total_forecast:,.0f} SAR",
            'Avg Monthly': f"{avg_monthly:,.0f} SAR",
            'Std Dev': f"{forecast_std:,.0f} SAR",
            'MAE': f"{mae:,.0f} SAR",
            'MAPE': f"{mape:.1f}%",
            'RMSE': f"{rmse:,.0f} SAR"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

def generate_business_insights(monthly_data, forecasts, model_performance):
    """
    Generate automated business insights and recommendations
    """
    st.subheader("💡 Business Insights & Recommendations")
    
    # Calculate key metrics
    historical_avg = monthly_data['Revenue'].mean()
    historical_growth = monthly_data['Revenue'].pct_change().mean() * 100
    
    # Best performing model
    best_model = min(model_performance.keys(), key=lambda x: model_performance[x].get('MAPE', float('inf')))
    best_forecast = forecasts[best_model]['Forecast'].mean()
    
    # Growth projection
    growth_projection = (best_forecast - historical_avg) / historical_avg * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Key Insights")
        st.write(f"• **Best Model**: {best_model}")
        st.write(f"• **Historical Avg Growth**: {historical_growth:.1f}%/month")
        st.write(f"• **Projected Growth**: {growth_projection:.1f}%")
        
        if growth_projection > 5:
            st.success("🎯 **Strong growth projected** - Consider scaling operations")
        elif growth_projection < -5:
            st.warning("⚠️ **Decline projected** - Review market conditions")
        else:
            st.info("📊 **Stable growth projected** - Maintain current strategy")
    
    with col2:
        st.markdown("### 🎯 Recommendations")
        
        # Model reliability assessment
        best_mape = model_performance[best_model].get('MAPE', 0)
        
        if best_mape < 10:
            st.success("✅ **High confidence** in forecasts")
        elif best_mape < 20:
            st.warning("⚠️ **Moderate confidence** - Monitor closely")
        else:
            st.error("❌ **Low confidence** - Use with caution")
        
        st.write("**Action Items:**")
        st.write("• Review forecast monthly")
        st.write("• Update models with new data")
        st.write("• Monitor key leading indicators")

def get_forecast_periods(forecast_period):
    """Get number of periods to forecast"""
    period_map = {
        "Next 3 Months": 3,
        "Next 6 Months": 6,
        "Next 12 Months": 12,
        "Next 18 Months": 18
    }
    return period_map.get(forecast_period, 6)

# Additional functions for hierarchical forecasting, model comparison, etc.
def create_hierarchical_forecast(df, forecast_period, model_selection):
    """
    Multi-level hierarchical forecasting
    """
    st.subheader("🏗️ Multi-level Hierarchical Forecast")
    st.info("This advanced feature creates forecasts at multiple levels and ensures consistency across the hierarchy.")
    
    # Implementation would include top-down and bottom-up reconciliation
    # This is a placeholder for the full hierarchical forecasting implementation
    st.warning("Full hierarchical forecasting implementation would require additional development.")

# Original functions from the base code with enhancements would continue here...
# Including enhanced versions of all the display, comparison, and analysis functions
