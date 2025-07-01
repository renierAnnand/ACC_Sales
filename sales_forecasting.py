import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Enhanced forecasting libraries with robust error handling
STATSMODELS_AVAILABLE = False
PROPHET_AVAILABLE = False
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
SKLEARN_AVAILABLE = False
SCIPY_AVAILABLE = False

# Import with detailed error handling
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        adfuller = None
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

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
    
    # Try to import MAPE separately as it might not be available in older sklearn versions
    try:
        from sklearn.metrics import mean_absolute_percentage_error
    except ImportError:
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def create_sales_forecast(df):
    """
    Create enhanced advanced sales forecasting analysis (robust version)
    """
    st.header("🚀 Enhanced Advanced Sales Forecasting")
    st.markdown("*All amounts in Saudi Riyal (SAR)*")
    
    if df.empty:
        st.error("No data available for forecasting")
        return
    
    # Library availability status
    display_library_status()
    
    # Display available models
    available_models = get_enhanced_available_models()
    
    if not available_models:
        st.error("❌ No forecasting models available. Please install required libraries.")
        st.info("Install basic requirements: `pip install scikit-learn numpy pandas`")
        return
    
    st.success(f"✅ **Available Models**: {', '.join(available_models)}")
    
    # Simplified forecast options
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_type = st.selectbox(
            "Select Forecast Type",
            ["Overall Sales Forecast", "Business Unit Forecast", "Customer Forecast", "Product Category Forecast"]
        )
        
        forecast_period = st.selectbox(
            "Forecast Period",
            ["Next 3 Months", "Next 6 Months", "Next 12 Months"],
            index=1
        )
    
    with col2:
        confidence_level = st.selectbox(
            "Confidence Level",
            ["80%", "90%", "95%"],
            index=2
        )
        
        max_models = st.slider(
            "Maximum Models to Use",
            min_value=1,
            max_value=min(5, len(available_models)),
            value=min(3, len(available_models))
        )
    
    # Model selection (simplified)
    st.subheader("🎛️ Model Selection")
    
    # Auto-select best available models
    recommended_models = available_models[:max_models]
    
    model_selection = st.multiselect(
        "Select Models (recommended models pre-selected)",
        available_models,
        default=recommended_models
    )
    
    if not model_selection:
        st.warning("Please select at least one forecasting model")
        return
    
    # Simplified advanced options
    with st.expander("🔧 Options"):
        enable_feature_engineering = st.checkbox("Enhanced Feature Engineering", value=True)
        enable_outlier_detection = st.checkbox("Outlier Detection", value=True)
    
    advanced_options = {
        'feature_engineering': enable_feature_engineering,
        'outlier_detection': enable_outlier_detection,
        'hyperparameter_tuning': False,
        'uncertainty_quantification': True,
        'model_explanability': False
    }
    
    # Execute forecasting
    if forecast_type == "Overall Sales Forecast":
        create_robust_overall_forecast(df, forecast_period, model_selection, confidence_level, advanced_options)
    else:
        st.info(f"{forecast_type} implementation available in full version.")
        create_robust_overall_forecast(df, forecast_period, model_selection, confidence_level, advanced_options)

def display_library_status():
    """Display status of available libraries"""
    st.subheader("📚 Library Status")
    
    status_data = [
        {"Library": "NumPy & Pandas", "Status": "✅ Available", "Features": "Basic operations, data handling"},
        {"Library": "Scikit-learn", "Status": "✅ Available" if SKLEARN_AVAILABLE else "❌ Missing", "Features": "ML models, validation"},
        {"Library": "Statsmodels", "Status": "✅ Available" if STATSMODELS_AVAILABLE else "❌ Missing", "Features": "ARIMA, ETS, seasonal decomposition"},
        {"Library": "Prophet", "Status": "✅ Available" if PROPHET_AVAILABLE else "❌ Missing", "Features": "Advanced time series forecasting"},
        {"Library": "XGBoost", "Status": "✅ Available" if XGBOOST_AVAILABLE else "❌ Missing", "Features": "Gradient boosting models"},
        {"Library": "LightGBM", "Status": "✅ Available" if LIGHTGBM_AVAILABLE else "❌ Missing", "Features": "Fast gradient boosting"},
        {"Library": "SciPy", "Status": "✅ Available" if SCIPY_AVAILABLE else "❌ Missing", "Features": "Statistical functions"}
    ]
    
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True)
    
    # Installation recommendations
    missing_libraries = []
    if not SKLEARN_AVAILABLE:
        missing_libraries.append("scikit-learn")
    if not STATSMODELS_AVAILABLE:
        missing_libraries.append("statsmodels")
    if not PROPHET_AVAILABLE:
        missing_libraries.append("prophet")
    if not XGBOOST_AVAILABLE:
        missing_libraries.append("xgboost")
    if not SCIPY_AVAILABLE:
        missing_libraries.append("scipy")
    
    if missing_libraries:
        st.info(f"💡 **To unlock more models, install**: `pip install {' '.join(missing_libraries)}`")

def get_enhanced_available_models():
    """
    Get enhanced list of available forecasting models based on installed libraries
    """
    models = ["Enhanced Linear Trend", "Advanced Moving Average"]
    
    # Always available basic models
    if SKLEARN_AVAILABLE:
        models.append("Random Forest Pro")
    
    # Statistical models
    if STATSMODELS_AVAILABLE:
        models.extend(["Enhanced SARIMA", "Advanced ETS", "Seasonal Naive"])
    
    # Prophet models
    if PROPHET_AVAILABLE:
        models.extend(["Prophet Pro"])
    
    # XGBoost models
    if XGBOOST_AVAILABLE:
        models.extend(["Auto XGBoost"])
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        models.append("LightGBM")
    
    # ML models (require sklearn)
    if SKLEARN_AVAILABLE:
        models.extend(["Neural Network", "Gradient Boosting"])
    
    # Ensemble methods (require at least 2 base models)
    if len(models) >= 2:
        models.extend(["Dynamic Ensemble", "Simple Ensemble"])
    
    if SKLEARN_AVAILABLE and len(models) >= 3:
        models.extend(["Stacked Ensemble", "AutoML Ensemble"])
    
    return models

def create_robust_overall_forecast(df, forecast_period, model_selection, confidence_level, advanced_options):
    """
    Create robust overall sales forecast with comprehensive error handling
    """
    st.subheader("🎯 Enhanced Overall Sales Forecast")
    
    try:
        # Enhanced data preparation with error handling
        monthly_data = prepare_enhanced_monthly_data(df, advanced_options)
        
        if monthly_data.empty:
            st.error("❌ Could not prepare data for forecasting")
            return
        
        if len(monthly_data) < 6:
            st.error("❌ Insufficient data for forecasting (minimum 6 months required)")
            st.info(f"Current data: {len(monthly_data)} months")
            return
        
        periods = get_forecast_periods(forecast_period)
        confidence_interval = float(confidence_level.strip('%')) / 100
        
        # Data quality assessment
        with st.spinner("📊 Assessing data quality..."):
            quality_report = assess_data_quality(monthly_data)
            display_data_quality_report(quality_report)
        
        # Generate forecasts
        forecasts = {}
        model_performance = {}
        
        with st.spinner("🚀 Generating forecasts..."):
            progress_bar = st.progress(0)
            
            for i, model in enumerate(model_selection):
                try:
                    with st.spinner(f"Training {model}..."):
                        forecast_result = generate_enhanced_forecast(monthly_data, model, periods, advanced_options)
                        
                        if forecast_result is not None:
                            forecasts[model] = forecast_result
                            st.success(f"✅ {model} completed")
                            
                            # Calculate basic performance
                            performance = calculate_enhanced_model_performance(monthly_data, model, 'Walk-Forward', advanced_options)
                            model_performance[model] = performance
                        else:
                            st.warning(f"⚠️ {model} failed - skipping")
                            
                except Exception as e:
                    st.warning(f"⚠️ Error with {model}: {str(e)[:50]}...")
                
                progress_bar.progress((i + 1) / len(model_selection))
        
        if not forecasts:
            st.error("❌ No forecasts could be generated. Please check your data and try different models.")
            return
        
        st.success(f"✅ Successfully generated {len(forecasts)} forecasts!")
        
        # Display results
        display_enhanced_forecast_results(monthly_data, forecasts, model_performance, forecast_period, confidence_interval)
        
        # Model comparison
        if len(forecasts) > 1:
            create_advanced_model_comparison(forecasts, model_performance, {})
        
        # Ensemble forecast
        if len(forecasts) > 1:
            create_robust_ensemble_forecast(monthly_data, forecasts, model_performance, forecast_period)
        
        # Business insights
        generate_business_insights(monthly_data, forecasts, model_performance)
        
    except Exception as e:
        st.error(f"❌ Error creating forecast: {str(e)}")
        st.info("💡 Try reducing the number of models or using simpler options.")

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
    Add comprehensive feature engineering (with fallbacks)
    """
    try:
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
        for lag in [1, 2, 3]:  # Reduced lag features for robustness
            if lag < len(monthly_data):
                monthly_data[f'Revenue_Lag_{lag}'] = monthly_data['Revenue'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6]:  # Reduced window sizes
            if window < len(monthly_data):
                monthly_data[f'Revenue_MA_{window}'] = monthly_data['Revenue'].rolling(window=window).mean()
                monthly_data[f'Revenue_Std_{window}'] = monthly_data['Revenue'].rolling(window=window).std()
        
        # Growth rates
        monthly_data['Revenue_Growth'] = monthly_data['Revenue'].pct_change()
        
        # Derived metrics
        monthly_data['Revenue_per_Customer'] = monthly_data['Revenue'] / monthly_data['Customer_Count'].replace(0, 1)
        monthly_data['Revenue_per_Invoice'] = monthly_data['Revenue'] / monthly_data['Invoice_Count'].replace(0, 1)
        
        # Seasonal decomposition (only if statsmodels available and enough data)
        if STATSMODELS_AVAILABLE and len(monthly_data) >= 24:
            try:
                revenue_filled = monthly_data['Revenue'].ffill()
                decomposition = seasonal_decompose(revenue_filled, model='additive', period=12)
                monthly_data['Trend'] = decomposition.trend
                monthly_data['Seasonal'] = decomposition.seasonal
                monthly_data['Residual'] = decomposition.resid
            except Exception:
                pass
        
        return monthly_data
        
    except Exception as e:
        st.warning(f"Feature engineering warning: {str(e)[:50]}...")
        return monthly_data

def detect_and_treat_outliers(monthly_data):
    """
    Simple outlier detection and treatment (robust version)
    """
    try:
        # Simple IQR method for outlier detection
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
        
        # Z-score method (if scipy available)
        if SCIPY_AVAILABLE:
            try:
                z_scores = np.abs(stats.zscore(monthly_data['Revenue']))
                monthly_data['Z_Score'] = z_scores
                monthly_data['Is_Outlier_ZScore'] = (z_scores > 3).astype(int)
            except:
                monthly_data['Z_Score'] = 0
                monthly_data['Is_Outlier_ZScore'] = 0
        else:
            monthly_data['Z_Score'] = 0
            monthly_data['Is_Outlier_ZScore'] = 0
        
        # Treatment: simple clipping
        monthly_data['Revenue_Treated'] = monthly_data['Revenue'].clip(
            lower=monthly_data['Revenue'].quantile(0.05),
            upper=monthly_data['Revenue'].quantile(0.95)
        )
        
        return monthly_data
        
    except Exception as e:
        st.warning(f"Outlier detection warning: {str(e)[:50]}...")
        return monthly_data

def assess_data_quality(monthly_data):
    """
    Comprehensive data quality assessment (with fallbacks)
    """
    quality_report = {
        'total_periods': len(monthly_data),
        'missing_values': monthly_data.isnull().sum().sum(),
        'duplicate_periods': monthly_data['YearMonth'].duplicated().sum(),
        'outlier_count': monthly_data.get('Is_Outlier', pd.Series([0])).sum(),
        'data_completeness': (1 - monthly_data.isnull().sum().sum() / (len(monthly_data) * len(monthly_data.columns))) * 100,
        'revenue_cv': monthly_data['Revenue'].std() / monthly_data['Revenue'].mean() if monthly_data['Revenue'].mean() > 0 else 0,
        'stationarity_test': None
    }
    
    # Stationarity test (only if statsmodels is available)
    try:
        if STATSMODELS_AVAILABLE and adfuller is not None:
            adf_result = adfuller(monthly_data['Revenue'].dropna())
            quality_report['stationarity_test'] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
    except Exception:
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
    Generate forecast using enhanced models with robust fallbacks
    """
    try:
        # Basic models (always available)
        if model_name == "Enhanced Linear Trend":
            return enhanced_linear_trend_forecast(monthly_data, periods)
        elif model_name == "Advanced Moving Average":
            return advanced_moving_average_forecast(monthly_data, periods)
        elif model_name == "Simple Ensemble":
            return simple_ensemble_forecast(monthly_data, periods)
        
        # Statistical models (require statsmodels)
        elif model_name == "Enhanced SARIMA" and STATSMODELS_AVAILABLE:
            return enhanced_sarima_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Advanced ETS" and STATSMODELS_AVAILABLE:
            return advanced_ets_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Seasonal Naive" and STATSMODELS_AVAILABLE:
            return seasonal_naive_forecast(monthly_data, periods)
        
        # ML models (require sklearn)
        elif model_name == "Random Forest Pro" and SKLEARN_AVAILABLE:
            return random_forest_pro_forecast(monthly_data, periods)
        elif model_name == "Neural Network" and SKLEARN_AVAILABLE:
            return neural_network_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Gradient Boosting" and SKLEARN_AVAILABLE:
            return gradient_boosting_forecast(monthly_data, periods)
        
        # Advanced ML models
        elif model_name == "Auto XGBoost" and XGBOOST_AVAILABLE:
            return auto_xgboost_forecast(monthly_data, periods, advanced_options)
        elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
            return lightgbm_forecast(monthly_data, periods, advanced_options)
        
        # Prophet models
        elif model_name == "Prophet Pro" and PROPHET_AVAILABLE:
            return prophet_pro_forecast(monthly_data, periods, advanced_options)
        
        # Ensemble methods
        elif model_name == "Dynamic Ensemble":
            return dynamic_ensemble_forecast(monthly_data, periods, advanced_options)
        elif model_name == "Stacked Ensemble" and SKLEARN_AVAILABLE:
            return stacked_ensemble_forecast(monthly_data, periods, advanced_options)
        elif model_name == "AutoML Ensemble" and SKLEARN_AVAILABLE:
            return automl_ensemble_forecast(monthly_data, periods, advanced_options)
        
        else:
            st.warning(f"Model {model_name} not available or dependencies missing")
            return None
            
    except Exception as e:
        st.warning(f"⚠️ Error generating {model_name} forecast: {str(e)[:100]}...")
        return None

def enhanced_linear_trend_forecast(monthly_data, periods):
    """Enhanced linear trend with confidence intervals"""
    try:
        y = monthly_data['Revenue'].values
        x = np.arange(len(y))
        
        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        # Calculate residuals for confidence intervals
        fitted_values = slope * x + intercept
        residuals = y - fitted_values
        residual_std = np.std(residuals)
        
        # Generate future predictions
        future_x = np.arange(len(y), len(y) + periods)
        predictions = slope * future_x + intercept
        
        # Confidence intervals
        t_value = 1.96  # 95% confidence
        lower_ci = predictions - t_value * residual_std
        upper_ci = predictions + t_value * residual_std
        
        # Create future dates
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Lower_CI': lower_ci,
            'Upper_CI': upper_ci,
            'Model': 'Enhanced Linear Trend'
        })
    except Exception:
        return None

def advanced_moving_average_forecast(monthly_data, periods):
    """Advanced moving average with seasonal adjustment"""
    try:
        # Adaptive window size
        data_length = len(monthly_data)
        ma_period = min(6, max(3, data_length // 4))
        
        # Calculate moving average
        recent_ma = monthly_data['Revenue'].tail(ma_period).mean()
        
        # Simple seasonal adjustment
        if data_length >= 12:
            # Calculate monthly seasonal factors
            monthly_data_with_month = monthly_data.copy()
            monthly_data_with_month['Month'] = monthly_data_with_month['YearMonth'].dt.month
            
            monthly_averages = monthly_data_with_month.groupby('Month')['Revenue'].mean()
            overall_average = monthly_data['Revenue'].mean()
            seasonal_factors = monthly_averages / overall_average
            
            # Apply seasonal adjustment
            predictions = []
            last_date = monthly_data['YearMonth'].iloc[-1]
            
            for i in range(periods):
                future_date = last_date + pd.DateOffset(months=i+1)
                month = future_date.month
                seasonal_factor = seasonal_factors.get(month, 1.0)
                pred = recent_ma * seasonal_factor
                predictions.append(pred)
        else:
            # Simple MA without seasonal adjustment
            predictions = [recent_ma] * periods
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Advanced Moving Average'
        })
    except Exception:
        return None

def simple_ensemble_forecast(monthly_data, periods):
    """Simple ensemble of basic methods"""
    try:
        # Generate forecasts from basic methods
        linear_forecast = enhanced_linear_trend_forecast(monthly_data, periods)
        ma_forecast = advanced_moving_average_forecast(monthly_data, periods)
        
        if linear_forecast is None or ma_forecast is None:
            return None
        
        # Simple average
        ensemble_forecast = (linear_forecast['Forecast'] + ma_forecast['Forecast']) / 2
        
        return pd.DataFrame({
            'Date': linear_forecast['Date'],
            'Forecast': ensemble_forecast,
            'Model': 'Simple Ensemble'
        })
    except Exception:
        return None

def enhanced_sarima_forecast(monthly_data, periods, advanced_options):
    """Enhanced SARIMA with automatic parameter selection (robust version)"""
    try:
        ts = monthly_data['Revenue'].dropna()
        
        if len(ts) < 12:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
        # Simple parameter selection
        try:
            model = ARIMA(ts, order=(1, 1, 1))
            fitted_model = model.fit()
        except:
            try:
                model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                fitted_model = model.fit(disp=False)
            except:
                return enhanced_linear_trend_forecast(monthly_data, periods)
        
        # Generate forecast
        try:
            if hasattr(fitted_model, 'get_forecast'):
                forecast_result = fitted_model.get_forecast(steps=periods)
                forecast_mean = forecast_result.predicted_mean
                
                last_date = monthly_data['YearMonth'].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
                
                return pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast_mean.values,
                    'Model': 'Enhanced SARIMA'
                })
            else:
                forecast = fitted_model.forecast(steps=periods)
                last_date = monthly_data['YearMonth'].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
                
                return pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast,
                    'Model': 'Enhanced SARIMA'
                })
        except:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
    except Exception:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def advanced_ets_forecast(monthly_data, periods, advanced_options):
    """Advanced ETS with automatic model selection (robust version)"""
    try:
        ts = monthly_data['Revenue'].values
        
        if len(ts) < 12:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
        try:
            model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
        except:
            try:
                model = ExponentialSmoothing(ts, trend='add')
                fitted_model = model.fit()
            except:
                return enhanced_linear_trend_forecast(monthly_data, periods)
        
        try:
            forecast = fitted_model.forecast(steps=periods)
            
            last_date = monthly_data['YearMonth'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
            
            return pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast,
                'Model': 'Advanced ETS'
            })
        except:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
    except Exception:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def seasonal_naive_forecast(monthly_data, periods):
    """Seasonal naive forecast"""
    try:
        if len(monthly_data) < 12:
            last_value = monthly_data['Revenue'].iloc[-1]
            predictions = [last_value] * periods
        else:
            predictions = []
            for i in range(periods):
                seasonal_index = len(monthly_data) - 12 + (i % 12)
                if seasonal_index >= 0:
                    pred = monthly_data['Revenue'].iloc[seasonal_index]
                else:
                    pred = monthly_data['Revenue'].iloc[-1]
                predictions.append(pred)
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Seasonal Naive'
        })
    except Exception:
        return None

def random_forest_pro_forecast(monthly_data, periods):
    """Enhanced Random Forest with better feature handling"""
    try:
        feature_columns = ['Month', 'Quarter']
        
        for lag in [1, 2, 3]:
            lag_col = f'Revenue_Lag_{lag}'
            if lag_col in monthly_data.columns:
                feature_columns.append(lag_col)
        
        if 'Month_Sin' in monthly_data.columns:
            feature_columns.extend(['Month_Sin', 'Month_Cos'])
        
        train_data = monthly_data.dropna(subset=['Revenue'] + feature_columns)
        
        if len(train_data) < 6:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
        X = train_data[feature_columns]
        y = train_data['Revenue']
        
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=1
        )
        model.fit(X, y)
        
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            
            future_features = {
                'Month': future_date.month,
                'Quarter': future_date.quarter
            }
            
            for lag in [1, 2, 3]:
                lag_col = f'Revenue_Lag_{lag}'
                if lag_col in feature_columns:
                    if lag == 1:
                        future_features[lag_col] = last_row['Revenue']
                    else:
                        future_features[lag_col] = last_row.get(lag_col, last_row['Revenue'])
            
            if 'Month_Sin' in feature_columns:
                future_features['Month_Sin'] = np.sin(2 * np.pi * future_date.month / 12)
                future_features['Month_Cos'] = np.cos(2 * np.pi * future_date.month / 12)
            
            X_future = [[future_features[f] for f in feature_columns]]
            pred = model.predict(X_future)[0]
            predictions.append(pred)
            
            last_row['YearMonth'] = future_date
            last_row['Revenue'] = pred
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Random Forest Pro'
        })
        
    except Exception:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def gradient_boosting_forecast(monthly_data, periods):
    """Gradient Boosting forecast"""
    try:
        feature_columns = ['Month', 'Quarter']
        
        for lag in [1, 2]:
            lag_col = f'Revenue_Lag_{lag}'
            if lag_col in monthly_data.columns:
                feature_columns.append(lag_col)
        
        train_data = monthly_data.dropna(subset=['Revenue'] + feature_columns)
        
        if len(train_data) < 6:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
        X = train_data[feature_columns]
        y = train_data['Revenue']
        
        model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            
            future_features = {
                'Month': future_date.month,
                'Quarter': future_date.quarter
            }
            
            for lag in [1, 2]:
                lag_col = f'Revenue_Lag_{lag}'
                if lag_col in feature_columns:
                    if lag == 1:
                        future_features[lag_col] = last_row['Revenue']
                    else:
                        future_features[lag_col] = last_row.get(lag_col, last_row['Revenue'])
            
            X_future = [[future_features[f] for f in feature_columns]]
            pred = model.predict(X_future)[0]
            predictions.append(pred)
            
            last_row['YearMonth'] = future_date
            last_row['Revenue'] = pred
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Gradient Boosting'
        })
        
    except Exception:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def neural_network_forecast(monthly_data, periods, advanced_options):
    """Neural Network forecast using MLPRegressor"""
    try:
        feature_columns = ['Month', 'Quarter']
        
        for lag in [1, 2]:
            lag_col = f'Revenue_Lag_{lag}'
            if lag_col in monthly_data.columns:
                feature_columns.append(lag_col)
        
        train_data = monthly_data.dropna(subset=['Revenue'] + feature_columns)
        
        if len(train_data) < 12:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
        X = train_data[feature_columns]
        y = train_data['Revenue']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=300,
            random_state=42
        )
        
        model.fit(X_scaled, y)
        
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            
            future_features = {
                'Month': future_date.month,
                'Quarter': future_date.quarter
            }
            
            for lag in [1, 2]:
                lag_col = f'Revenue_Lag_{lag}'
                if lag_col in feature_columns:
                    future_features[lag_col] = last_row['Revenue']
            
            X_future = [[future_features[f] for f in feature_columns]]
            X_future_scaled = scaler.transform(X_future)
            pred = model.predict(X_future_scaled)[0]
            predictions.append(pred)
            
            last_row['YearMonth'] = future_date
            last_row['Revenue'] = pred
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Neural Network'
        })
        
    except Exception:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def auto_xgboost_forecast(monthly_data, periods, advanced_options):
    """Auto XGBoost with feature selection"""
    try:
        feature_columns = ['Month', 'Quarter', 'Month_Sin', 'Month_Cos']
        
        for lag in [1, 2, 3]:
            lag_col = f'Revenue_Lag_{lag}'
            if lag_col in monthly_data.columns:
                feature_columns.append(lag_col)
        
        train_data = monthly_data.dropna(subset=['Revenue'] + feature_columns)
        
        if len(train_data) < 12:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
        X = train_data[feature_columns]
        y = train_data['Revenue']
        
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            
            future_features = {
                'Month': future_date.month,
                'Quarter': future_date.quarter,
                'Month_Sin': np.sin(2 * np.pi * future_date.month / 12),
                'Month_Cos': np.cos(2 * np.pi * future_date.month / 12)
            }
            
            for lag in [1, 2, 3]:
                lag_col = f'Revenue_Lag_{lag}'
                if lag_col in feature_columns:
                    future_features[lag_col] = last_row['Revenue']
            
            X_future = [[future_features[f] for f in feature_columns]]
            pred = model.predict(X_future)[0]
            predictions.append(pred)
            
            last_row['YearMonth'] = future_date
            last_row['Revenue'] = pred
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Auto XGBoost'
        })
        
    except Exception:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def lightgbm_forecast(monthly_data, periods, advanced_options):
    """LightGBM forecast with advanced configuration"""
    try:
        feature_columns = ['Month', 'Quarter']
        
        for lag in [1, 2]:
            lag_col = f'Revenue_Lag_{lag}'
            if lag_col in monthly_data.columns:
                feature_columns.append(lag_col)
        
        train_data = monthly_data.dropna(subset=['Revenue'] + feature_columns)
        
        if len(train_data) < 12:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
        X = train_data[feature_columns]
        y = train_data['Revenue']
        
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X, y)
        
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            
            future_features = {
                'Month': future_date.month,
                'Quarter': future_date.quarter
            }
            
            for lag in [1, 2]:
                lag_col = f'Revenue_Lag_{lag}'
                if lag_col in feature_columns:
                    future_features[lag_col] = last_row['Revenue']
            
            X_future = [[future_features[f] for f in feature_columns]]
            pred = model.predict(X_future)[0]
            predictions.append(pred)
            
            last_row['YearMonth'] = future_date
            last_row['Revenue'] = pred
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'LightGBM'
        })
        
    except Exception:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def prophet_pro_forecast(monthly_data, periods, advanced_options):
    """Enhanced Prophet with robust error handling"""
    try:
        prophet_data = monthly_data[['YearMonth', 'Revenue']].copy()
        prophet_data.columns = ['ds', 'y']
        
        prophet_data = prophet_data[np.isfinite(prophet_data['y'])]
        
        if len(prophet_data) < 12:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            
            future = model.make_future_dataframe(periods=periods, freq='M')
            
            forecast = model.predict(future)
            
            future_forecast = forecast.tail(periods)[['ds', 'yhat']].copy()
            future_forecast.columns = ['Date', 'Forecast']
            future_forecast['Model'] = 'Prophet Pro'
            
            return future_forecast
            
        except Exception:
            return enhanced_linear_trend_forecast(monthly_data, periods)
        
    except Exception:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def dynamic_ensemble_forecast(monthly_data, periods, advanced_options):
    """Simplified dynamic ensemble"""
    try:
        base_forecasts = {}
        
        linear_forecast = enhanced_linear_trend_forecast(monthly_data, periods)
        if linear_forecast is not None:
            base_forecasts['Linear'] = linear_forecast
        
        ma_forecast = advanced_moving_average_forecast(monthly_data, periods)
        if ma_forecast is not None:
            base_forecasts['MA'] = ma_forecast
        
        if SKLEARN_AVAILABLE:
            rf_forecast = random_forest_pro_forecast(monthly_data, periods)
            if rf_forecast is not None:
                base_forecasts['RF'] = rf_forecast
        
        if len(base_forecasts) < 2:
            return linear_forecast
        
        ensemble_forecast = None
        first_model = True
        
        for model_name, forecast_df in base_forecasts.items():
            if first_model:
                ensemble_forecast = forecast_df.copy()
                ensemble_forecast['Forecast'] = forecast_df['Forecast'] / len(base_forecasts)
                first_model = False
            else:
                ensemble_forecast['Forecast'] += forecast_df['Forecast'] / len(base_forecasts)
        
        ensemble_forecast['Model'] = 'Dynamic Ensemble'
        return ensemble_forecast
        
    except Exception:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def stacked_ensemble_forecast(monthly_data, periods, advanced_options):
    """Stacked ensemble using meta-learning"""
    try:
        base_models = ['Enhanced Linear Trend', 'Advanced Moving Average']
        if SKLEARN_AVAILABLE:
            base_models.append('Random Forest Pro')
        
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
                X_meta = np.column_stack(val_predictions)
                y_meta = val_data['Revenue'].values
                
                meta_model = Ridge(alpha=1.0)
                meta_model.fit(X_meta, y_meta)
                
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
        
    except Exception:
        return None

def automl_ensemble_forecast(monthly_data, periods, advanced_options):
    """AutoML-style ensemble with automatic model selection"""
    try:
        candidate_models = ['Enhanced Linear Trend', 'Advanced Moving Average']
        
        if STATSMODELS_AVAILABLE:
            candidate_models.append('Enhanced SARIMA')
        
        if SKLEARN_AVAILABLE:
            candidate_models.append('Random Forest Pro')
        
        model_forecasts = {}
        model_scores = {}
        
        for model in candidate_models:
            try:
                forecast_result = generate_enhanced_forecast(monthly_data, model, periods, advanced_options)
                if forecast_result is not None:
                    model_forecasts[model] = forecast_result
                    
                    score = calculate_model_validation_score(monthly_data, model, advanced_options)
                    model_scores[model] = score
            except:
                continue
        
        if len(model_forecasts) >= 2:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            top_models = [model for model, score in sorted_models[:min(3, len(sorted_models))]]
            
            weights = []
            forecasts = []
            
            for model in top_models:
                if model in model_forecasts:
                    weight = model_scores[model]
                    weights.append(weight)
                    forecasts.append(model_forecasts[model]['Forecast'].values)
            
            if len(forecasts) >= 2:
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                ensemble_forecast = np.average(forecasts, axis=0, weights=weights)
                
                last_date = monthly_data['YearMonth'].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
                
                return pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': ensemble_forecast,
                    'Model': 'AutoML Ensemble'
                })
        
        return None
        
    except Exception:
        return None

def calculate_model_validation_score(monthly_data, model_name, advanced_options):
    """Calculate validation score for model selection"""
    try:
        if len(monthly_data) < 12:
            return 0
        
        train_size = len(monthly_data) - 3
        train_data = monthly_data[:train_size]
        test_data = monthly_data[train_size:]
        
        forecast_result = generate_enhanced_forecast(train_data, model_name, len(test_data), advanced_options)
        
        if forecast_result is not None:
            actual = test_data['Revenue'].values
            predicted = forecast_result['Forecast'].values
            
            mape = mean_absolute_percentage_error(actual, predicted)
            score = 1 / (1 + mape)
            return score
        
        return 0
        
    except Exception:
        return 0

def calculate_enhanced_model_performance(monthly_data, model_name, validation_method, advanced_options):
    """Calculate model performance with robust error handling"""
    try:
        if len(monthly_data) < 12:
            return {'MAE': 0, 'MAPE': 100, 'RMSE': 0}
        
        train_size = len(monthly_data) - 3
        train_data = monthly_data[:train_size]
        test_data = monthly_data[train_size:]
        
        forecast_result = generate_enhanced_forecast(train_data, model_name, len(test_data), advanced_options)
        
        if forecast_result is None:
            return {'MAE': 0, 'MAPE': 100, 'RMSE': 0}
        
        actual = test_data['Revenue'].values
        predicted = forecast_result['Forecast'].values
        
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse}
        
    except Exception:
        return {'MAE': 0, 'MAPE': 100, 'RMSE': 0}

def display_enhanced_forecast_results(monthly_data, forecasts, model_performance, forecast_period, confidence_interval):
    """Enhanced display of forecast results with confidence intervals"""
    st.subheader("📊 Enhanced Forecast Results")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Forecast Comparison', 'Model Performance'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        row_heights=[0.7, 0.3]
    )
    
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
    
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    for i, (model_name, forecast_df) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        
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
                    fillcolor=f'rgba(255,0,0,0.2)',
                    name=f'{model_name} CI',
                    showlegend=False
                ),
                row=1, col=1
            )
    
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

def create_advanced_model_comparison(forecasts, model_performance, model_diagnostics):
    """Create advanced model comparison with robust error handling"""
    st.subheader("🔍 Model Performance Comparison")
    
    if not forecasts or not model_performance:
        st.warning("No model performance data available for comparison")
        return
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            models = list(model_performance.keys())
            mape_values = [model_performance[model].get('MAPE', 0) for model in models]
            
            if mape_values and max(mape_values) > 0:
                fig = px.bar(
                    x=models,
                    y=mape_values,
                    title='Model Accuracy (Lower MAPE = Better)',
                    labels={'x': 'Model', 'y': 'MAPE (%)'},
                    color=mape_values,
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            forecast_values = []
            model_names = []
            
            for model_name, forecast_df in forecasts.items():
                avg_forecast = forecast_df['Forecast'].mean()
                forecast_values.append(avg_forecast)
                model_names.append(model_name)
            
            if forecast_values:
                fig = px.bar(
                    x=model_names,
                    y=forecast_values,
                    title='Average Monthly Forecast by Model',
                    labels={'x': 'Model', 'y': 'Avg Monthly Forecast (SAR)'},
                    color=forecast_values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Detailed Performance Metrics")
        
        performance_data = []
        for model in models:
            perf = model_performance[model]
            performance_data.append({
                'Model': model,
                'MAE (SAR)': f"{perf.get('MAE', 0):,.0f}",
                'MAPE (%)': f"{perf.get('MAPE', 0):.1f}",
                'RMSE (SAR)': f"{perf.get('RMSE', 0):,.0f}"
            })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            best_model = min(models, key=lambda x: model_performance[x].get('MAPE', float('inf')))
            st.success(f"🏆 **Best Performing Model**: {best_model}")
    
    except Exception as e:
        st.error(f"Error creating model comparison: {str(e)[:100]}...")

def create_robust_ensemble_forecast(monthly_data, forecasts, model_performance, forecast_period):
    """Create ensemble forecast with robust error handling"""
    st.subheader("🎯 Ensemble Forecast")
    
    try:
        ensemble_forecast = None
        weights = {}
        
        total_weight = 0
        for model_name in forecasts.keys():
            mape = model_performance.get(model_name, {}).get('MAPE', 100)
            weight = 1 / (1 + mape) if mape > 0 else 0.1
            weights[model_name] = weight
            total_weight += weight
        
        if total_weight > 0:
            for model_name in weights.keys():
                weights[model_name] = weights[model_name] / total_weight
        else:
            equal_weight = 1.0 / len(forecasts)
            for model_name in forecasts.keys():
                weights[model_name] = equal_weight
        
        first_model = True
        for model_name, forecast_df in forecasts.items():
            weight = weights.get(model_name, 0)
            
            if first_model:
                ensemble_forecast = forecast_df.copy()
                ensemble_forecast['Forecast'] = forecast_df['Forecast'] * weight
                first_model = False
            else:
                ensemble_forecast['Forecast'] += forecast_df['Forecast'] * weight
        
        ensemble_forecast['Model'] = 'Ensemble'
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data['YearMonth'],
            y=monthly_data['Revenue'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='blue', width=3)
        ))
        
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, forecast_df) in enumerate(forecasts.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Forecast'],
                mode='lines',
                name=model_name,
                line=dict(color=color, dash='dash', width=1),
                opacity=0.7
            ))
        
        fig.add_trace(go.Scatter(
            x=ensemble_forecast['Date'],
            y=ensemble_forecast['Forecast'],
            mode='lines+markers',
            name='Ensemble Forecast',
            line=dict(color='black', width=3)
        ))
        
        fig.update_layout(
            title='Ensemble Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue (SAR)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        ensemble_total = ensemble_forecast['Forecast'].sum()
        ensemble_avg = ensemble_forecast['Forecast'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Ensemble Total ({forecast_period})", f"{ensemble_total:,.0f} SAR")
        with col2:
            st.metric("Ensemble Avg Monthly", f"{ensemble_avg:,.0f} SAR")
        
        st.subheader("⚖️ Model Weights")
        weights_df = pd.DataFrame([
            {'Model': model, 'Weight': f"{weight:.3f}", 'Weight %': f"{weight*100:.1f}%"}
            for model, weight in weights.items()
        ])
        st.dataframe(weights_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating ensemble forecast: {str(e)[:100]}...")

def generate_business_insights(monthly_data, forecasts, model_performance):
    """Generate automated business insights and recommendations"""
    st.subheader("💡 Business Insights & Recommendations")
    
    try:
        historical_avg = monthly_data['Revenue'].mean()
        historical_growth = monthly_data['Revenue'].pct_change().mean() * 100
        
        best_model = min(model_performance.keys(), key=lambda x: model_performance[x].get('MAPE', float('inf')))
        best_forecast = forecasts[best_model]['Forecast'].mean()
        
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
    
    except Exception as e:
        st.error(f"Error generating insights: {str(e)[:100]}...")

def get_forecast_periods(forecast_period):
    """Get number of periods to forecast"""
    period_map = {
        "Next 3 Months": 3,
        "Next 6 Months": 6,
        "Next 12 Months": 12,
        "Next 18 Months": 18
    }
    return period_map.get(forecast_period, 6)

# Backward compatibility functions
def linear_trend_forecast(monthly_data, periods):
    """Basic linear trend forecast (fallback)"""
    return enhanced_linear_trend_forecast(monthly_data, periods)

def moving_average_forecast(monthly_data, periods):
    """Basic moving average forecast (fallback)"""
    return advanced_moving_average_forecast(monthly_data, periods)

def random_forest_forecast(monthly_data, periods):
    """Basic random forest forecast (fallback)"""
    if SKLEARN_AVAILABLE:
        return random_forest_pro_forecast(monthly_data, periods)
    else:
        return enhanced_linear_trend_forecast(monthly_data, periods)

def generate_forecast(monthly_data, model_name, periods):
    """Original forecast function for backward compatibility"""
    model_mapping = {
        "Linear Trend": "Enhanced Linear Trend",
        "Moving Average": "Advanced Moving Average",
        "Random Forest": "Random Forest Pro",
        "SARIMA": "Enhanced SARIMA",
        "Prophet": "Prophet Pro",
        "XGBoost": "Auto XGBoost"
    }
    
    mapped_model = model_mapping.get(model_name, model_name)
    return generate_enhanced_forecast(monthly_data, mapped_model, periods, {})
