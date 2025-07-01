import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Safe library imports with detailed error handling
def safe_import_libraries():
    """Safely import all optional libraries with detailed error reporting"""
    
    libraries = {
        'statsmodels': False,
        'prophet': False,
        'xgboost': False,
        'lightgbm': False,
        'sklearn': False,
        'scipy': False
    }
    
    # Try importing statsmodels
    try:
        import statsmodels
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        libraries['statsmodels'] = True
        globals()['ARIMA'] = ARIMA
        globals()['seasonal_decompose'] = seasonal_decompose
        globals()['SARIMAX'] = SARIMAX
        globals()['ExponentialSmoothing'] = ExponentialSmoothing
    except Exception as e:
        st.info(f"📊 Statsmodels not available: Install with `pip install statsmodels`")
    
    # Try importing prophet with extra care
    try:
        import prophet
        from prophet import Prophet
        libraries['prophet'] = True
        globals()['Prophet'] = Prophet
    except ImportError:
        st.info(f"📈 Prophet not available: Install with `pip install prophet`")
    except Exception as e:
        st.warning(f"📈 Prophet installation issue: {str(e)[:100]}... Try: `pip install prophet`")
    
    # Try importing XGBoost
    try:
        import xgboost as xgb
        from xgboost import XGBRegressor
        libraries['xgboost'] = True
        globals()['xgb'] = xgb
        globals()['XGBRegressor'] = XGBRegressor
    except Exception as e:
        st.info(f"🚀 XGBoost not available: Install with `pip install xgboost`")
    
    # Try importing LightGBM
    try:
        import lightgbm as lgb
        libraries['lightgbm'] = True
        globals()['lgb'] = lgb
    except Exception as e:
        st.info(f"⚡ LightGBM not available: Install with `pip install lightgbm`")
    
    # Try importing sklearn
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.neural_network import MLPRegressor
        from sklearn.linear_model import Ridge
        libraries['sklearn'] = True
        globals()['RandomForestRegressor'] = RandomForestRegressor
        globals()['GradientBoostingRegressor'] = GradientBoostingRegressor
        globals()['StandardScaler'] = StandardScaler
        globals()['mean_absolute_error'] = mean_absolute_error
        globals()['mean_squared_error'] = mean_squared_error
        globals()['MLPRegressor'] = MLPRegressor
        globals()['Ridge'] = Ridge
        
        # Try to import MAPE
        try:
            from sklearn.metrics import mean_absolute_percentage_error
            globals()['mean_absolute_percentage_error'] = mean_absolute_percentage_error
        except ImportError:
            def mean_absolute_percentage_error(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
            globals()['mean_absolute_percentage_error'] = mean_absolute_percentage_error
            
    except Exception as e:
        st.error(f"❌ Scikit-learn is required but not available: {str(e)[:100]}...")
        st.info("Install with: `pip install scikit-learn`")
    
    # Try importing scipy
    try:
        from scipy import stats
        libraries['scipy'] = True
        globals()['stats'] = stats
    except Exception as e:
        st.info(f"📊 SciPy not available: Install with `pip install scipy`")
    
    return libraries

# Initialize libraries
LIBRARY_STATUS = safe_import_libraries()

def create_sales_forecast(df):
    """
    Create enhanced advanced sales forecasting analysis (ultra-robust version)
    """
    st.header("🚀 Enhanced Advanced Sales Forecasting")
    st.markdown("*All amounts in Saudi Riyal (SAR)*")
    
    if df.empty:
        st.error("No data available for forecasting")
        return
    
    # Display library status
    display_library_status()
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No forecasting models available.")
        st.info("Basic models require NumPy and Pandas which should be available.")
        return
    
    st.success(f"✅ **{len(available_models)} Models Available**: {', '.join(available_models[:3])}{'...' if len(available_models) > 3 else ''}")
    
    # Simplified interface
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_period = st.selectbox(
            "Forecast Period",
            ["Next 3 Months", "Next 6 Months", "Next 12 Months"],
            index=1
        )
        
        max_models = st.slider(
            "Number of Models",
            min_value=1,
            max_value=min(5, len(available_models)),
            value=min(3, len(available_models))
        )
    
    with col2:
        auto_select = st.checkbox("Auto-select best models", value=True)
        
        if auto_select:
            model_selection = available_models[:max_models]
        else:
            model_selection = st.multiselect(
                "Select Models",
                available_models,
                default=available_models[:max_models]
            )
    
    if not model_selection:
        st.warning("Please select at least one forecasting model")
        return
    
    # Run forecasting
    create_forecasting_analysis(df, forecast_period, model_selection)

def display_library_status():
    """Display current library status"""
    st.subheader("📚 Library Status")
    
    status_data = []
    for lib, available in LIBRARY_STATUS.items():
        status = "✅ Available" if available else "❌ Missing"
        features = {
            'statsmodels': "ARIMA, SARIMA, ETS models",
            'prophet': "Advanced time series forecasting",
            'xgboost': "Gradient boosting models",
            'lightgbm': "Fast gradient boosting",
            'sklearn': "Machine learning models",
            'scipy': "Statistical functions"
        }
        
        status_data.append({
            "Library": lib.title(),
            "Status": status,
            "Features": features.get(lib, "")
        })
    
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True)

def get_available_models():
    """Get list of available models based on installed libraries"""
    models = ["Linear Trend", "Moving Average"]  # Always available
    
    if LIBRARY_STATUS.get('sklearn', False):
        models.extend(["Random Forest", "Gradient Boosting"])
    
    if LIBRARY_STATUS.get('statsmodels', False):
        models.extend(["SARIMA", "ETS"])
    
    if LIBRARY_STATUS.get('prophet', False):
        models.append("Prophet")
    
    if LIBRARY_STATUS.get('xgboost', False):
        models.append("XGBoost")
    
    if LIBRARY_STATUS.get('sklearn', False) and len(models) >= 3:
        models.append("Ensemble")
    
    return models

def create_forecasting_analysis(df, forecast_period, model_selection):
    """Main forecasting analysis with robust error handling"""
    st.subheader("🎯 Forecasting Analysis")
    
    try:
        # Prepare data
        monthly_data = prepare_monthly_data(df)
        
        if monthly_data.empty:
            st.error("❌ Could not prepare data for forecasting")
            return
        
        if len(monthly_data) < 6:
            st.error(f"❌ Need at least 6 months of data. You have {len(monthly_data)} months.")
            return
        
        st.success(f"✅ Data prepared: {len(monthly_data)} months of sales data")
        
        # Show data quality
        show_data_quality(monthly_data)
        
        # Generate forecasts
        periods = get_forecast_periods(forecast_period)
        forecasts = {}
        performance = {}
        
        progress_bar = st.progress(0)
        
        for i, model_name in enumerate(model_selection):
            with st.spinner(f"Training {model_name}..."):
                try:
                    forecast_result = generate_forecast_safe(monthly_data, model_name, periods)
                    
                    if forecast_result is not None and len(forecast_result) > 0:
                        forecasts[model_name] = forecast_result
                        
                        # Calculate performance
                        perf = calculate_performance(monthly_data, model_name, periods)
                        performance[model_name] = perf
                        
                        st.success(f"✅ {model_name} completed")
                    else:
                        st.warning(f"⚠️ {model_name} failed")
                        
                except Exception as e:
                    st.warning(f"⚠️ {model_name} error: {str(e)[:50]}...")
            
            progress_bar.progress((i + 1) / len(model_selection))
        
        if not forecasts:
            st.error("❌ No forecasts generated. Try simpler models.")
            return
        
        st.success(f"✅ Generated {len(forecasts)} successful forecasts!")
        
        # Display results
        display_forecast_results(monthly_data, forecasts, performance, forecast_period)
        
        # Create ensemble if multiple models
        if len(forecasts) > 1:
            create_ensemble_analysis(monthly_data, forecasts, performance, forecast_period)
        
        # Business insights
        create_business_insights(monthly_data, forecasts, performance)
        
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")
        st.info("💡 Try with fewer models or simpler options")

def prepare_monthly_data(df):
    """Prepare monthly data with robust error handling"""
    try:
        # Group by month
        monthly_data = df.groupby('YearMonth').agg({
            'Total Line Amount': 'sum',
            'Invoice No.': 'nunique',
            'Cust Name': 'nunique'
        }).reset_index()
        
        monthly_data.columns = ['YearMonth', 'Revenue', 'Invoices', 'Customers']
        monthly_data['YearMonth'] = pd.to_datetime(monthly_data['YearMonth'])
        monthly_data = monthly_data.sort_values('YearMonth').reset_index(drop=True)
        
        # Add basic features
        monthly_data['Month'] = monthly_data['YearMonth'].dt.month
        monthly_data['Quarter'] = monthly_data['YearMonth'].dt.quarter
        
        # Add lag features
        monthly_data['Revenue_Lag1'] = monthly_data['Revenue'].shift(1)
        monthly_data['Revenue_MA3'] = monthly_data['Revenue'].rolling(3).mean()
        
        return monthly_data
        
    except Exception as e:
        st.error(f"Data preparation error: {e}")
        return pd.DataFrame()

def show_data_quality(monthly_data):
    """Show data quality metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(monthly_data))
    
    with col2:
        avg_revenue = monthly_data['Revenue'].mean()
        st.metric("Avg Monthly Revenue", f"{avg_revenue:,.0f} SAR")
    
    with col3:
        growth = monthly_data['Revenue'].pct_change().mean() * 100
        st.metric("Avg Growth", f"{growth:.1f}%")
    
    with col4:
        volatility = monthly_data['Revenue'].std() / monthly_data['Revenue'].mean()
        st.metric("Volatility", f"{volatility:.2f}")

def generate_forecast_safe(monthly_data, model_name, periods):
    """Generate forecast with comprehensive error handling"""
    try:
        if model_name == "Linear Trend":
            return linear_trend_forecast(monthly_data, periods)
        elif model_name == "Moving Average":
            return moving_average_forecast(monthly_data, periods)
        elif model_name == "Random Forest" and LIBRARY_STATUS.get('sklearn', False):
            return random_forest_forecast(monthly_data, periods)
        elif model_name == "Gradient Boosting" and LIBRARY_STATUS.get('sklearn', False):
            return gradient_boosting_forecast(monthly_data, periods)
        elif model_name == "SARIMA" and LIBRARY_STATUS.get('statsmodels', False):
            return sarima_forecast(monthly_data, periods)
        elif model_name == "ETS" and LIBRARY_STATUS.get('statsmodels', False):
            return ets_forecast(monthly_data, periods)
        elif model_name == "Prophet" and LIBRARY_STATUS.get('prophet', False):
            return prophet_forecast(monthly_data, periods)
        elif model_name == "XGBoost" and LIBRARY_STATUS.get('xgboost', False):
            return xgboost_forecast(monthly_data, periods)
        elif model_name == "Ensemble":
            return ensemble_forecast(monthly_data, periods)
        else:
            return None
            
    except Exception as e:
        return None

def linear_trend_forecast(monthly_data, periods):
    """Simple linear trend forecast"""
    try:
        y = monthly_data['Revenue'].values
        x = np.arange(len(y))
        
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        future_x = np.arange(len(y), len(y) + periods)
        predictions = slope * future_x + intercept
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=periods, 
            freq='M'
        )
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Linear Trend'
        })
    except Exception:
        return None

def moving_average_forecast(monthly_data, periods):
    """Moving average forecast with seasonality"""
    try:
        window = min(6, len(monthly_data) // 2)
        recent_avg = monthly_data['Revenue'].tail(window).mean()
        
        # Simple seasonal adjustment if enough data
        if len(monthly_data) >= 12:
            monthly_data['Month'] = monthly_data['YearMonth'].dt.month
            monthly_avg = monthly_data.groupby('Month')['Revenue'].mean()
            overall_avg = monthly_data['Revenue'].mean()
            seasonal_factors = monthly_avg / overall_avg
            
            predictions = []
            last_date = monthly_data['YearMonth'].iloc[-1]
            
            for i in range(periods):
                future_date = last_date + pd.DateOffset(months=i+1)
                month = future_date.month
                seasonal_factor = seasonal_factors.get(month, 1.0)
                pred = recent_avg * seasonal_factor
                predictions.append(pred)
        else:
            predictions = [recent_avg] * periods
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=periods, 
            freq='M'
        )
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Moving Average'
        })
    except Exception:
        return None

def random_forest_forecast(monthly_data, periods):
    """Random Forest forecast"""
    try:
        features = ['Month', 'Quarter']
        
        if 'Revenue_Lag1' in monthly_data.columns:
            features.append('Revenue_Lag1')
        
        train_data = monthly_data.dropna(subset=['Revenue'] + features)
        
        if len(train_data) < 6:
            return linear_trend_forecast(monthly_data, periods)
        
        X = train_data[features]
        y = train_data['Revenue']
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            future_features = {
                'Month': future_date.month,
                'Quarter': future_date.quarter
            }
            
            if 'Revenue_Lag1' in features:
                future_features['Revenue_Lag1'] = last_row['Revenue']
            
            X_future = [[future_features[f] for f in features]]
            pred = model.predict(X_future)[0]
            predictions.append(pred)
            
            last_row['YearMonth'] = future_date
            last_row['Revenue'] = pred
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=periods, 
            freq='M'
        )
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Random Forest'
        })
        
    except Exception:
        return linear_trend_forecast(monthly_data, periods)

def gradient_boosting_forecast(monthly_data, periods):
    """Gradient Boosting forecast"""
    try:
        features = ['Month', 'Quarter']
        
        if 'Revenue_Lag1' in monthly_data.columns:
            features.append('Revenue_Lag1')
        
        train_data = monthly_data.dropna(subset=['Revenue'] + features)
        
        if len(train_data) < 6:
            return linear_trend_forecast(monthly_data, periods)
        
        X = train_data[features]
        y = train_data['Revenue']
        
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            future_features = {
                'Month': future_date.month,
                'Quarter': future_date.quarter
            }
            
            if 'Revenue_Lag1' in features:
                future_features['Revenue_Lag1'] = last_row['Revenue']
            
            X_future = [[future_features[f] for f in features]]
            pred = model.predict(X_future)[0]
            predictions.append(pred)
            
            last_row['YearMonth'] = future_date
            last_row['Revenue'] = pred
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=periods, 
            freq='M'
        )
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'Gradient Boosting'
        })
        
    except Exception:
        return linear_trend_forecast(monthly_data, periods)

def sarima_forecast(monthly_data, periods):
    """SARIMA forecast"""
    try:
        ts = monthly_data['Revenue'].dropna()
        
        if len(ts) < 12:
            return linear_trend_forecast(monthly_data, periods)
        
        try:
            model = ARIMA(ts, order=(1, 1, 1))
            fitted_model = model.fit()
        except:
            try:
                model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                fitted_model = model.fit(disp=False)
            except:
                return linear_trend_forecast(monthly_data, periods)
        
        forecast = fitted_model.forecast(steps=periods)
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=periods, 
            freq='M'
        )
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast,
            'Model': 'SARIMA'
        })
        
    except Exception:
        return linear_trend_forecast(monthly_data, periods)

def ets_forecast(monthly_data, periods):
    """ETS forecast"""
    try:
        ts = monthly_data['Revenue'].values
        
        if len(ts) < 12:
            return linear_trend_forecast(monthly_data, periods)
        
        try:
            model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
        except:
            try:
                model = ExponentialSmoothing(ts, trend='add')
                fitted_model = model.fit()
            except:
                return linear_trend_forecast(monthly_data, periods)
        
        forecast = fitted_model.forecast(steps=periods)
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=periods, 
            freq='M'
        )
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast,
            'Model': 'ETS'
        })
        
    except Exception:
        return linear_trend_forecast(monthly_data, periods)

def prophet_forecast(monthly_data, periods):
    """Prophet forecast with safe handling"""
    try:
        prophet_data = monthly_data[['YearMonth', 'Revenue']].copy()
        prophet_data.columns = ['ds', 'y']
        
        if len(prophet_data) < 12:
            return linear_trend_forecast(monthly_data, periods)
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        
        model.fit(prophet_data)
        
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        future_forecast = forecast.tail(periods)[['ds', 'yhat']].copy()
        future_forecast.columns = ['Date', 'Forecast']
        future_forecast['Model'] = 'Prophet'
        
        return future_forecast
        
    except Exception:
        return linear_trend_forecast(monthly_data, periods)

def xgboost_forecast(monthly_data, periods):
    """XGBoost forecast"""
    try:
        features = ['Month', 'Quarter']
        
        if 'Revenue_Lag1' in monthly_data.columns:
            features.append('Revenue_Lag1')
        
        train_data = monthly_data.dropna(subset=['Revenue'] + features)
        
        if len(train_data) < 12:
            return linear_trend_forecast(monthly_data, periods)
        
        X = train_data[features]
        y = train_data['Revenue']
        
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        predictions = []
        last_row = train_data.iloc[-1].copy()
        
        for i in range(periods):
            future_date = last_row['YearMonth'] + pd.DateOffset(months=1)
            future_features = {
                'Month': future_date.month,
                'Quarter': future_date.quarter
            }
            
            if 'Revenue_Lag1' in features:
                future_features['Revenue_Lag1'] = last_row['Revenue']
            
            X_future = [[future_features[f] for f in features]]
            pred = model.predict(X_future)[0]
            predictions.append(pred)
            
            last_row['YearMonth'] = future_date
            last_row['Revenue'] = pred
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=periods, 
            freq='M'
        )
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions,
            'Model': 'XGBoost'
        })
        
    except Exception:
        return linear_trend_forecast(monthly_data, periods)

def ensemble_forecast(monthly_data, periods):
    """Simple ensemble forecast"""
    try:
        forecasts = []
        
        # Generate base forecasts
        linear_fc = linear_trend_forecast(monthly_data, periods)
        if linear_fc is not None:
            forecasts.append(linear_fc['Forecast'].values)
        
        ma_fc = moving_average_forecast(monthly_data, periods)
        if ma_fc is not None:
            forecasts.append(ma_fc['Forecast'].values)
        
        if LIBRARY_STATUS.get('sklearn', False):
            rf_fc = random_forest_forecast(monthly_data, periods)
            if rf_fc is not None:
                forecasts.append(rf_fc['Forecast'].values)
        
        if len(forecasts) < 2:
            return linear_trend_forecast(monthly_data, periods)
        
        # Simple average
        ensemble_pred = np.mean(forecasts, axis=0)
        
        last_date = monthly_data['YearMonth'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=periods, 
            freq='M'
        )
        
        return pd.DataFrame({
            'Date': future_dates,
            'Forecast': ensemble_pred,
            'Model': 'Ensemble'
        })
        
    except Exception:
        return linear_trend_forecast(monthly_data, periods)

def calculate_performance(monthly_data, model_name, periods):
    """Calculate simple model performance"""
    try:
        if len(monthly_data) < 12:
            return {'MAPE': 100, 'MAE': 0}
        
        # Use last 3 months for validation
        train_size = len(monthly_data) - 3
        train_data = monthly_data[:train_size]
        test_data = monthly_data[train_size:]
        
        forecast_result = generate_forecast_safe(train_data, model_name, len(test_data))
        
        if forecast_result is None:
            return {'MAPE': 100, 'MAE': 0}
        
        actual = test_data['Revenue'].values
        predicted = forecast_result['Forecast'].values
        
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted)
        
        return {'MAPE': mape, 'MAE': mae}
        
    except Exception:
        return {'MAPE': 100, 'MAE': 0}

def display_forecast_results(monthly_data, forecasts, performance, forecast_period):
    """Display forecast results"""
    st.subheader("📊 Forecast Results")
    
    # Main chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=monthly_data['YearMonth'],
        y=monthly_data['Revenue'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='blue', width=3)
    ))
    
    # Forecasts
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, forecast_df) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast'],
            mode='lines+markers',
            name=f'{model_name}',
            line=dict(color=color, dash='dash', width=2)
        ))
    
    fig.update_layout(
        title='Sales Forecast Comparison',
        xaxis_title='Date',
        yaxis_title='Revenue (SAR)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.subheader("📋 Forecast Summary")
    
    summary_data = []
    for model_name, forecast_df in forecasts.items():
        total_forecast = forecast_df['Forecast'].sum()
        avg_monthly = forecast_df['Forecast'].mean()
        
        perf = performance.get(model_name, {})
        mape = perf.get('MAPE', 0)
        mae = perf.get('MAE', 0)
        
        summary_data.append({
            'Model': model_name,
            f'Total ({forecast_period})': f"{total_forecast:,.0f} SAR",
            'Avg Monthly': f"{avg_monthly:,.0f} SAR",
            'MAPE': f"{mape:.1f}%",
            'MAE': f"{mae:,.0f} SAR"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

def create_ensemble_analysis(monthly_data, forecasts, performance, forecast_period):
    """Create ensemble analysis"""
    st.subheader("🎯 Ensemble Analysis")
    
    try:
        # Calculate weights based on performance
        weights = {}
        total_weight = 0
        
        for model_name in forecasts.keys():
            mape = performance.get(model_name, {}).get('MAPE', 100)
            weight = 1 / (1 + mape) if mape > 0 else 0.1
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for model_name in weights.keys():
                weights[model_name] = weights[model_name] / total_weight
        
        # Calculate ensemble
        ensemble_forecast = None
        first_model = True
        
        for model_name, forecast_df in forecasts.items():
            weight = weights.get(model_name, 0)
            
            if first_model:
                ensemble_forecast = forecast_df.copy()
                ensemble_forecast['Forecast'] = forecast_df['Forecast'] * weight
                first_model = False
            else:
                ensemble_forecast['Forecast'] += forecast_df['Forecast'] * weight
        
        # Display ensemble
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data['YearMonth'],
            y=monthly_data['Revenue'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=ensemble_forecast['Date'],
            y=ensemble_forecast['Forecast'],
            mode='lines+markers',
            name='Smart Ensemble',
            line=dict(color='black', width=3)
        ))
        
        fig.update_layout(
            title='Smart Ensemble Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue (SAR)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show metrics
        ensemble_total = ensemble_forecast['Forecast'].sum()
        ensemble_avg = ensemble_forecast['Forecast'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Ensemble Total ({forecast_period})", f"{ensemble_total:,.0f} SAR")
        with col2:
            st.metric("Ensemble Avg Monthly", f"{ensemble_avg:,.0f} SAR")
        
        # Show weights
        st.write("**Model Weights:**")
        weights_df = pd.DataFrame([
            {'Model': model, 'Weight': f"{weight:.1%}"}
            for model, weight in weights.items()
        ])
        st.dataframe(weights_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Ensemble analysis error: {str(e)[:100]}...")

def create_business_insights(monthly_data, forecasts, performance):
    """Generate business insights"""
    st.subheader("💡 Business Insights")
    
    try:
        # Find best model
        best_model = min(performance.keys(), key=lambda x: performance[x].get('MAPE', float('inf')))
        best_forecast = forecasts[best_model]['Forecast'].mean()
        
        # Calculate metrics
        historical_avg = monthly_data['Revenue'].mean()
        growth_projection = (best_forecast - historical_avg) / historical_avg * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Key Metrics")
            st.write(f"• **Best Model**: {best_model}")
            st.write(f"• **Projected Growth**: {growth_projection:.1f}%")
            st.write(f"• **Historical Average**: {historical_avg:,.0f} SAR")
            
            if growth_projection > 5:
                st.success("🎯 Strong growth projected")
            elif growth_projection < -5:
                st.warning("⚠️ Decline projected")
            else:
                st.info("📊 Stable growth projected")
        
        with col2:
            st.markdown("### 🎯 Recommendations")
            
            best_mape = performance[best_model].get('MAPE', 0)
            
            if best_mape < 15:
                st.success("✅ High confidence in forecasts")
            elif best_mape < 30:
                st.warning("⚠️ Moderate confidence")
            else:
                st.error("❌ Low confidence - use with caution")
            
            st.write("**Next Steps:**")
            st.write("• Monitor monthly performance")
            st.write("• Update forecasts with new data")
            st.write("• Review model performance quarterly")
    
    except Exception as e:
        st.error(f"Insights error: {str(e)[:100]}...")

def get_forecast_periods(forecast_period):
    """Get number of periods to forecast"""
    period_map = {
        "Next 3 Months": 3,
        "Next 6 Months": 6,
        "Next 12 Months": 12
    }
    return period_map.get(forecast_period, 6)

# Backward compatibility functions for existing apps
def create_advanced_overall_forecast(df, forecast_period, model_selection, confidence_level, validation_method, advanced_options):
    """Backward compatibility wrapper"""
    return create_forecasting_analysis(df, forecast_period, model_selection)

# Note: get_available_models() function is already defined above, no need for alias
