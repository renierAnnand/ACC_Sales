import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Handle Prophet import with version compatibility
PROPHET_AVAILABLE = False
try:
    # Try to fix numpy compatibility issues
    import numpy as np
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'int_'):
        np.int_ = np.int64
    if not hasattr(np, 'bool_'):
        np.bool_ = bool
    
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not available. Install with: pip install prophet")
except Exception as e:
    PROPHET_AVAILABLE = False
    st.warning(f"Prophet compatibility issue: {str(e)}")

# Handle XGBoost and sklearn imports
XGB_AVAILABLE = False
try:
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Always available imports
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

def load_data():
    """Load sales data from data_loader"""
    try:
        import data_loader
        return data_loader.load_and_merge_data()
    except ImportError:
        st.error("Data loader not found. Please ensure data_loader.py is properly configured.")
        return pd.DataFrame()

def prepare_time_series_data(df, date_col='Invoice Date', value_col='Total Sales', 
                           frequency='M', filter_col=None, filter_value=None):
    """
    Prepare time series data for forecasting
    """
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    
    # Filter data if specified
    if filter_col and filter_value and filter_col in df.columns:
        df = df[df[filter_col] == filter_value].copy()
    
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Aggregate data by frequency
    if frequency == 'D':  # Daily
        ts_data = df.groupby(df[date_col].dt.date)[value_col].sum().reset_index()
        ts_data.columns = ['ds', 'y']
        ts_data['ds'] = pd.to_datetime(ts_data['ds'])
    elif frequency == 'W':  # Weekly
        ts_data = df.groupby(df[date_col].dt.to_period('W'))[value_col].sum().reset_index()
        ts_data['ds'] = ts_data[date_col].dt.start_time
        ts_data = ts_data[['ds', 'y']]
    elif frequency == 'M':  # Monthly
        ts_data = df.groupby(df[date_col].dt.to_period('M'))[value_col].sum().reset_index()
        ts_data['ds'] = ts_data[date_col].dt.start_time
        ts_data = ts_data[['ds', 'y']]
    elif frequency == 'Q':  # Quarterly
        ts_data = df.groupby(df[date_col].dt.to_period('Q'))[value_col].sum().reset_index()
        ts_data['ds'] = ts_data[date_col].dt.start_time
        ts_data = ts_data[['ds', 'y']]
    
    # Sort by date
    ts_data = ts_data.sort_values('ds').reset_index(drop=True)
    
    # Remove any rows with zero or negative values for better forecasting
    ts_data = ts_data[ts_data['y'] > 0]
    
    return ts_data

def create_prophet_forecast(ts_data, periods=6, freq='M', include_holidays=False):
    """
    Create forecast using Facebook Prophet with error handling
    """
    if not PROPHET_AVAILABLE:
        return None, "Prophet not available. Please install: pip install prophet"
    
    if ts_data.empty or len(ts_data) < 10:
        return None, "Insufficient data for Prophet forecasting (minimum 10 points required)"
    
    try:
        # Initialize Prophet model with simpler configuration
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=freq == 'D',
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            interval_width=0.8,
            uncertainty_samples=100  # Reduce for faster processing
        )
        
        # Add holidays only if requested and available
        if include_holidays:
            try:
                model.add_country_holidays(country_name='SA')  # Saudi Arabia holidays
            except Exception:
                st.warning("Could not add Saudi Arabia holidays, continuing without them")
        
        # Fit the model
        with st.spinner("Training Prophet model..."):
            model.fit(ts_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='MS' if freq == 'M' else freq)
        
        # Make forecast
        forecast = model.predict(future)
        
        # Calculate accuracy metrics on historical data
        historical_forecast = forecast[:-periods]
        
        # Ensure we have matching lengths for accuracy calculation
        min_length = min(len(ts_data), len(historical_forecast))
        actual_values = ts_data['y'][:min_length]
        predicted_values = historical_forecast['yhat'][:min_length]
        
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        return {
            'model': model,
            'forecast': forecast,
            'metrics': metrics,
            'future_periods': periods
        }, None
        
    except Exception as e:
        return None, f"Prophet error: {str(e)}"

def create_xgboost_forecast(ts_data, periods=6, freq='M'):
    """
    Create forecast using XGBoost with enhanced error handling
    """
    if not XGB_AVAILABLE:
        return None, "XGBoost not available. Install with: pip install xgboost scikit-learn"
    
    if ts_data.empty or len(ts_data) < 20:
        return None, "Insufficient data for XGBoost forecasting (minimum 20 points required)"
    
    try:
        # Prepare features
        df = ts_data.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Create time-based features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['quarter'] = df['ds'].dt.quarter
        df['day_of_year'] = df['ds'].dt.dayofyear
        
        # Create lag features (more conservative)
        for lag in [1, 2, 3]:
            if len(df) > lag:
                df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Create rolling statistics (more conservative)
        for window in [3, 6]:
            if len(df) > window:
                df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std()
        
        # Create trend feature
        df['trend'] = range(len(df))
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < 10:
            return None, "Insufficient clean data for XGBoost after feature engineering"
        
        # Prepare features and target
        feature_cols = [col for col in df_clean.columns if col not in ['ds', 'y']]
        X = df_clean[feature_cols]
        y = df_clean['y']
        
        # Split data for training and validation
        split_idx = max(1, int(len(X) * 0.8))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train XGBoost model with conservative parameters
        model = xgb.XGBRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=4,      # Reduced to prevent overfitting
            learning_rate=0.1,
            random_state=42,
            verbosity=0       # Suppress output
        )
        
        model.fit(X_train, y_train)
        
        # Validate model if we have validation data
        if len(X_val) > 0:
            val_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100 if len(y_val) > 0 else 0
        else:
            # Use training data for metrics if no validation data
            train_pred = model.predict(X_train)
            mae = mean_absolute_error(y_train, train_pred)
            rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            mape = np.mean(np.abs((y_train - train_pred) / y_train)) * 100
        
        # Create future predictions
        future_dates = pd.date_range(
            start=df['ds'].max() + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )
        
        # Prepare future features
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['year'] = future_df['ds'].dt.year
        future_df['month'] = future_df['ds'].dt.month
        future_df['quarter'] = future_df['ds'].dt.quarter
        future_df['day_of_year'] = future_df['ds'].dt.dayofyear
        
        # Extend trend
        last_trend = df['trend'].max()
        future_df['trend'] = range(last_trend + 1, last_trend + 1 + periods)
        
        # For lag features, use last known values (simplified approach)
        for lag in [1, 2, 3]:
            col_name = f'lag_{lag}'
            if col_name in feature_cols:
                if lag <= len(df):
                    future_df[col_name] = df['y'].iloc[-min(lag, len(df)):].mean()
                else:
                    future_df[col_name] = df['y'].mean()
        
        # For rolling features, use recent values
        for window in [3, 6]:
            mean_col = f'rolling_mean_{window}'
            std_col = f'rolling_std_{window}'
            if mean_col in feature_cols:
                future_df[mean_col] = df['y'].tail(min(window, len(df))).mean()
            if std_col in feature_cols:
                future_df[std_col] = df['y'].tail(min(window, len(df))).std()
                # Fill NaN with 0 for std
                future_df[std_col] = future_df[std_col].fillna(0)
        
        # Make future predictions
        future_X = future_df[feature_cols]
        future_pred = model.predict(future_X)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': future_pred,
            'yhat_lower': future_pred * 0.9,  # Simple confidence interval
            'yhat_upper': future_pred * 1.1
        })
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        return {
            'model': model,
            'forecast': forecast_df,
            'historical': df_clean,
            'metrics': metrics,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }, None
        
    except Exception as e:
        return None, f"XGBoost error: {str(e)}"

def create_simple_forecast(ts_data, periods=6, method='moving_average', window=3):
    """
    Create simple forecasts (Moving Average, Linear Trend, Exponential Smoothing)
    """
    if ts_data.empty or len(ts_data) < window:
        return None, f"Insufficient data for {method} forecasting (need at least {window} points)"
    
    try:
        df = ts_data.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        if method == 'moving_average':
            # Simple moving average forecast
            last_values = df['y'].tail(window).mean()
            future_values = [last_values] * periods
            
        elif method == 'linear_trend':
            # Linear trend forecast
            x = np.arange(len(df))
            coeffs = np.polyfit(x, df['y'], 1)
            
            future_x = np.arange(len(df), len(df) + periods)
            future_values = np.polyval(coeffs, future_x)
            
        elif method == 'exponential_smoothing':
            # Simple exponential smoothing
            alpha = 0.3
            smoothed = [df['y'].iloc[0]]
            
            for i in range(1, len(df)):
                smoothed.append(alpha * df['y'].iloc[i] + (1 - alpha) * smoothed[-1])
            
            # Forecast using last smoothed value
            future_values = [smoothed[-1]] * periods
        
        # Create future dates
        last_date = df['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': future_values,
            'yhat_lower': np.array(future_values) * 0.85,
            'yhat_upper': np.array(future_values) * 1.15
        })
        
        # Calculate simple accuracy (using last few points)
        if len(df) >= 6:
            test_size = min(3, len(df) // 4)
            test_actual = df['y'].tail(test_size).values
            
            if method == 'moving_average':
                test_pred = []
                for i in range(len(df)-test_size, len(df)):
                    pred = df['y'].iloc[max(0, i-window):i].mean()
                    test_pred.append(pred)
            elif method == 'linear_trend':
                test_x = np.arange(len(df)-test_size, len(df))
                test_pred = np.polyval(coeffs, test_x)
            else:
                test_pred = [smoothed[-1]] * test_size
            
            # Calculate metrics
            test_pred = np.array(test_pred)
            mae = np.mean(np.abs(test_actual - test_pred))
            rmse = np.sqrt(np.mean((test_actual - test_pred)**2))
            mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100
        else:
            mae = rmse = mape = 0
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        return {
            'forecast': forecast_df,
            'historical': df,
            'metrics': metrics,
            'method': method
        }, None
        
    except Exception as e:
        return None, f"{method} error: {str(e)}"

def create_forecast_visualization(ts_data, forecast_results, model_name):
    """
    Create visualization comparing historical data with forecasts
    """
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=ts_data['ds'],
        y=ts_data['y'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    if forecast_results:
        forecast_data = forecast_results.get('forecast')
        
        if forecast_data is not None:
            # Plot forecast
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines+markers',
                name=f'{model_name} Forecast',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4)
            ))
            
            # Add confidence intervals if available
            if 'yhat_lower' in forecast_data.columns and 'yhat_upper' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    name='Upper Bound'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    name='Confidence Interval',
                    showlegend=True
                ))
    
    fig.update_layout(
        title=f'📈 Sales Forecast - {model_name}',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_model_comparison_chart(results_dict):
    """
    Create comparison chart of different models' performance
    """
    if not results_dict:
        return None
    
    models = []
    mae_scores = []
    rmse_scores = []
    mape_scores = []
    
    for model_name, result in results_dict.items():
        if result and 'metrics' in result:
            models.append(model_name)
            mae_scores.append(result['metrics']['MAE'])
            rmse_scores.append(result['metrics']['RMSE'])
            mape_scores.append(result['metrics']['MAPE'])
    
    if not models:
        return None
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Mean Absolute Error', 'Root Mean Square Error', 'Mean Absolute Percentage Error'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=models, y=mae_scores, name='MAE', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=models, y=mape_scores, name='MAPE (%)', marker_color='lightcoral'),
        row=1, col=3
    )
    
    fig.update_layout(
        title_text="🏆 Model Performance Comparison",
        showlegend=False,
        height=400
    )
    
    return fig

def create_seasonal_decomposition(ts_data, freq=12):
    """
    Create seasonal decomposition visualization
    """
    if len(ts_data) < 2 * freq:
        return None
    
    try:
        # Prepare data
        df = ts_data.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.set_index('ds')
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(df['y'], model='additive', period=freq, extrapolate_trend='freq')
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original Data', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        # Original data
        fig.add_trace(
            go.Scatter(x=df.index, y=df['y'], name='Original', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=df.index, y=decomposition.trend, name='Trend', line=dict(color='red')),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=df.index, y=decomposition.seasonal, name='Seasonal', line=dict(color='green')),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(x=df.index, y=decomposition.resid, name='Residual', line=dict(color='orange')),
            row=4, col=1
        )
        
        fig.update_layout(
            title_text="📊 Time Series Decomposition",
            height=800,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create seasonal decomposition: {str(e)}")
        return None

def export_forecast_to_csv(forecast_data, filename="sales_forecast.csv"):
    """
    Create downloadable CSV of forecast results
    """
    if forecast_data is None or forecast_data.empty:
        return None
    
    csv = forecast_data.to_csv(index=False)
    return csv

def main():
    """Main function for sales forecasting"""
    
    st.title("📈 Sales Forecasting")
    st.markdown("---")
    
    # Display library availability status
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "✅ Available" if PROPHET_AVAILABLE else "❌ Not Available"
        st.info(f"**Prophet**: {status}")
    with col2:
        status = "✅ Available" if XGB_AVAILABLE else "❌ Not Available"
        st.info(f"**XGBoost**: {status}")
    with col3:
        st.info("**Simple Models**: ✅ Available")
    
    # Load data
    with st.spinner("Loading sales data..."):
        df = load_data()
    
    if df.empty:
        st.error("No data available. Please check your data source.")
        return
    
    # Sidebar filters and options
    st.sidebar.header("🔍 Filters & Options")
    
    # Business Unit filter
    filter_options = ['All Data']
    filter_col = None
    filter_value = None
    
    if 'Business Unit' in df.columns:
        filter_options.extend(['By Business Unit'])
        
    if 'Item Description' in df.columns:
        filter_options.extend(['By Product'])
    
    filter_type = st.sidebar.selectbox("Forecast Scope", filter_options)
    
    if filter_type == 'By Business Unit' and 'Business Unit' in df.columns:
        business_units = sorted(df['Business Unit'].unique().tolist())
        selected_bu = st.sidebar.selectbox("Select Business Unit", business_units)
        filter_col = 'Business Unit'
        filter_value = selected_bu
    
    elif filter_type == 'By Product' and 'Item Description' in df.columns:
        products = sorted(df['Item Description'].unique().tolist())
        selected_product = st.sidebar.selectbox("Select Product", products)
        filter_col = 'Item Description'
        filter_value = selected_product
    
    # Forecasting options
    st.sidebar.header("⚙️ Forecasting Options")
    
    forecast_months = st.sidebar.slider("Forecast Horizon (Months)", min_value=1, max_value=12, value=6)
    
    frequency = st.sidebar.selectbox(
        "Data Frequency",
        ['M', 'Q', 'W'],
        format_func=lambda x: {'M': 'Monthly', 'Q': 'Quarterly', 'W': 'Weekly'}[x]
    )
    
    # Model selection based on availability
    available_models = ['Moving Average', 'Linear Trend', 'Exponential Smoothing']
    
    if PROPHET_AVAILABLE:
        available_models.append('Prophet')
    
    if XGB_AVAILABLE:
        available_models.append('XGBoost')
    
    selected_models = st.sidebar.multiselect(
        "Select Models",
        available_models,
        default=['Moving Average', 'Linear Trend']
    )
    
    if not selected_models:
        st.sidebar.error("Please select at least one model")
        return
    
    # Prepare time series data
    with st.spinner("Preparing time series data..."):
        ts_data = prepare_time_series_data(
            df, 
            date_col='Invoice Date', 
            value_col='Total Sales',
            frequency=frequency,
            filter_col=filter_col,
            filter_value=filter_value
        )
    
    if ts_data.empty:
        st.error("No time series data available with current filters.")
        return
    
    # Display data summary
    st.subheader("📊 Time Series Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(ts_data))
    
    with col2:
        date_range = f"{ts_data['ds'].min().strftime('%Y-%m')} to {ts_data['ds'].max().strftime('%Y-%m')}"
        st.metric("Date Range", date_range)
    
    with col3:
        st.metric("Total Sales", f"${ts_data['y'].sum():,.0f}")
    
    with col4:
        st.metric("Average Sales", f"${ts_data['y'].mean():,.0f}")
    
    # Show historical data chart
    fig_historical = px.line(
        ts_data, 
        x='ds', 
        y='y',
        title='📈 Historical Sales Data',
        labels={'ds': 'Date', 'y': 'Sales ($)'}
    )
    st.plotly_chart(fig_historical, use_container_width=True)
    
    # Seasonal decomposition
    if len(ts_data) >= 24:  # Need at least 2 years of monthly data
        with st.expander("📊 Seasonal Analysis"):
            fig_decomp = create_seasonal_decomposition(ts_data)
            if fig_decomp:
                st.plotly_chart(fig_decomp, use_container_width=True)
    
    # Run forecasting models
    st.subheader("🔮 Forecasting Results")
    
    forecast_results = {}
    
    for model_name in selected_models:
        with st.spinner(f"Running {model_name} forecast..."):
            
            if model_name == 'Prophet':
                result, error = create_prophet_forecast(ts_data, periods=forecast_months, freq='MS')
                
            elif model_name == 'XGBoost':
                result, error = create_xgboost_forecast(ts_data, periods=forecast_months, freq='M')
                
            elif model_name == 'Moving Average':
                result, error = create_simple_forecast(ts_data, periods=forecast_months, method='moving_average')
                
            elif model_name == 'Linear Trend':
                result, error = create_simple_forecast(ts_data, periods=forecast_months, method='linear_trend')
                
            elif model_name == 'Exponential Smoothing':
                result, error = create_simple_forecast(ts_data, periods=forecast_months, method='exponential_smoothing')
            
            if error:
                st.error(f"{model_name}: {error}")
            else:
                forecast_results[model_name] = result
    
    if not forecast_results:
        st.error("No forecasting models ran successfully.")
        return
    
    # Display individual forecast results
    tabs = st.tabs(list(forecast_results.keys()))
    
    for i, (model_name, result) in enumerate(forecast_results.items()):
        with tabs[i]:
            
            # Show forecast chart
            fig_forecast = create_forecast_visualization(ts_data, result, model_name)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("MAE", f"${result['metrics']['MAE']:,.0f}")
            
            with col2:
                st.metric("RMSE", f"${result['metrics']['RMSE']:,.0f}")
            
            with col3:
                st.metric("MAPE", f"{result['metrics']['MAPE']:.1f}%")
            
            # Show forecast table
            forecast_data = result.get('forecast')
            if forecast_data is not None:
                st.write("**Forecast Details:**")
                display_forecast = forecast_data.copy()
                display_forecast['ds'] = display_forecast['ds'].dt.strftime('%Y-%m')
                
                cols_to_show = ['ds', 'yhat']
                if 'yhat_lower' in display_forecast.columns:
                    cols_to_show.extend(['yhat_lower', 'yhat_upper'])
                
                st.dataframe(
                    display_forecast[cols_to_show].style.format({
                        'yhat': '${:,.0f}',
                        'yhat_lower': '${:,.0f}',
                        'yhat_upper': '${:,.0f}'
                    })
                )
                
                # Download button
                csv_data = export_forecast_to_csv(display_forecast[cols_to_show])
                if csv_data:
                    st.download_button(
                        label=f"📥 Download {model_name} Forecast CSV",
                        data=csv_data,
                        file_name=f"{model_name.lower().replace(' ', '_')}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            # Show feature importance for XGBoost
            if model_name == 'XGBoost' and 'feature_importance' in result:
                with st.expander("🔍 Feature Importance"):
                    importance_df = pd.DataFrame(
                        list(result['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        importance_df.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Most Important Features'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model comparison
    if len(forecast_results) > 1:
        st.subheader("🏆 Model Comparison")
        
        fig_comparison = create_model_comparison_chart(forecast_results)
        if fig_comparison:
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Best model recommendation
        valid_models = {k: v for k, v in forecast_results.items() if v and 'metrics' in v and v['metrics']['MAPE'] > 0}
        if valid_models:
            best_model = min(valid_models.keys(), 
                            key=lambda x: valid_models[x]['metrics']['MAPE'])
            
            st.success(f"🎯 **Recommended Model**: {best_model} (Lowest MAPE: {valid_models[best_model]['metrics']['MAPE']:.1f}%)")
    
    # Insights and recommendations
    st.subheader("💡 Forecasting Insights")
    
    # Calculate forecast summary
    total_forecast = 0
    for result in forecast_results.values():
        if result and 'forecast' in result:
            total_forecast += result['forecast']['yhat'].sum()
    
    avg_forecast = total_forecast / len(forecast_results) if forecast_results else 0
    
    # Historical comparison
    recent_avg = ts_data['y'].tail(forecast_months).mean() if len(ts_data) >= forecast_months else ts_data['y'].mean()
    growth_rate = ((avg_forecast / forecast_months / recent_avg) - 1) * 100 if recent_avg > 0 else 0
    
    insights = [
        f"📊 **Forecast Summary**: Expected ${avg_forecast:,.0f} total sales over next {forecast_months} months",
        f"📈 **Growth Trend**: {growth_rate:+.1f}% compared to recent average",
        f"🎯 **Monthly Target**: ${avg_forecast/forecast_months:,.0f} average monthly sales",
        f"⚡ **Data Quality**: {len(ts_data)} historical data points available for training"
    ]
    
    # Seasonality insights
    if len(ts_data) >= 12:
        monthly_avg = ts_data.groupby(ts_data['ds'].dt.month)['y'].mean()
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        insights.append(f"📅 **Seasonality**: Peak sales in {month_names[peak_month-1]}, lowest in {month_names[low_month-1]}")
    
    for insight in insights:
        st.markdown(insight)
    
    # Recommendations
    with st.expander("🎯 Strategic Recommendations"):
        st.markdown("""
        **Based on your forecasting results:**
        
        **📈 Revenue Planning:**
        - Use forecasts for budget planning and resource allocation
        - Set realistic sales targets based on predicted trends
        - Plan inventory levels according to demand forecasts
        
        **🎯 Sales Strategy:**
        - Focus marketing efforts during predicted low-demand periods
        - Prepare promotional campaigns to boost sales during slower months
        - Align sales team targets with seasonal patterns
        
        **⚡ Model Improvement:**
        - Collect more granular data (daily/weekly) for better accuracy
        - Include external factors (holidays, economic events) in models
        - Regularly update models with new data for better predictions
        
        **🔍 Monitoring:**
        - Track actual vs. predicted performance monthly
        - Update forecasts quarterly with new data
        - Set up alerts for significant deviations from predictions
        """)

if __name__ == "__main__":
    main()
