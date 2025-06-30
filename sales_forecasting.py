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
except Exception as e:
    PROPHET_AVAILABLE = False

# Handle XGBoost and sklearn imports
XGB_AVAILABLE = False
try:
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

def load_data():
    """Load sales data using the app's built-in data loading functions"""
    import sys
    
    # Get the main module (app.py) functions
    if 'app' in sys.modules:
        app_module = sys.modules['app']
        if hasattr(app_module, 'load_data'):
            return app_module.load_data()
    
    # Fallback: try to access data via session state mechanism
    if st.session_state.get('uploaded_file') is not None:
        try:
            uploaded_file = st.session_state['uploaded_file']
            df = pd.read_excel(uploaded_file)
            return df
        except:
            pass
    
    # If all else fails, return empty dataframe
    st.error("Unable to load data. Please ensure the main app is running properly.")
    return pd.DataFrame()

def show_data_source():
    """Show information about current data source"""
    if st.session_state.get('uploaded_file') is not None:
        filename = st.session_state['uploaded_file'].name
        st.success(f"📊 **Data Source**: {filename}")
    else:
        st.info("📝 **Data Source**: Sample data")

def prepare_time_series_data(df, date_col='Invoice Date', value_col='Total Sales', 
                           frequency='M', filter_col=None, filter_value=None):
    """Prepare time series data for forecasting"""
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    
    # Filter data if specified
    if filter_col and filter_value and filter_col in df.columns:
        df = df[df[filter_col] == filter_value].copy()
    
    # Convert date column to datetime and remove timezone
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    
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

def create_simple_forecast(ts_data, periods=6, method='moving_average', window=3):
    """Create simple forecasts (Moving Average, Linear Trend, Exponential Smoothing)"""
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
    """Create visualization comparing historical data with forecasts"""
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

def main():
    """Main function for sales forecasting"""
    
    st.title("📈 Sales Forecasting")
    st.markdown("---")
    
    # Show data source information
    show_data_source()
    
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
    
    # Run forecasting models
    st.subheader("🔮 Forecasting Results")
    
    forecast_results = {}
    
    for model_name in selected_models:
        with st.spinner(f"Running {model_name} forecast..."):
            
            if model_name == 'Moving Average':
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
                
                cols_to_show = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                
                st.dataframe(
                    display_forecast[cols_to_show].style.format({
                        'yhat': '${:,.0f}',
                        'yhat_lower': '${:,.0f}',
                        'yhat_upper': '${:,.0f}'
                    })
                )

if __name__ == "__main__":
    main()
