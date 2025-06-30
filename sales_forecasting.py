import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_sales_forecast(df):
    """
    Create sales forecasting analysis with multiple models
    """
    st.header("📈 Sales Forecasting")
    
    if df.empty:
        st.error("No data available for forecasting")
        return
    
    # Forecast options
    forecast_type = st.selectbox(
        "Select Forecast Type",
        ["Overall Sales Forecast", "Business Unit Forecast", "Customer Forecast", "Product Category Forecast"]
    )
    
    forecast_period = st.selectbox(
        "Forecast Period",
        ["Next 3 Months", "Next 6 Months", "Next 12 Months"],
        index=1
    )
    
    if forecast_type == "Overall Sales Forecast":
        create_overall_forecast(df, forecast_period)
    elif forecast_type == "Business Unit Forecast":
        create_bu_forecast(df, forecast_period)
    elif forecast_type == "Customer Forecast":
        create_customer_forecast(df, forecast_period)
    elif forecast_type == "Product Category Forecast":
        create_product_forecast(df, forecast_period)

def create_overall_forecast(df, forecast_period):
    """
    Create overall sales forecast
    """
    st.subheader("🎯 Overall Sales Forecast")
    
    try:
        # Prepare time series data
        monthly_data = prepare_monthly_data(df)
        
        if monthly_data.empty:
            st.error("Insufficient data for forecasting")
            return
        
        # Get forecast periods
        periods = get_forecast_periods(forecast_period)
        
        # Multiple forecasting methods
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Trend Analysis")
            
            # Linear trend forecast
            trend_forecast = create_trend_forecast(monthly_data, periods)
            
            # Display trend chart
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=monthly_data['Date'],
                y=monthly_data['Revenue'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='blue')
            ))
            
            # Trend forecast
            if trend_forecast is not None:
                fig.add_trace(go.Scatter(
                    x=trend_forecast['Date'],
                    y=trend_forecast['Forecast'],
                    mode='lines+markers',
                    name='Trend Forecast',
                    line=dict(color='red', dash='dash')
                ))
            
            fig.update_layout(
                title='Sales Trend Forecast',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Moving Average Forecast")
            
            # Moving average forecast
            ma_forecast = create_moving_average_forecast(monthly_data, periods)
            
            # Display moving average chart
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=monthly_data['Date'],
                y=monthly_data['Revenue'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='blue')
            ))
            
            # Moving average
            if len(monthly_data) >= 3:
                monthly_data['MA_3'] = monthly_data['Revenue'].rolling(window=3).mean()
                fig.add_trace(go.Scatter(
                    x=monthly_data['Date'],
                    y=monthly_data['MA_3'],
                    mode='lines',
                    name='3-Month MA',
                    line=dict(color='orange')
                ))
            
            # MA forecast
            if ma_forecast is not None:
                fig.add_trace(go.Scatter(
                    x=ma_forecast['Date'],
                    y=ma_forecast['Forecast'],
                    mode='lines+markers',
                    name='MA Forecast',
                    line=dict(color='green', dash='dash')
                ))
            
            fig.update_layout(
                title='Moving Average Forecast',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        create_forecast_summary(monthly_data, trend_forecast, ma_forecast, forecast_period)
        
        # Seasonal analysis
        create_seasonal_analysis(monthly_data)
        
    except Exception as e:
        st.error(f"Error creating overall forecast: {e}")

def prepare_monthly_data(df):
    """
    Prepare monthly aggregated data for forecasting
    """
    try:
        # Group by month
        monthly_data = df.groupby('YearMonth').agg({
            'Total Line Amount': 'sum',
            'Invoice No.': 'nunique',
            'Cust Name': 'nunique'
        }).reset_index()
        
        monthly_data.columns = ['YearMonth', 'Revenue', 'Orders', 'Customers']
        monthly_data['YearMonth'] = pd.to_datetime(monthly_data['YearMonth'])
        monthly_data = monthly_data.sort_values('YearMonth')
        monthly_data['Date'] = monthly_data['YearMonth']
        
        return monthly_data
        
    except Exception as e:
        st.error(f"Error preparing monthly data: {e}")
        return pd.DataFrame()

def get_forecast_periods(forecast_period):
    """
    Get number of periods to forecast
    """
    if forecast_period == "Next 3 Months":
        return 3
    elif forecast_period == "Next 6 Months":
        return 6
    else:
        return 12

def create_trend_forecast(monthly_data, periods):
    """
    Create linear trend forecast
    """
    try:
        if len(monthly_data) < 3:
            return None
        
        # Prepare data for linear regression
        monthly_data['Period'] = range(len(monthly_data))
        X = monthly_data[['Period']]
        y = monthly_data['Revenue']
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Create future periods
        last_period = monthly_data['Period'].max()
        future_periods = range(last_period + 1, last_period + periods + 1)
        future_X = pd.DataFrame({'Period': future_periods})
        
        # Predict
        predictions = model.predict(future_X)
        
        # Create forecast dataframe
        last_date = monthly_data['Date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='M'
        )
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': predictions
        })
        
        return forecast_df
        
    except Exception as e:
        st.error(f"Error creating trend forecast: {e}")
        return None

def create_moving_average_forecast(monthly_data, periods):
    """
    Create moving average forecast
    """
    try:
        if len(monthly_data) < 3:
            return None
        
        # Calculate 3-month moving average
        last_3_months = monthly_data['Revenue'].tail(3).mean()
        
        # Create forecast dataframe
        last_date = monthly_data['Date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='M'
        )
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': [last_3_months] * periods
        })
        
        return forecast_df
        
    except Exception as e:
        st.error(f"Error creating moving average forecast: {e}")
        return None

def create_forecast_summary(monthly_data, trend_forecast, ma_forecast, forecast_period):
    """
    Create forecast summary table
    """
    st.subheader("📋 Forecast Summary")
    
    try:
        # Calculate historical averages
        avg_monthly_revenue = monthly_data['Revenue'].mean()
        recent_3_months = monthly_data['Revenue'].tail(3).mean()
        
        # Create summary
        summary_data = {
            'Metric': [
                'Historical Monthly Average',
                'Recent 3-Month Average',
                f'Trend Forecast ({forecast_period})',
                f'Moving Average Forecast ({forecast_period})'
            ],
            'Value': [
                f"${avg_monthly_revenue:,.0f}",
                f"${recent_3_months:,.0f}",
                f"${trend_forecast['Forecast'].sum():,.0f}" if trend_forecast is not None else "N/A",
                f"${ma_forecast['Forecast'].sum():,.0f}" if ma_forecast is not None else "N/A"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Growth rate analysis
        if len(monthly_data) >= 2:
            last_month = monthly_data['Revenue'].iloc[-1]
            prev_month = monthly_data['Revenue'].iloc[-2]
            growth_rate = ((last_month - prev_month) / prev_month) * 100
            
            st.metric(
                label="Month-over-Month Growth",
                value=f"{growth_rate:.1f}%",
                delta=f"${last_month - prev_month:,.0f}"
            )
        
    except Exception as e:
        st.error(f"Error creating forecast summary: {e}")

def create_seasonal_analysis(monthly_data):
    """
    Create seasonal analysis
    """
    st.subheader("📅 Seasonal Analysis")
    
    try:
        # Add month name
        monthly_data['Month'] = monthly_data['Date'].dt.month_name()
        
        # Monthly performance
        monthly_avg = monthly_data.groupby('Month')['Revenue'].mean().reset_index()
        
        # Reorder months
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=month_order, ordered=True)
        monthly_avg = monthly_avg.sort_values('Month')
        
        # Create seasonal chart
        fig = px.bar(
            monthly_avg,
            x='Month',
            y='Revenue',
            title='Average Monthly Sales (Seasonal Pattern)',
            color='Revenue',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating seasonal analysis: {e}")

def create_bu_forecast(df, forecast_period):
    """
    Create business unit forecast
    """
    st.subheader("🏢 Business Unit Forecast")
    
    try:
        # BU selection
        bu_options = df['BU Name'].unique()
        selected_bu = st.selectbox("Select Business Unit", bu_options)
        
        # Filter data for selected BU
        bu_data = df[df['BU Name'] == selected_bu]
        
        # Prepare monthly data for BU
        bu_monthly = bu_data.groupby('YearMonth')['Total Line Amount'].sum().reset_index()
        bu_monthly.columns = ['YearMonth', 'Revenue']
        bu_monthly['YearMonth'] = pd.to_datetime(bu_monthly['YearMonth'])
        bu_monthly = bu_monthly.sort_values('YearMonth')
        bu_monthly['Date'] = bu_monthly['YearMonth']
        
        if len(bu_monthly) < 3:
            st.warning(f"Insufficient data for {selected_bu} forecasting")
            return
        
        # Create forecast
        periods = get_forecast_periods(forecast_period)
        bu_forecast = create_trend_forecast(bu_monthly, periods)
        
        # Display chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=bu_monthly['Date'],
            y=bu_monthly['Revenue'],
            mode='lines+markers',
            name=f'{selected_bu} Historical',
            line=dict(color='blue')
        ))
        
        if bu_forecast is not None:
            fig.add_trace(go.Scatter(
                x=bu_forecast['Date'],
                y=bu_forecast['Forecast'],
                mode='lines+markers',
                name=f'{selected_bu} Forecast',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=f'{selected_bu} Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # BU comparison
        st.subheader("📊 BU Performance Comparison")
        
        bu_comparison = df.groupby('BU Name')['Total Line Amount'].sum().reset_index()
        bu_comparison = bu_comparison.sort_values('Total Line Amount', ascending=False)
        
        fig = px.bar(
            bu_comparison,
            x='BU Name',
            y='Total Line Amount',
            title='Total Revenue by Business Unit',
            color='Total Line Amount',
            color_continuous_scale='blues'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating BU forecast: {e}")

def create_customer_forecast(df, forecast_period):
    """
    Create customer-specific forecast
    """
    st.subheader("👥 Customer Forecast")
    
    try:
        # Top customers by revenue
        top_customers = df.groupby('Cust Name')['Total Line Amount'].sum().nlargest(10)
        
        selected_customer = st.selectbox(
            "Select Customer",
            top_customers.index
        )
        
        # Filter data for selected customer
        customer_data = df[df['Cust Name'] == selected_customer]
        
        # Monthly customer data
        customer_monthly = customer_data.groupby('YearMonth')['Total Line Amount'].sum().reset_index()
        customer_monthly.columns = ['YearMonth', 'Revenue']
        customer_monthly['YearMonth'] = pd.to_datetime(customer_monthly['YearMonth'])
        customer_monthly = customer_monthly.sort_values('YearMonth')
        customer_monthly['Date'] = customer_monthly['YearMonth']
        
        if len(customer_monthly) < 2:
            st.warning(f"Insufficient data for {selected_customer} forecasting")
            return
        
        # Create forecast
        periods = get_forecast_periods(forecast_period)
        customer_forecast = create_trend_forecast(customer_monthly, periods)
        
        # Display chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=customer_monthly['Date'],
            y=customer_monthly['Revenue'],
            mode='lines+markers',
            name=f'{selected_customer} Historical',
            line=dict(color='blue')
        ))
        
        if customer_forecast is not None:
            fig.add_trace(go.Scatter(
                x=customer_forecast['Date'],
                y=customer_forecast['Forecast'],
                mode='lines+markers',
                name=f'{selected_customer} Forecast',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=f'{selected_customer} Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating customer forecast: {e}")

def create_product_forecast(df, forecast_period):
    """
    Create product category forecast
    """
    st.subheader("📦 Product Category Forecast")
    
    try:
        # Brand analysis
        brand_revenue = df.groupby('Brand')['Total Line Amount'].sum().reset_index()
        brand_revenue = brand_revenue.sort_values('Total Line Amount', ascending=False)
        
        # Remove null/NA brands
        brand_revenue = brand_revenue[brand_revenue['Brand'].notna()]
        brand_revenue = brand_revenue[brand_revenue['Brand'] != 'NA']
        
        if brand_revenue.empty:
            st.warning("No brand data available for forecasting")
            return
        
        selected_brand = st.selectbox(
            "Select Brand",
            brand_revenue['Brand'].head(10)
        )
        
        # Filter data for selected brand
        brand_data = df[df['Brand'] == selected_brand]
        
        # Monthly brand data
        brand_monthly = brand_data.groupby('YearMonth')['Total Line Amount'].sum().reset_index()
        brand_monthly.columns = ['YearMonth', 'Revenue']
        brand_monthly['YearMonth'] = pd.to_datetime(brand_monthly['YearMonth'])
        brand_monthly = brand_monthly.sort_values('YearMonth')
        brand_monthly['Date'] = brand_monthly['YearMonth']
        
        if len(brand_monthly) < 2:
            st.warning(f"Insufficient data for {selected_brand} forecasting")
            return
        
        # Create forecast
        periods = get_forecast_periods(forecast_period)
        brand_forecast = create_trend_forecast(brand_monthly, periods)
        
        # Display chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=brand_monthly['Date'],
            y=brand_monthly['Revenue'],
            mode='lines+markers',
            name=f'{selected_brand} Historical',
            line=dict(color='blue')
        ))
        
        if brand_forecast is not None:
            fig.add_trace(go.Scatter(
                x=brand_forecast['Date'],
                y=brand_forecast['Forecast'],
                mode='lines+markers',
                name=f'{selected_brand} Forecast',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=f'{selected_brand} Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Brand comparison
        st.subheader("📊 Brand Performance Comparison")
        
        fig = px.bar(
            brand_revenue.head(10),
            x='Brand',
            y='Total Line Amount',
            title='Top 10 Brands by Revenue',
            color='Total Line Amount',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating product forecast: {e}")
