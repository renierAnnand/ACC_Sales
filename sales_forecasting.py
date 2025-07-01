import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def create_sales_forecast(df):
    """
    Ultra-minimal sales forecasting that focuses on working first
    """
    st.header("🚀 Simple Sales Forecasting")
    st.markdown("*All amounts in Saudi Riyal (SAR)*")
    
    if df.empty:
        st.error("No data available for forecasting")
        return
    
    # Show what we have
    st.subheader("📊 Data Overview")
    st.write(f"Total records: {len(df)}")
    st.write(f"Columns: {list(df.columns)}")
    
    # Simple settings
    forecast_period = st.selectbox(
        "Forecast Period",
        ["Next 3 Months", "Next 6 Months", "Next 12 Months"],
        index=1
    )
    
    periods = {"Next 3 Months": 3, "Next 6 Months": 6, "Next 12 Months": 12}[forecast_period]
    
    if st.button("Generate Simple Forecast"):
        create_simple_forecast(df, periods, forecast_period)

def create_simple_forecast(df, periods, forecast_period):
    """Create the simplest possible forecast"""
    
    st.subheader("🎯 Simple Forecast")
    
    try:
        st.write("**Step 1: Data Preparation**")
        
        # Check columns
        if 'YearMonth' not in df.columns:
            st.error("Missing 'YearMonth' column")
            return
            
        if 'Total Line Amount' not in df.columns:
            st.error("Missing 'Total Line Amount' column")
            return
        
        # Simple grouping
        monthly_data = df.groupby('YearMonth')['Total Line Amount'].sum().reset_index()
        monthly_data.columns = ['YearMonth', 'Revenue']
        
        st.write(f"Grouped into {len(monthly_data)} monthly records")
        st.dataframe(monthly_data)
        
        if len(monthly_data) < 2:
            st.error("Need at least 2 months of data")
            return
        
        st.write("**Step 2: Simple Average Forecast**")
        
        # Ultra-simple forecast: average of all months
        avg_revenue = monthly_data['Revenue'].mean()
        st.write(f"Average monthly revenue: {avg_revenue:,.0f} SAR")
        
        # Generate forecast
        future_months = []
        future_values = []
        
        for i in range(periods):
            month_name = f"Month +{i+1}"
            future_months.append(month_name)
            future_values.append(avg_revenue)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Month': future_months,
            'Forecast': future_values
        })
        
        st.write("**Forecast Results:**")
        st.dataframe(forecast_df)
        
        # Simple chart
        fig = go.Figure()
        
        # Historical (just show last few months)
        recent_data = monthly_data.tail(6)
        fig.add_trace(go.Scatter(
            x=list(range(len(recent_data))),
            y=recent_data['Revenue'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        forecast_x = list(range(len(recent_data), len(recent_data) + periods))
        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=future_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Simple Sales Forecast',
            xaxis_title='Time Period',
            yaxis_title='Revenue (SAR)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        total_forecast = sum(future_values)
        st.success(f"✅ Forecast completed!")
        st.metric(f"Total Forecast ({forecast_period})", f"{total_forecast:,.0f} SAR")
        st.metric("Average Monthly Forecast", f"{avg_revenue:,.0f} SAR")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
