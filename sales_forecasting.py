import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

def create_sales_forecast(df):
    """
    Enhanced sales forecasting with Business Unit analysis
    """
    st.header("🚀 Enhanced Sales Forecasting")
    st.markdown("*All amounts in Saudi Riyal (SAR)*")
    
    if df.empty:
        st.error("No data available for forecasting")
        return
    
    # Show what we have
    st.subheader("📊 Data Overview")
    st.write(f"Total records: {len(df)}")
    st.write(f"Columns: {list(df.columns)}")
    
    # Check if we have Business Unit data
    has_bu_data = 'BU Name' in df.columns
    
    if has_bu_data:
        bu_names = df['BU Name'].dropna().unique()
        st.write(f"Business Units found: {len(bu_names)}")
        if len(bu_names) > 10:
            st.write(f"First 10 BUs: {list(bu_names[:10])}")
        else:
            st.write(f"BUs: {list(bu_names)}")
    
    # Forecasting options
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_type = st.selectbox(
            "Forecast Type",
            ["Overall Sales", "Business Unit Analysis", "BU Comparison"] if has_bu_data else ["Overall Sales"]
        )
        
        forecast_period = st.selectbox(
            "Forecast Period",
            ["Next 3 Months", "Next 6 Months", "Next 12 Months"],
            index=1
        )
    
    with col2:
        if forecast_type == "Business Unit Analysis" and has_bu_data:
            selected_bu = st.selectbox(
                "Select Business Unit",
                bu_names
            )
        elif forecast_type == "BU Comparison" and has_bu_data:
            selected_bus = st.multiselect(
                "Select Business Units to Compare",
                bu_names,
                default=list(bu_names[:min(5, len(bu_names))])
            )
    
    periods = {"Next 3 Months": 3, "Next 6 Months": 6, "Next 12 Months": 12}[forecast_period]
    
    if st.button("Generate Forecast"):
        if forecast_type == "Overall Sales":
            create_overall_forecast(df, periods, forecast_period)
        elif forecast_type == "Business Unit Analysis":
            create_bu_forecast(df, selected_bu, periods, forecast_period)
        elif forecast_type == "BU Comparison":
            create_bu_comparison_forecast(df, selected_bus, periods, forecast_period)

def create_overall_forecast(df, periods, forecast_period):
    """Create overall company forecast"""
    
    st.subheader("🎯 Overall Sales Forecast")
    
    try:
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
        monthly_data = monthly_data.sort_values('YearMonth')
        
        st.write(f"✅ Data prepared: {len(monthly_data)} monthly records")
        
        if len(monthly_data) < 2:
            st.error("Need at least 2 months of data")
            return
        
        # Show recent performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_revenue = monthly_data['Revenue'].sum()
            st.metric("Total Revenue", f"{total_revenue:,.0f} SAR")
        
        with col2:
            avg_revenue = monthly_data['Revenue'].mean()
            st.metric("Avg Monthly Revenue", f"{avg_revenue:,.0f} SAR")
        
        with col3:
            if len(monthly_data) >= 2:
                growth = ((monthly_data['Revenue'].iloc[-1] / monthly_data['Revenue'].iloc[-2]) - 1) * 100
                st.metric("Last Month Growth", f"{growth:.1f}%")
        
        # Generate simple forecast
        forecast_results = generate_simple_forecast(monthly_data, periods)
        
        # Display results
        display_forecast_chart(monthly_data, forecast_results, "Overall Sales Forecast")
        display_forecast_summary(forecast_results, forecast_period, "Overall")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def create_bu_forecast(df, selected_bu, periods, forecast_period):
    """Create forecast for specific Business Unit"""
    
    st.subheader(f"🏢 {selected_bu} - Business Unit Forecast")
    
    try:
        # Filter for selected BU
        bu_data = df[df['BU Name'] == selected_bu].copy()
        
        if bu_data.empty:
            st.error(f"No data found for {selected_bu}")
            return
        
        st.write(f"📊 Analyzing {len(bu_data)} records for {selected_bu}")
        
        # Group by month
        monthly_data = bu_data.groupby('YearMonth')['Total Line Amount'].sum().reset_index()
        monthly_data.columns = ['YearMonth', 'Revenue']
        monthly_data = monthly_data.sort_values('YearMonth')
        
        if len(monthly_data) < 2:
            st.error(f"Need at least 2 months of data for {selected_bu}")
            return
        
        # Show BU performance
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bu_total = monthly_data['Revenue'].sum()
            st.metric("BU Total Revenue", f"{bu_total:,.0f} SAR")
        
        with col2:
            bu_avg = monthly_data['Revenue'].mean()
            st.metric("BU Avg Monthly", f"{bu_avg:,.0f} SAR")
        
        with col3:
            company_total = df['Total Line Amount'].sum()
            bu_share = (bu_total / company_total) * 100
            st.metric("Share of Total", f"{bu_share:.1f}%")
        
        with col4:
            if len(monthly_data) >= 2:
                bu_growth = ((monthly_data['Revenue'].iloc[-1] / monthly_data['Revenue'].iloc[-2]) - 1) * 100
                st.metric("Last Month Growth", f"{bu_growth:.1f}%")
        
        # Generate forecast
        forecast_results = generate_simple_forecast(monthly_data, periods)
        
        # Display results
        display_forecast_chart(monthly_data, forecast_results, f"{selected_bu} Forecast")
        display_forecast_summary(forecast_results, forecast_period, selected_bu)
        
        # Show detailed data
        with st.expander("📋 Detailed Monthly Data"):
            st.dataframe(monthly_data)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def create_bu_comparison_forecast(df, selected_bus, periods, forecast_period):
    """Create comparison forecast for multiple Business Units"""
    
    st.subheader("⚖️ Business Unit Comparison Forecast")
    
    try:
        if not selected_bus:
            st.warning("Please select at least one Business Unit")
            return
        
        # Prepare data for each BU
        bu_forecasts = {}
        bu_historical = {}
        
        for bu in selected_bus:
            bu_data = df[df['BU Name'] == bu].copy()
            
            if not bu_data.empty:
                monthly_data = bu_data.groupby('YearMonth')['Total Line Amount'].sum().reset_index()
                monthly_data.columns = ['YearMonth', 'Revenue']
                monthly_data = monthly_data.sort_values('YearMonth')
                
                if len(monthly_data) >= 2:
                    bu_historical[bu] = monthly_data
                    bu_forecasts[bu] = generate_simple_forecast(monthly_data, periods)
        
        if not bu_forecasts:
            st.error("No valid data found for selected Business Units")
            return
        
        # Comparison metrics
        st.subheader("📊 BU Performance Comparison")
        
        comparison_data = []
        for bu in selected_bus:
            if bu in bu_historical:
                historical = bu_historical[bu]
                forecast = bu_forecasts[bu]
                
                total_historical = historical['Revenue'].sum()
                avg_monthly = historical['Revenue'].mean()
                forecast_total = forecast['Forecast'].sum()
                growth_projection = ((forecast['Forecast'].mean() / avg_monthly) - 1) * 100
                
                comparison_data.append({
                    'Business Unit': bu,
                    'Historical Total': f"{total_historical:,.0f} SAR",
                    'Avg Monthly': f"{avg_monthly:,.0f} SAR",
                    f'Forecast Total ({forecast_period})': f"{forecast_total:,.0f} SAR",
                    'Growth Projection': f"{growth_projection:.1f}%"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Comparison charts
        create_bu_comparison_charts(bu_historical, bu_forecasts, forecast_period)
        
        # Top performers
        st.subheader("🏆 Top Performers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top by revenue
            revenue_ranking = [(bu, bu_historical[bu]['Revenue'].sum()) for bu in bu_historical.keys()]
            revenue_ranking.sort(key=lambda x: x[1], reverse=True)
            
            st.write("**Top by Historical Revenue:**")
            for i, (bu, revenue) in enumerate(revenue_ranking[:3], 1):
                st.write(f"{i}. {bu}: {revenue:,.0f} SAR")
        
        with col2:
            # Top by growth projection
            growth_ranking = []
            for bu in bu_historical.keys():
                if bu in bu_forecasts:
                    historical_avg = bu_historical[bu]['Revenue'].mean()
                    forecast_avg = bu_forecasts[bu]['Forecast'].mean()
                    growth = ((forecast_avg / historical_avg) - 1) * 100
                    growth_ranking.append((bu, growth))
            
            growth_ranking.sort(key=lambda x: x[1], reverse=True)
            
            st.write("**Top by Growth Projection:**")
            for i, (bu, growth) in enumerate(growth_ranking[:3], 1):
                st.write(f"{i}. {bu}: {growth:.1f}%")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def create_bu_comparison_charts(bu_historical, bu_forecasts, forecast_period):
    """Create comparison charts for multiple BUs"""
    
    # Historical comparison
    st.subheader("📈 Historical Performance Comparison")
    
    fig_historical = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, (bu, historical_data) in enumerate(bu_historical.items()):
        color = colors[i % len(colors)]
        fig_historical.add_trace(go.Scatter(
            x=historical_data['YearMonth'],
            y=historical_data['Revenue'],
            mode='lines+markers',
            name=bu,
            line=dict(color=color)
        ))
    
    fig_historical.update_layout(
        title='Historical Revenue by Business Unit',
        xaxis_title='Month',
        yaxis_title='Revenue (SAR)',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_historical, use_container_width=True)
    
    # Forecast comparison
    st.subheader(f"🔮 {forecast_period} Forecast Comparison")
    
    fig_forecast = go.Figure()
    
    for i, (bu, forecast_data) in enumerate(bu_forecasts.items()):
        color = colors[i % len(colors)]
        fig_forecast.add_trace(go.Bar(
            x=[f"Month {j+1}" for j in range(len(forecast_data))],
            y=forecast_data['Forecast'],
            name=bu,
            marker_color=color
        ))
    
    fig_forecast.update_layout(
        title=f'Forecast Comparison - {forecast_period}',
        xaxis_title='Forecast Period',
        yaxis_title='Forecasted Revenue (SAR)',
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)

def generate_simple_forecast(monthly_data, periods):
    """Generate simple forecast using multiple methods"""
    
    revenue = monthly_data['Revenue'].values
    
    # Method 1: Simple average
    avg_forecast = np.mean(revenue)
    
    # Method 2: Weighted average (more weight to recent months)
    weights = np.linspace(1, 2, len(revenue))
    weighted_avg = np.average(revenue, weights=weights)
    
    # Method 3: Trend-based (simple linear trend)
    if len(revenue) >= 3:
        x = np.arange(len(revenue))
        slope = (revenue[-1] - revenue[0]) / (len(revenue) - 1)
        trend_forecast = revenue[-1] + slope
    else:
        trend_forecast = avg_forecast
    
    # Combine methods (simple ensemble)
    final_forecast = (avg_forecast * 0.4 + weighted_avg * 0.4 + trend_forecast * 0.2)
    
    # Generate forecast dataframe
    forecast_months = [f"Month +{i+1}" for i in range(periods)]
    forecast_values = [final_forecast] * periods
    
    return pd.DataFrame({
        'Month': forecast_months,
        'Forecast': forecast_values
    })

def display_forecast_chart(monthly_data, forecast_results, title):
    """Display forecast chart"""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=monthly_data['YearMonth'],
        y=monthly_data['Revenue'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=3)
    ))
    
    # Forecast (create simple x-axis for forecast)
    last_date_idx = len(monthly_data) - 1
    forecast_x = [f"Forecast {i+1}" for i in range(len(forecast_results))]
    
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_results['Forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash', width=3)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Period',
        yaxis_title='Revenue (SAR)',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_forecast_summary(forecast_results, forecast_period, entity_name):
    """Display forecast summary"""
    
    total_forecast = forecast_results['Forecast'].sum()
    avg_forecast = forecast_results['Forecast'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            f"{entity_name} Total Forecast ({forecast_period})", 
            f"{total_forecast:,.0f} SAR"
        )
    
    with col2:
        st.metric(
            f"{entity_name} Avg Monthly Forecast", 
            f"{avg_forecast:,.0f} SAR"
        )
    
    # Detailed forecast table
    with st.expander("📋 Detailed Forecast Breakdown"):
        st.dataframe(forecast_results, use_container_width=True)
