import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

def create_sales_dashboard(df):
    """
    Create comprehensive sales dashboard with key metrics and visualizations
    """
    st.header("📊 Sales Dashboard")
    
    if df.empty:
        st.error("No data available for dashboard")
        return
    
    # Key Metrics Row
    create_key_metrics(df)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        create_monthly_sales_trend(df)
    
    with col2:
        create_bu_performance_chart(df)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        create_top_customers_chart(df)
    
    with col2:
        create_sales_by_salesperson(df)
    
    # Charts Row 3
    col1, col2 = st.columns(2)
    
    with col1:
        create_product_performance(df)
    
    with col2:
        create_customer_class_distribution(df)
    
    # Additional insights
    create_geographic_analysis(df)

def create_key_metrics(df):
    """
    Create key performance metrics cards
    """
    st.subheader("📈 Key Performance Metrics")
    
    # Calculate metrics
    total_revenue = df['Total Line Amount'].sum()
    total_cost = df['Total Cost'].sum() if 'Total Cost' in df.columns else 0
    total_profit = total_revenue - total_cost
    avg_order_value = df['Total Line Amount'].mean()
    total_orders = df['Invoice No.'].nunique()
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta=f"{total_revenue/1000000:.1f}M"
        )
    
    with col2:
        st.metric(
            label="Total Profit", 
            value=f"${total_profit:,.0f}",
            delta=f"{((total_profit/total_revenue)*100):.1f}% margin" if total_revenue > 0 else "0%"
        )
    
    with col3:
        st.metric(
            label="Avg Order Value",
            value=f"${avg_order_value:,.0f}",
            delta=f"Per order"
        )
    
    with col4:
        st.metric(
            label="Total Orders",
            value=f"{total_orders:,}",
            delta=f"Unique invoices"
        )
    
    with col5:
        customers = df['Cust Name'].nunique()
        st.metric(
            label="Active Customers",
            value=f"{customers:,}",
            delta=f"Unique customers"
        )

def create_monthly_sales_trend(df):
    """
    Create monthly sales trend chart
    """
    st.subheader("📈 Monthly Sales Trend")
    
    try:
        # Group by month
        monthly_sales = df.groupby('YearMonth')['Total Line Amount'].sum().reset_index()
        monthly_sales['YearMonth'] = pd.to_datetime(monthly_sales['YearMonth'])
        monthly_sales = monthly_sales.sort_values('YearMonth')
        
        # Create line chart
        fig = px.line(
            monthly_sales,
            x='YearMonth',
            y='Total Line Amount',
            title='Monthly Sales Revenue Trend',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Revenue ($)",
            showlegend=False,
            height=400
        )
        
        fig.update_traces(line_color='#1f77b4', line_width=3)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating monthly trend chart: {e}")

def create_bu_performance_chart(df):
    """
    Create business unit performance chart
    """
    st.subheader("🏢 Business Unit Performance")
    
    try:
        # Group by BU
        bu_performance = df.groupby('BU Name').agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        bu_performance['Profit'] = bu_performance['Total Line Amount'] - bu_performance['Total Cost']
        bu_performance = bu_performance.sort_values('Total Line Amount', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            bu_performance,
            x='Total Line Amount',
            y='BU Name',
            orientation='h',
            title='Revenue by Business Unit',
            color='Profit',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            xaxis_title="Revenue ($)",
            yaxis_title="Business Unit",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating BU performance chart: {e}")

def create_top_customers_chart(df):
    """
    Create top customers chart
    """
    st.subheader("👥 Top Customers by Revenue")
    
    try:
        # Group by customer
        customer_sales = df.groupby('Cust Name')['Total Line Amount'].sum().reset_index()
        top_customers = customer_sales.nlargest(10, 'Total Line Amount')
        
        # Create bar chart
        fig = px.bar(
            top_customers,
            x='Total Line Amount',
            y='Cust Name',
            orientation='h',
            title='Top 10 Customers by Revenue'
        )
        
        fig.update_layout(
            xaxis_title="Revenue ($)",
            yaxis_title="Customer",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating top customers chart: {e}")

def create_sales_by_salesperson(df):
    """
    Create sales by salesperson chart
    """
    st.subheader("🏆 Sales by Salesperson")
    
    try:
        # Group by salesperson
        salesperson_sales = df.groupby('Salesperson Name').agg({
            'Total Line Amount': 'sum',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        salesperson_sales = salesperson_sales.sort_values('Total Line Amount', ascending=False).head(10)
        
        # Create bar chart
        fig = px.bar(
            salesperson_sales,
            x='Salesperson Name',
            y='Total Line Amount',
            title='Top 10 Salespeople by Revenue',
            color='Total Line Amount',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Salesperson",
            yaxis_title="Revenue ($)",
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating salesperson chart: {e}")

def create_product_performance(df):
    """
    Create product performance chart
    """
    st.subheader("📦 Product Performance by Brand")
    
    try:
        # Group by brand
        brand_performance = df.groupby('Brand')['Total Line Amount'].sum().reset_index()
        brand_performance = brand_performance.sort_values('Total Line Amount', ascending=False).head(8)
        
        # Create pie chart
        fig = px.pie(
            brand_performance,
            values='Total Line Amount',
            names='Brand',
            title='Revenue Distribution by Brand'
        )
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating product performance chart: {e}")

def create_customer_class_distribution(df):
    """
    Create customer class distribution chart
    """
    st.subheader("🎯 Customer Class Distribution")
    
    try:
        # Group by customer class
        class_distribution = df.groupby('Cust Class Code').agg({
            'Total Line Amount': 'sum',
            'Cust Name': 'nunique'
        }).reset_index()
        
        # Create donut chart
        fig = px.pie(
            class_distribution,
            values='Total Line Amount',
            names='Cust Class Code',
            title='Revenue by Customer Class',
            hole=0.4
        )
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating customer class chart: {e}")

def create_geographic_analysis(df):
    """
    Create geographic analysis
    """
    st.subheader("🌍 Geographic Analysis")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Local vs International
            local_intl = df.groupby('Local')['Total Line Amount'].sum().reset_index()
            
            fig = px.bar(
                local_intl,
                x='Local',
                y='Total Line Amount',
                title='Local vs International Sales',
                color='Local'
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sales by country
            country_sales = df.groupby('Country')['Total Line Amount'].sum().reset_index()
            country_sales = country_sales.sort_values('Total Line Amount', ascending=False).head(5)
            
            fig = px.bar(
                country_sales,
                x='Country',
                y='Total Line Amount',
                title='Top Countries by Sales',
                color='Total Line Amount',
                color_continuous_scale='blues'
            )
            
            fig.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error creating geographic analysis: {e}")
