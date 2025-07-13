import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

def create_sales_dashboard(df):
    """
    Create the main sales dashboard with key metrics and visualizations
    """
    st.title("📊 Sales Dashboard")
    st.markdown("**Key performance metrics and business insights**")
    
    # Data validation
    if df is None or df.empty:
        st.error("No data available. Please upload a file first.")
        return
    
    # Ensure required columns exist
    required_columns = ['Total Line Amount', 'Invoice Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Date range filter
        if 'Invoice Date' in df.columns:
            min_date = df['Invoice Date'].min()
            max_date = df['Invoice Date'].max()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df[
                    (df['Invoice Date'] >= pd.Timestamp(start_date)) & 
                    (df['Invoice Date'] <= pd.Timestamp(end_date))
                ]
            else:
                df_filtered = df.copy()
        else:
            df_filtered = df.copy()
        
        # Business Unit filter
        if 'BU Name' in df.columns:
            bu_options = ['All'] + list(df['BU Name'].dropna().unique())
            selected_bu = st.selectbox("Business Unit", bu_options)
            
            if selected_bu != 'All':
                df_filtered = df_filtered[df_filtered['BU Name'] == selected_bu]
        
        # Salesperson filter
        if 'Salesperson Name' in df.columns:
            sales_options = ['All'] + list(df['Salesperson Name'].dropna().unique())
            selected_sales = st.selectbox("Salesperson", sales_options)
            
            if selected_sales != 'All':
                df_filtered = df_filtered[df_filtered['Salesperson Name'] == selected_sales]
    
    # Key Metrics Row
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df_filtered['Total Line Amount'].sum()
        st.metric(
            "Total Revenue",
            f"{total_revenue:,.0f} SAR",
            delta=None
        )
    
    with col2:
        if 'Total Cost' in df_filtered.columns:
            total_cost = df_filtered['Total Cost'].sum()
            total_profit = total_revenue - total_cost
            profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
            st.metric(
                "Total Profit",
                f"{total_profit:,.0f} SAR",
                delta=f"{profit_margin:.1f}% margin"
            )
        else:
            st.metric("Total Orders", f"{len(df_filtered):,}")
    
    with col3:
        if 'Invoice No.' in df_filtered.columns:
            unique_orders = df_filtered['Invoice No.'].nunique()
            avg_order_value = total_revenue / unique_orders if unique_orders > 0 else 0
            st.metric("Average Order Value", f"{avg_order_value:,.0f} SAR")
        else:
            st.metric("Data Points", f"{len(df_filtered):,}")
    
    with col4:
        if 'Cust Name' in df_filtered.columns:
            unique_customers = df_filtered['Cust Name'].nunique()
            st.metric("Unique Customers", f"{unique_customers:,}")
        else:
            st.metric("Date Range", f"{len(df_filtered['Invoice Date'].dt.date.unique())} days")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly Revenue Trend
        st.subheader("📈 Monthly Revenue Trend")
        if 'Invoice Date' in df_filtered.columns:
            monthly_data = df_filtered.groupby(df_filtered['Invoice Date'].dt.to_period('M')).agg({
                'Total Line Amount': 'sum'
            }).reset_index()
            monthly_data['Month'] = monthly_data['Invoice Date'].astype(str)
            
            fig = px.line(
                monthly_data,
                x='Month',
                y='Total Line Amount',
                title='Monthly Revenue Trend',
                markers=True
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Revenue (SAR)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date column not available for trend analysis")
    
    with col2:
        # Top Business Units
        st.subheader("🏢 Revenue by Business Unit")
        if 'BU Name' in df_filtered.columns:
            bu_revenue = df_filtered.groupby('BU Name')['Total Line Amount'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=bu_revenue.values,
                y=bu_revenue.index,
                orientation='h',
                title='Revenue by Business Unit',
                labels={'x': 'Revenue (SAR)', 'y': 'Business Unit'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Business Unit data not available")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Customers
        st.subheader("👥 Top Customers")
        if 'Cust Name' in df_filtered.columns:
            customer_revenue = df_filtered.groupby('Cust Name')['Total Line Amount'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=customer_revenue.index,
                y=customer_revenue.values,
                title='Top 10 Customers by Revenue',
                labels={'x': 'Customer', 'y': 'Revenue (SAR)'}
            )
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Customer data not available")
    
    with col2:
        # Sales Team Performance
        st.subheader("🌟 Sales Team Performance")
        if 'Salesperson Name' in df_filtered.columns:
            sales_performance = df_filtered.groupby('Salesperson Name')['Total Line Amount'].sum().sort_values(ascending=False).head(8)
            
            fig = px.pie(
                values=sales_performance.values,
                names=sales_performance.index,
                title='Revenue Distribution by Salesperson'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Salesperson data not available")
    
    # Detailed Tables Section
    st.subheader("📋 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Summary Statistics", "🏆 Top Performers", "📈 Trends"])
    
    with tab1:
        st.markdown("**Dataset Summary**")
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Revenue Statistics**")
            revenue_stats = df_filtered['Total Line Amount'].describe()
            stats_df = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [f"{revenue_stats['count']:,.0f}",
                         f"{revenue_stats['mean']:,.2f} SAR",
                         f"{revenue_stats['std']:,.2f} SAR",
                         f"{revenue_stats['min']:,.2f} SAR",
                         f"{revenue_stats['25%']:,.2f} SAR",
                         f"{revenue_stats['50%']:,.2f} SAR",
                         f"{revenue_stats['75%']:,.2f} SAR",
                         f"{revenue_stats['max']:,.2f} SAR"]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Data Overview**")
            overview_data = {
                'Metric': ['Total Records', 'Date Range', 'Business Units', 'Customers', 'Salespeople'],
                'Value': [
                    f"{len(df_filtered):,}",
                    f"{df_filtered['Invoice Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Invoice Date'].max().strftime('%Y-%m-%d')}" if 'Invoice Date' in df_filtered.columns else "N/A",
                    f"{df_filtered['BU Name'].nunique()}" if 'BU Name' in df_filtered.columns else "N/A",
                    f"{df_filtered['Cust Name'].nunique()}" if 'Cust Name' in df_filtered.columns else "N/A",
                    f"{df_filtered['Salesperson Name'].nunique()}" if 'Salesperson Name' in df_filtered.columns else "N/A"
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("**Top Performers**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Cust Name' in df_filtered.columns:
                st.markdown("**Top 10 Customers**")
                top_customers = df_filtered.groupby('Cust Name').agg({
                    'Total Line Amount': 'sum',
                    'Invoice No.': 'nunique' if 'Invoice No.' in df_filtered.columns else 'count'
                }).round(2).sort_values('Total Line Amount', ascending=False).head(10)
                top_customers.columns = ['Revenue (SAR)', 'Orders']
                st.dataframe(top_customers, use_container_width=True)
        
        with col2:
            if 'Salesperson Name' in df_filtered.columns:
                st.markdown("**Top Salespeople**")
                top_sales = df_filtered.groupby('Salesperson Name').agg({
                    'Total Line Amount': 'sum',
                    'Cust Name': 'nunique' if 'Cust Name' in df_filtered.columns else 'count'
                }).round(2).sort_values('Total Line Amount', ascending=False).head(10)
                top_sales.columns = ['Revenue (SAR)', 'Customers']
                st.dataframe(top_sales, use_container_width=True)
    
    with tab3:
        st.markdown("**Trend Analysis**")
        
        if 'Invoice Date' in df_filtered.columns:
            # Weekly trends
            df_filtered['Week'] = df_filtered['Invoice Date'].dt.isocalendar().week
            weekly_trends = df_filtered.groupby('Week')['Total Line Amount'].sum()
            
            fig = px.bar(
                x=weekly_trends.index,
                y=weekly_trends.values,
                title='Weekly Revenue Distribution',
                labels={'x': 'Week of Year', 'y': 'Revenue (SAR)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show trends table
            monthly_summary = df_filtered.groupby(df_filtered['Invoice Date'].dt.to_period('M')).agg({
                'Total Line Amount': ['sum', 'mean', 'count']
            }).round(2)
            monthly_summary.columns = ['Total Revenue', 'Avg Transaction', 'Transaction Count']
            monthly_summary.index = monthly_summary.index.astype(str)
            st.dataframe(monthly_summary, use_container_width=True)
        else:
            st.info("Date information not available for trend analysis")
    
    # Export functionality
    st.subheader("💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Export Summary"):
            # Create summary report
            summary_data = {
                'Metric': ['Total Revenue', 'Total Records', 'Date Range', 'Business Units', 'Customers'],
                'Value': [
                    f"{df_filtered['Total Line Amount'].sum():,.2f} SAR",
                    f"{len(df_filtered):,}",
                    f"{df_filtered['Invoice Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Invoice Date'].max().strftime('%Y-%m-%d')}" if 'Invoice Date' in df_filtered.columns else "N/A",
                    f"{df_filtered['BU Name'].nunique()}" if 'BU Name' in df_filtered.columns else "N/A",
                    f"{df_filtered['Cust Name'].nunique()}" if 'Cust Name' in df_filtered.columns else "N/A"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary CSV",
                data=csv,
                file_name=f"dashboard_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Export Filtered Data"):
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data CSV",
                data=csv,
                file_name=f"filtered_sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        st.info(f"**Current View:** {len(df_filtered):,} records")
        st.info(f"**Revenue:** {df_filtered['Total Line Amount'].sum():,.0f} SAR")
