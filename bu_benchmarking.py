import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

def create_bu_benchmark(df):
    """
    Create comprehensive business unit benchmarking analysis
    """
    st.header("🏢 Business Unit Benchmarking")
    
    if df.empty:
        st.error("No data available for BU benchmarking")
        return
    
    # BU Performance Overview
    create_bu_overview(df)
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Benchmarking Analysis",
        ["Overall Performance Comparison", "Financial Metrics", "Operational Metrics", "Customer Analysis", "Trend Analysis"]
    )
    
    if analysis_type == "Overall Performance Comparison":
        create_overall_bu_comparison(df)
    elif analysis_type == "Financial Metrics":
        create_financial_metrics(df)
    elif analysis_type == "Operational Metrics":
        create_operational_metrics(df)
    elif analysis_type == "Customer Analysis":
        create_customer_analysis_by_bu(df)
    elif analysis_type == "Trend Analysis":
        create_bu_trend_analysis(df)

def create_bu_overview(df):
    """
    Create BU performance overview
    """
    st.subheader("📊 Business Unit Overview")
    
    try:
        # Calculate BU metrics
        bu_metrics = df.groupby('BU Name').agg({
            'Total Line Amount': ['sum', 'mean', 'count'],
            'Total Cost': 'sum',
            'QTY': 'sum',
            'Cust Name': 'nunique',
            'Salesperson Name': 'nunique',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        # Flatten column names
        bu_metrics.columns = [
            'BU_Name', 'Total_Revenue', 'Avg_Deal_Size', 'Deal_Count',
            'Total_Cost', 'Total_Quantity', 'Customer_Count', 'Salesperson_Count', 'Order_Count'
        ]
        
        # Calculate additional metrics
        bu_metrics['Profit'] = bu_metrics['Total_Revenue'] - bu_metrics['Total_Cost']
        bu_metrics['Profit_Margin'] = (
            bu_metrics['Profit'] / bu_metrics['Total_Revenue'] * 100
        ).round(2)
        bu_metrics['Revenue_Per_Customer'] = (
            bu_metrics['Total_Revenue'] / bu_metrics['Customer_Count']
        ).round(0)
        bu_metrics['Revenue_Per_Salesperson'] = (
            bu_metrics['Total_Revenue'] / bu_metrics['Salesperson_Count']
        ).round(0)
        
        # Display key metrics
        total_revenue = bu_metrics['Total_Revenue'].sum()
        total_profit = bu_metrics['Profit'].sum()
        overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Business Units", len(bu_metrics))
        
        with col2:
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        
        with col3:
            st.metric("Overall Profit Margin", f"{overall_margin:.1f}%")
        
        with col4:
            best_bu = bu_metrics.loc[bu_metrics['Total_Revenue'].idxmax(), 'BU_Name']
            st.metric("Top Performing BU", best_bu)
        
        # BU Performance Table
        st.subheader("📋 BU Performance Summary")
        
        # Format for display
        display_metrics = bu_metrics.copy()
        display_metrics['Total_Revenue'] = display_metrics['Total_Revenue'].apply(lambda x: f"${x:,.0f}")
        display_metrics['Avg_Deal_Size'] = display_metrics['Avg_Deal_Size'].apply(lambda x: f"${x:,.0f}")
        display_metrics['Profit'] = display_metrics['Profit'].apply(lambda x: f"${x:,.0f}")
        display_metrics['Revenue_Per_Customer'] = display_metrics['Revenue_Per_Customer'].apply(lambda x: f"${x:,.0f}")
        display_metrics['Revenue_Per_Salesperson'] = display_metrics['Revenue_Per_Salesperson'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_metrics, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating BU overview: {e}")

def create_overall_bu_comparison(df):
    """
    Create overall BU performance comparison
    """
    st.subheader("🔄 Overall Performance Comparison")
    
    try:
        # Calculate BU metrics
        bu_metrics = df.groupby('BU Name').agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Invoice No.': 'nunique',
            'Cust Name': 'nunique'
        }).reset_index()
        
        bu_metrics['Profit'] = bu_metrics['Total Line Amount'] - bu_metrics['Total Cost']
        bu_metrics['Profit_Margin'] = (
            bu_metrics['Profit'] / bu_metrics['Total Line Amount'] * 100
        ).round(2)
        
        # Revenue comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue bar chart
            fig = px.bar(
                bu_metrics.sort_values('Total Line Amount', ascending=True),
                x='Total Line Amount',
                y='BU Name',
                orientation='h',
                title='Total Revenue by Business Unit',
                color='Total Line Amount',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Market share pie chart
            fig = px.pie(
                bu_metrics,
                values='Total Line Amount',
                names='BU Name',
                title='Revenue Market Share by BU'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison radar chart
        st.subheader("🎯 Multi-Dimensional Performance Comparison")
        
        # Normalize metrics for radar chart
        metrics_for_radar = ['Total Line Amount', 'Profit_Margin', 'Invoice No.', 'Cust Name']
        bu_radar = bu_metrics.copy()
        
        for metric in metrics_for_radar:
            max_val = bu_radar[metric].max()
            bu_radar[f'{metric}_norm'] = (bu_radar[metric] / max_val * 100).round(1)
        
        # Create radar chart
        fig = go.Figure()
        
        for _, row in bu_radar.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[
                    row['Total Line Amount_norm'],
                    row['Profit_Margin_norm'],
                    row['Invoice No._norm'],
                    row['Cust Name_norm'],
                    row['Total Line Amount_norm']  # Close the polygon
                ],
                theta=['Revenue', 'Profit Margin', 'Orders', 'Customers', 'Revenue'],
                fill='toself',
                name=row['BU Name']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="BU Performance Radar Chart (Normalized Scores)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance ranking
        st.subheader("🏆 BU Performance Rankings")
        
        # Calculate composite performance score
        bu_metrics['Revenue_Score'] = (bu_metrics['Total Line Amount'] / bu_metrics['Total Line Amount'].max() * 40).round(1)
        bu_metrics['Margin_Score'] = (bu_metrics['Profit_Margin'] / bu_metrics['Profit_Margin'].max() * 30).round(1)
        bu_metrics['Customer_Score'] = (bu_metrics['Cust Name'] / bu_metrics['Cust Name'].max() * 20).round(1)
        bu_metrics['Order_Score'] = (bu_metrics['Invoice No.'] / bu_metrics['Invoice No.'].max() * 10).round(1)
        
        bu_metrics['Composite_Score'] = (
            bu_metrics['Revenue_Score'] + 
            bu_metrics['Margin_Score'] + 
            bu_metrics['Customer_Score'] + 
            bu_metrics['Order_Score']
        ).round(1)
        
        bu_ranking = bu_metrics.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        bu_ranking['Rank'] = range(1, len(bu_ranking) + 1)
        
        fig = px.bar(
            bu_ranking,
            x='BU Name',
            y='Composite_Score',
            title='BU Composite Performance Score',
            color='Composite_Score',
            color_continuous_scale='viridis',
            text='Rank'
        )
        fig.update_traces(texttemplate='#%{text}', textposition="outside")
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating overall BU comparison: {e}")

def create_financial_metrics(df):
    """
    Create financial metrics comparison
    """
    st.subheader("💰 Financial Metrics Analysis")
    
    try:
        # Calculate detailed financial metrics
        bu_financial = df.groupby('BU Name').agg({
            'Total Line Amount': ['sum', 'mean', 'std'],
            'Total Cost': 'sum',
            'Tax Amount': 'sum',
            'QTY': 'sum'
        }).reset_index()
        
        # Flatten columns
        bu_financial.columns = [
            'BU_Name', 'Total_Revenue', 'Avg_Revenue', 'Revenue_StdDev',
            'Total_Cost', 'Total_Tax', 'Total_Quantity'
        ]
        
        # Calculate financial ratios
        bu_financial['Gross_Profit'] = bu_financial['Total_Revenue'] - bu_financial['Total_Cost']
        bu_financial['Gross_Margin'] = (
            bu_financial['Gross_Profit'] / bu_financial['Total_Revenue'] * 100
        ).round(2)
        bu_financial['Cost_Ratio'] = (
            bu_financial['Total_Cost'] / bu_financial['Total_Revenue'] * 100
        ).round(2)
        bu_financial['Revenue_Per_Unit'] = (
            bu_financial['Total_Revenue'] / bu_financial['Total_Quantity']
        ).round(2)
        bu_financial['Cost_Per_Unit'] = (
            bu_financial['Total_Cost'] / bu_financial['Total_Quantity']
        ).round(2)
        
        # Financial metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Gross margin comparison
            fig = px.bar(
                bu_financial.sort_values('Gross_Margin', ascending=True),
                x='Gross_Margin',
                y='BU_Name',
                orientation='h',
                title='Gross Profit Margin by BU (%)',
                color='Gross_Margin',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue vs Cost scatter
            fig = px.scatter(
                bu_financial,
                x='Total_Cost',
                y='Total_Revenue',
                size='Total_Quantity',
                color='Gross_Margin',
                hover_name='BU_Name',
                title='Revenue vs Cost Analysis',
                color_continuous_scale='viridis'
            )
            
            # Add diagonal line for break-even
            max_val = max(bu_financial['Total_Revenue'].max(), bu_financial['Total_Cost'].max())
            fig.add_shape(
                type="line",
                x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="red", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=max_val*0.8, y=max_val*0.8,
                text="Break-even line",
                showarrow=True,
                arrowhead=2
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Unit economics
        st.subheader("📊 Unit Economics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue per unit
            fig = px.bar(
                bu_financial.sort_values('Revenue_Per_Unit', ascending=True),
                x='Revenue_Per_Unit',
                y='BU_Name',
                orientation='h',
                title='Revenue per Unit by BU',
                color='Revenue_Per_Unit',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost per unit
            fig = px.bar(
                bu_financial.sort_values('Cost_Per_Unit', ascending=True),
                x='Cost_Per_Unit',
                y='BU_Name',
                orientation='h',
                title='Cost per Unit by BU',
                color='Cost_Per_Unit',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Financial performance table
        st.subheader("📋 Financial Performance Summary")
        
        display_financial = bu_financial.copy()
        financial_cols = ['Total_Revenue', 'Total_Cost', 'Gross_Profit']
        for col in financial_cols:
            display_financial[col] = display_financial[col].apply(lambda x: f"${x:,.0f}")
        
        display_financial['Revenue_Per_Unit'] = display_financial['Revenue_Per_Unit'].apply(lambda x: f"${x:,.2f}")
        display_financial['Cost_Per_Unit'] = display_financial['Cost_Per_Unit'].apply(lambda x: f"${x:,.2f}")
        display_financial['Gross_Margin'] = display_financial['Gross_Margin'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_financial[[
            'BU_Name', 'Total_Revenue', 'Total_Cost', 'Gross_Profit', 
            'Gross_Margin', 'Revenue_Per_Unit', 'Cost_Per_Unit'
        ]], use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating financial metrics: {e}")

def create_operational_metrics(df):
    """
    Create operational metrics comparison
    """
    st.subheader("⚙️ Operational Metrics Analysis")
    
    try:
        # Calculate operational metrics
        bu_operational = df.groupby('BU Name').agg({
            'Invoice No.': 'nunique',
            'Cust Name': 'nunique',
            'Salesperson Name': 'nunique',
            'Total Line Amount': ['sum', 'mean', 'count'],
            'QTY': ['sum', 'mean'],
            'Invoice Date': ['min', 'max']
        }).reset_index()
        
        # Flatten columns
        bu_operational.columns = [
            'BU_Name', 'Order_Count', 'Customer_Count', 'Salesperson_Count',
            'Total_Revenue', 'Avg_Deal_Size', 'Deal_Count', 'Total_Quantity', 'Avg_Quantity',
            'First_Sale', 'Last_Sale'
        ]
        
        # Calculate efficiency metrics
        bu_operational['Revenue_Per_Order'] = (
            bu_operational['Total_Revenue'] / bu_operational['Order_Count']
        ).round(0)
        bu_operational['Revenue_Per_Customer'] = (
            bu_operational['Total_Revenue'] / bu_operational['Customer_Count']
        ).round(0)
        bu_operational['Revenue_Per_Salesperson'] = (
            bu_operational['Total_Revenue'] / bu_operational['Salesperson_Count']
        ).round(0)
        bu_operational['Orders_Per_Customer'] = (
            bu_operational['Order_Count'] / bu_operational['Customer_Count']
        ).round(2)
        bu_operational['Customers_Per_Salesperson'] = (
            bu_operational['Customer_Count'] / bu_operational['Salesperson_Count']
        ).round(1)
        
        # Operational efficiency comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue per salesperson
            fig = px.bar(
                bu_operational.sort_values('Revenue_Per_Salesperson', ascending=True),
                x='Revenue_Per_Salesperson',
                y='BU_Name',
                orientation='h',
                title='Revenue per Salesperson by BU',
                color='Revenue_Per_Salesperson',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Orders per customer
            fig = px.bar(
                bu_operational.sort_values('Orders_Per_Customer', ascending=True),
                x='Orders_Per_Customer',
                y='BU_Name',
                orientation='h',
                title='Orders per Customer by BU',
                color='Orders_Per_Customer',
                color_continuous_scale='plasma'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer and sales efficiency
        st.subheader("👥 Customer & Sales Efficiency")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer concentration
            fig = px.scatter(
                bu_operational,
                x='Customer_Count',
                y='Total_Revenue',
                size='Salesperson_Count',
                color='Revenue_Per_Customer',
                hover_name='BU_Name',
                title='Customer Base vs Revenue',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Salesperson efficiency
            fig = px.scatter(
                bu_operational,
                x='Salesperson_Count',
                y='Revenue_Per_Salesperson',
                size='Customer_Count',
                color='BU_Name',
                title='Salesperson Count vs Productivity'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Operational metrics table
        st.subheader("📊 Operational Efficiency Summary")
        
        efficiency_summary = bu_operational[[
            'BU_Name', 'Order_Count', 'Customer_Count', 'Salesperson_Count',
            'Revenue_Per_Order', 'Revenue_Per_Customer', 'Revenue_Per_Salesperson',
            'Orders_Per_Customer', 'Customers_Per_Salesperson'
        ]].copy()
        
        # Format monetary columns
        money_cols = ['Revenue_Per_Order', 'Revenue_Per_Customer', 'Revenue_Per_Salesperson']
        for col in money_cols:
            efficiency_summary[col] = efficiency_summary[col].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(efficiency_summary, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating operational metrics: {e}")

def create_customer_analysis_by_bu(df):
    """
    Create customer analysis by BU
    """
    st.subheader("👥 Customer Analysis by BU")
    
    try:
        # Customer distribution by BU
        bu_customers = df.groupby(['BU Name', 'Cust Class Code']).agg({
            'Cust Name': 'nunique',
            'Total Line Amount': 'sum'
        }).reset_index()
        
        # Customer class distribution
        fig = px.sunburst(
            bu_customers,
            path=['BU Name', 'Cust Class Code'],
            values='Total Line Amount',
            title='Revenue Distribution: BU → Customer Class'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographic distribution by BU
        st.subheader("🌍 Geographic Distribution by BU")
        
        bu_geography = df.groupby(['BU Name', 'Country']).agg({
            'Total Line Amount': 'sum',
            'Cust Name': 'nunique'
        }).reset_index()
        
        # Top countries by BU
        col1, col2 = st.columns(2)
        
        with col1:
            # Select BU for detailed analysis
            bu_options = df['BU Name'].unique()
            selected_bu = st.selectbox("Select BU for Geographic Analysis", bu_options)
            
            bu_geo_data = bu_geography[bu_geography['BU Name'] == selected_bu]
            bu_geo_data = bu_geo_data.sort_values('Total Line Amount', ascending=False).head(10)
            
            fig = px.bar(
                bu_geo_data,
                x='Country',
                y='Total Line Amount',
                title=f'{selected_bu} - Revenue by Country',
                color='Total Line Amount',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer concentration by BU
            bu_customer_metrics = df.groupby('BU Name').agg({
                'Cust Name': 'nunique',
                'Total Line Amount': 'sum'
            }).reset_index()
            
            bu_customer_metrics['Revenue_Per_Customer'] = (
                bu_customer_metrics['Total Line Amount'] / bu_customer_metrics['Cust Name']
            ).round(0)
            
            fig = px.scatter(
                bu_customer_metrics,
                x='Cust Name',
                y='Revenue_Per_Customer',
                size='Total Line Amount',
                hover_name='BU Name',
                title='Customer Base Size vs Revenue per Customer',
                color='BU Name'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top customers by BU
        st.subheader("🏆 Top Customers by BU")
        
        top_customers_by_bu = df.groupby(['BU Name', 'Cust Name'])['Total Line Amount'].sum().reset_index()
        
        # Get top 3 customers per BU
        top_customers = []
        for bu in df['BU Name'].unique():
            bu_customers = top_customers_by_bu[top_customers_by_bu['BU Name'] == bu]
            top_3 = bu_customers.nlargest(3, 'Total Line Amount')
            top_customers.append(top_3)
        
        top_customers_df = pd.concat(top_customers)
        
        fig = px.bar(
            top_customers_df,
            x='Total Line Amount',
            y='Cust Name',
            color='BU Name',
            title='Top 3 Customers per BU by Revenue',
            orientation='h',
            height=600
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating customer analysis by BU: {e}")

def create_bu_trend_analysis(df):
    """
    Create BU trend analysis
    """
    st.subheader("📈 BU Trend Analysis")
    
    try:
        # Monthly trends by BU
        bu_monthly = df.groupby(['YearMonth', 'BU Name'])['Total Line Amount'].sum().reset_index()
        bu_monthly['YearMonth'] = pd.to_datetime(bu_monthly['YearMonth'])
        
        # Overall trend
        fig = px.line(
            bu_monthly,
            x='YearMonth',
            y='Total Line Amount',
            color='BU Name',
            title='Monthly Revenue Trends by BU',
            markers=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth rate analysis
        st.subheader("📊 Growth Rate Analysis")
        
        # Calculate month-over-month growth
        bu_growth = bu_monthly.copy()
        bu_growth['Revenue_Growth'] = bu_growth.groupby('BU Name')['Total Line Amount'].pct_change() * 100
        bu_growth = bu_growth.dropna()
        
        # Average growth rate by BU
        avg_growth = bu_growth.groupby('BU Name')['Revenue_Growth'].mean().reset_index()
        avg_growth = avg_growth.sort_values('Revenue_Growth', ascending=True)
        
        fig = px.bar(
            avg_growth,
            x='Revenue_Growth',
            y='BU Name',
            orientation='h',
            title='Average Monthly Growth Rate by BU (%)',
            color='Revenue_Growth',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal analysis
        st.subheader("📅 Seasonal Analysis")
        
        # Add month name
        bu_monthly['Month'] = bu_monthly['YearMonth'].dt.month_name()
        
        # Monthly averages by BU
        seasonal_data = bu_monthly.groupby(['Month', 'BU Name'])['Total Line Amount'].mean().reset_index()
        
        # Month order
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        seasonal_data['Month'] = pd.Categorical(seasonal_data['Month'], categories=month_order, ordered=True)
        
        fig = px.line(
            seasonal_data.sort_values('Month'),
            x='Month',
            y='Total Line Amount',
            color='BU Name',
            title='Seasonal Patterns by BU (Average Monthly Revenue)',
            markers=True
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance consistency
        st.subheader("📏 Performance Consistency")
        
        # Calculate coefficient of variation for each BU
        bu_consistency = bu_monthly.groupby('BU Name')['Total Line Amount'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        bu_consistency['CV'] = (bu_consistency['std'] / bu_consistency['mean'] * 100).round(2)
        bu_consistency = bu_consistency[bu_consistency['count'] >= 3]  # At least 3 months
        bu_consistency = bu_consistency.sort_values('CV')
        
        fig = px.bar(
            bu_consistency,
            x='BU Name',
            y='CV',
            title='Performance Consistency by BU (Lower CV = More Consistent)',
            color='CV',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-year comparison (if data available)
        if 'Year' in df.columns and df['Year'].nunique() > 1:
            st.subheader("📆 Year-over-Year Comparison")
            
            yearly_bu = df.groupby(['Year', 'BU Name'])['Total Line Amount'].sum().reset_index()
            
            fig = px.bar(
                yearly_bu,
                x='BU Name',
                y='Total Line Amount',
                color='Year',
                title='Annual Revenue Comparison by BU',
                barmode='group'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating BU trend analysis: {e}")
