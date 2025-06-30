import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def create_performance_analysis(df):
    """
    Create comprehensive salesperson performance analysis
    """
    st.header("🏆 Salesperson Performance Analysis")
    st.markdown("*All amounts in Saudi Riyal (SAR)*")
    
    if df.empty:
        st.error("No data available for performance analysis")
        return
    
    # Performance overview
    create_performance_overview(df)
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Individual Performance", "Comparative Analysis", "Performance Trends", "Target Analysis"]
    )
    
    if analysis_type == "Individual Performance":
        create_individual_performance(df)
    elif analysis_type == "Comparative Analysis":
        create_comparative_analysis(df)
    elif analysis_type == "Performance Trends":
        create_performance_trends(df)
    elif analysis_type == "Target Analysis":
        create_target_analysis(df)

def create_performance_overview(df):
    """
    Create performance overview metrics
    """
    st.subheader("📊 Performance Overview")
    
    try:
        # Calculate salesperson metrics
        salesperson_metrics = df.groupby('Salesperson Name').agg({
            'Total Line Amount': ['sum', 'mean', 'count'],
            'Total Cost': 'sum',
            'QTY': 'sum',
            'Cust Name': 'nunique',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        # Flatten column names
        salesperson_metrics.columns = [
            'Salesperson', 'Total_Revenue', 'Avg_Deal_Size', 'Deal_Count',
            'Total_Cost', 'Total_Quantity', 'Unique_Customers', 'Unique_Orders'
        ]
        
        # Calculate additional metrics
        salesperson_metrics['Profit'] = salesperson_metrics['Total_Revenue'] - salesperson_metrics['Total_Cost']
        salesperson_metrics['Profit_Margin'] = (
            salesperson_metrics['Profit'] / salesperson_metrics['Total_Revenue'] * 100
        ).round(2)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            top_performer = salesperson_metrics.loc[salesperson_metrics['Total_Revenue'].idxmax()]
            st.metric(
                label="Top Performer (Revenue)",
                value=top_performer['Salesperson'],
                delta=f"{top_performer['Total_Revenue']:,.0f} SAR"
            )
        
        with col2:
            most_deals = salesperson_metrics.loc[salesperson_metrics['Deal_Count'].idxmax()]
            st.metric(
                label="Most Deals Closed",
                value=most_deals['Salesperson'],
                delta=f"{most_deals['Deal_Count']} deals"
            )
        
        with col3:
            best_margin = salesperson_metrics.loc[salesperson_metrics['Profit_Margin'].idxmax()]
            st.metric(
                label="Best Profit Margin",
                value=best_margin['Salesperson'],
                delta=f"{best_margin['Profit_Margin']:.1f}%"
            )
        
        with col4:
            most_customers = salesperson_metrics.loc[salesperson_metrics['Unique_Customers'].idxmax()]
            st.metric(
                label="Most Customers",
                value=most_customers['Salesperson'],
                delta=f"{most_customers['Unique_Customers']} customers"
            )
        
        # Performance ranking table
        st.subheader("🏅 Performance Rankings")
        
        # Sort by total revenue and add ranking
        performance_ranking = salesperson_metrics.sort_values('Total_Revenue', ascending=False).reset_index(drop=True)
        performance_ranking['Rank'] = range(1, len(performance_ranking) + 1)
        
        # Format for display
        display_df = performance_ranking[[
            'Rank', 'Salesperson', 'Total_Revenue', 'Deal_Count', 
            'Avg_Deal_Size', 'Profit_Margin', 'Unique_Customers'
        ]].copy()
        
        display_df['Total_Revenue'] = display_df['Total_Revenue'].apply(lambda x: f"{x:,.0f} SAR")
        display_df['Avg_Deal_Size'] = display_df['Avg_Deal_Size'].apply(lambda x: f"{x:,.0f} SAR")
        display_df['Profit_Margin'] = display_df['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating performance overview: {e}")

def create_individual_performance(df):
    """
    Create individual salesperson performance analysis
    """
    st.subheader("👤 Individual Performance Analysis")
    
    try:
        # Salesperson selection
        salespeople = df['Salesperson Name'].unique()
        selected_salesperson = st.selectbox("Select Salesperson", salespeople)
        
        # Filter data for selected salesperson
        salesperson_data = df[df['Salesperson Name'] == selected_salesperson]
        
        # Individual metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_revenue = salesperson_data['Total Line Amount'].sum()
            st.metric("Total Revenue", f"{total_revenue:,.0f} SAR")
            
            deal_count = salesperson_data['Invoice No.'].nunique()
            st.metric("Deals Closed", f"{deal_count}")
        
        with col2:
            avg_deal_size = salesperson_data['Total Line Amount'].mean()
            st.metric("Average Deal Size", f"{avg_deal_size:,.0f} SAR")
            
            customers = salesperson_data['Cust Name'].nunique()
            st.metric("Unique Customers", f"{customers}")
        
        with col3:
            total_cost = salesperson_data['Total Cost'].sum()
            profit = total_revenue - total_cost
            profit_margin = (profit / total_revenue * 100) if total_revenue > 0 else 0
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
            
            total_qty = salesperson_data['QTY'].sum()
            st.metric("Total Quantity", f"{total_qty:,.0f}")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly performance
            monthly_perf = salesperson_data.groupby('YearMonth')['Total Line Amount'].sum().reset_index()
            monthly_perf['YearMonth'] = pd.to_datetime(monthly_perf['YearMonth'])
            
            fig = px.line(
                monthly_perf,
                x='YearMonth',
                y='Total Line Amount',
                title=f'{selected_salesperson} - Monthly Performance',
                markers=True
            )
            fig.update_layout(height=400, yaxis_title="Revenue (SAR)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer distribution
            customer_revenue = salesperson_data.groupby('Cust Name')['Total Line Amount'].sum().reset_index()
            top_customers = customer_revenue.nlargest(5, 'Total Line Amount')
            
            fig = px.bar(
                top_customers,
                x='Total Line Amount',
                y='Cust Name',
                orientation='h',
                title=f'{selected_salesperson} - Top 5 Customers'
            )
            fig.update_layout(height=400, xaxis_title="Revenue (SAR)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Product performance
        st.subheader("📦 Product Performance")
        
        product_perf = salesperson_data.groupby('Brand')['Total Line Amount'].sum().reset_index()
        product_perf = product_perf.sort_values('Total Line Amount', ascending=False).head(8)
        
        fig = px.pie(
            product_perf,
            values='Total Line Amount',
            names='Brand',
            title=f'{selected_salesperson} - Revenue by Brand'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Deal size distribution
        st.subheader("💰 Deal Size Distribution")
        
        fig = px.histogram(
            salesperson_data,
            x='Total Line Amount',
            title=f'{selected_salesperson} - Deal Size Distribution',
            nbins=20
        )
        fig.update_layout(height=400, xaxis_title="Deal Size (SAR)")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating individual performance analysis: {e}")

def create_comparative_analysis(df):
    """
    Create comparative analysis between salespeople
    """
    st.subheader("📈 Comparative Analysis")
    
    try:
        # Multi-select for salespeople
        salespeople = df['Salesperson Name'].unique()
        selected_salespeople = st.multiselect(
            "Select Salespeople to Compare (max 5)",
            salespeople,
            default=salespeople[:3] if len(salespeople) >= 3 else salespeople
        )
        
        if len(selected_salespeople) == 0:
            st.warning("Please select at least one salesperson")
            return
        
        if len(selected_salespeople) > 5:
            st.warning("Please select maximum 5 salespeople for better visualization")
            selected_salespeople = selected_salespeople[:5]
        
        # Filter data
        comparison_data = df[df['Salesperson Name'].isin(selected_salespeople)]
        
        # Comparative metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue comparison
            revenue_comparison = comparison_data.groupby('Salesperson Name')['Total Line Amount'].sum().reset_index()
            
            fig = px.bar(
                revenue_comparison,
                x='Salesperson Name',
                y='Total Line Amount',
                title='Total Revenue Comparison',
                color='Total Line Amount',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45, yaxis_title="Revenue (SAR)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Deal count comparison
            deal_comparison = comparison_data.groupby('Salesperson Name')['Invoice No.'].nunique().reset_index()
            deal_comparison.columns = ['Salesperson Name', 'Deal_Count']
            
            fig = px.bar(
                deal_comparison,
                x='Salesperson Name',
                y='Deal_Count',
                title='Deal Count Comparison',
                color='Deal_Count',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance radar chart
        st.subheader("🎯 Performance Radar Chart")
        
        # Calculate normalized metrics for radar chart
        salesperson_metrics = comparison_data.groupby('Salesperson Name').agg({
            'Total Line Amount': 'sum',
            'Invoice No.': 'nunique',
            'Cust Name': 'nunique',
            'Total Line Amount': ['sum', 'mean']
        })
        
        # Flatten columns
        salesperson_metrics.columns = ['Total_Revenue', 'Deal_Count', 'Customer_Count', 'Avg_Deal_Size']
        
        # Normalize metrics (0-100 scale)
        for col in salesperson_metrics.columns:
            max_val = salesperson_metrics[col].max()
            salesperson_metrics[f'{col}_Normalized'] = (salesperson_metrics[col] / max_val * 100).round(1)
        
        # Create radar chart
        fig = go.Figure()
        
        for salesperson in selected_salespeople:
            if salesperson in salesperson_metrics.index:
                values = [
                    salesperson_metrics.loc[salesperson, 'Total_Revenue_Normalized'],
                    salesperson_metrics.loc[salesperson, 'Deal_Count_Normalized'],
                    salesperson_metrics.loc[salesperson, 'Customer_Count_Normalized'],
                    salesperson_metrics.loc[salesperson, 'Avg_Deal_Size_Normalized']
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=['Revenue', 'Deal Count', 'Customer Count', 'Avg Deal Size', 'Revenue'],
                    fill='toself',
                    name=salesperson
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Performance Comparison (Normalized Scores)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trend comparison
        st.subheader("📅 Monthly Trend Comparison")
        
        monthly_trends = comparison_data.groupby(['YearMonth', 'Salesperson Name'])['Total Line Amount'].sum().reset_index()
        monthly_trends['YearMonth'] = pd.to_datetime(monthly_trends['YearMonth'])
        
        fig = px.line(
            monthly_trends,
            x='YearMonth',
            y='Total Line Amount',
            color='Salesperson Name',
            title='Monthly Revenue Trends',
            markers=True
        )
        fig.update_layout(height=400, yaxis_title="Revenue (SAR)")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating comparative analysis: {e}")

def create_performance_trends(df):
    """
    Create performance trends analysis
    """
    st.subheader("📈 Performance Trends")
    
    try:
        # Overall team performance trend
        team_monthly = df.groupby('YearMonth').agg({
            'Total Line Amount': 'sum',
            'Invoice No.': 'nunique',
            'Salesperson Name': 'nunique'
        }).reset_index()
        
        team_monthly['YearMonth'] = pd.to_datetime(team_monthly['YearMonth'])
        team_monthly.columns = ['Month', 'Total_Revenue', 'Total_Deals', 'Active_Salespeople']
        
        # Calculate growth rates
        team_monthly['Revenue_Growth'] = team_monthly['Total_Revenue'].pct_change() * 100
        team_monthly['Deal_Growth'] = team_monthly['Total_Deals'].pct_change() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue trend
            fig = px.line(
                team_monthly,
                x='Month',
                y='Total_Revenue',
                title='Team Revenue Trend',
                markers=True
            )
            fig.update_layout(height=400, yaxis_title="Revenue (SAR)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Deal count trend
            fig = px.line(
                team_monthly,
                x='Month',
                y='Total_Deals',
                title='Team Deal Count Trend',
                markers=True,
                line_shape='spline'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Growth rates
        st.subheader("📊 Growth Rates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue growth
            fig = px.bar(
                team_monthly.dropna(),
                x='Month',
                y='Revenue_Growth',
                title='Monthly Revenue Growth Rate (%)',
                color='Revenue_Growth',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Deal growth
            fig = px.bar(
                team_monthly.dropna(),
                x='Month',
                y='Deal_Growth',
                title='Monthly Deal Growth Rate (%)',
                color='Deal_Growth',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance consistency
        st.subheader("📏 Performance Consistency")
        
        salesperson_monthly = df.groupby(['Salesperson Name', 'YearMonth'])['Total Line Amount'].sum().reset_index()
        consistency_metrics = salesperson_monthly.groupby('Salesperson Name')['Total Line Amount'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        consistency_metrics['CV'] = (consistency_metrics['std'] / consistency_metrics['mean'] * 100).round(2)
        consistency_metrics = consistency_metrics[consistency_metrics['count'] >= 3]  # At least 3 months
        consistency_metrics = consistency_metrics.sort_values('CV')
        
        consistency_metrics.columns = ['Salesperson', 'Average_Revenue', 'Std_Dev', 'Months_Active', 'Coefficient_of_Variation']
        
        fig = px.bar(
            consistency_metrics.head(10),
            x='Salesperson',
            y='Coefficient_of_Variation',
            title='Performance Consistency (Lower CV = More Consistent)',
            color='Coefficient_of_Variation',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating performance trends: {e}")

def create_target_analysis(df):
    """
    Create target analysis (simulated targets)
    """
    st.subheader("🎯 Target Analysis")
    
    try:
        st.info("Note: This analysis uses simulated targets based on historical performance")
        
        # Calculate salesperson metrics
        salesperson_metrics = df.groupby('Salesperson Name').agg({
            'Total Line Amount': 'sum',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        salesperson_metrics.columns = ['Salesperson', 'Actual_Revenue', 'Actual_Deals']
        
        # Simulate targets (e.g., 120% of average performance)
        avg_revenue = salesperson_metrics['Actual_Revenue'].mean()
        avg_deals = salesperson_metrics['Actual_Deals'].mean()
        
        salesperson_metrics['Target_Revenue'] = avg_revenue * 1.2
        salesperson_metrics['Target_Deals'] = avg_deals * 1.2
        
        # Calculate achievement percentages
        salesperson_metrics['Revenue_Achievement'] = (
            salesperson_metrics['Actual_Revenue'] / salesperson_metrics['Target_Revenue'] * 100
        ).round(1)
        
        salesperson_metrics['Deal_Achievement'] = (
            salesperson_metrics['Actual_Deals'] / salesperson_metrics['Target_Deals'] * 100
        ).round(1)
        
        # Target achievement visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue achievement
            fig = px.bar(
                salesperson_metrics.sort_values('Revenue_Achievement', ascending=False),
                x='Salesperson',
                y='Revenue_Achievement',
                title='Revenue Target Achievement (%)',
                color='Revenue_Achievement',
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Target")
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Deal achievement
            fig = px.bar(
                salesperson_metrics.sort_values('Deal_Achievement', ascending=False),
                x='Salesperson',
                y='Deal_Achievement',
                title='Deal Count Target Achievement (%)',
                color='Deal_Achievement',
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Target")
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Achievement summary
        st.subheader("📋 Achievement Summary")
        
        # Format for display
        display_metrics = salesperson_metrics.copy()
        display_metrics['Actual_Revenue'] = display_metrics['Actual_Revenue'].apply(lambda x: f"{x:,.0f} SAR")
        display_metrics['Target_Revenue'] = display_metrics['Target_Revenue'].apply(lambda x: f"{x:,.0f} SAR")
        display_metrics['Revenue_Achievement'] = display_metrics['Revenue_Achievement'].apply(lambda x: f"{x:.1f}%")
        display_metrics['Deal_Achievement'] = display_metrics['Deal_Achievement'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_metrics, use_container_width=True)
        
        # Performance categories
        st.subheader("🏅 Performance Categories")
        
        def categorize_performance(row):
            if row['Revenue_Achievement'] >= 100 and row['Deal_Achievement'] >= 100:
                return 'Exceeds Both Targets'
            elif row['Revenue_Achievement'] >= 100:
                return 'Exceeds Revenue Target'
            elif row['Deal_Achievement'] >= 100:
                return 'Exceeds Deal Target'
            else:
                return 'Below Targets'
        
        # Convert back to numeric for categorization
        numeric_metrics = df.groupby('Salesperson Name').agg({
            'Total Line Amount': 'sum',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        numeric_metrics.columns = ['Salesperson', 'Actual_Revenue', 'Actual_Deals']
        numeric_metrics['Target_Revenue'] = avg_revenue * 1.2
        numeric_metrics['Target_Deals'] = avg_deals * 1.2
        numeric_metrics['Revenue_Achievement'] = (
            numeric_metrics['Actual_Revenue'] / numeric_metrics['Target_Revenue'] * 100
        )
        numeric_metrics['Deal_Achievement'] = (
            numeric_metrics['Actual_Deals'] / numeric_metrics['Target_Deals'] * 100
        )
        
        numeric_metrics['Performance_Category'] = numeric_metrics.apply(categorize_performance, axis=1)
        
        category_counts = numeric_metrics['Performance_Category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Performance Category Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating target analysis: {e}")
