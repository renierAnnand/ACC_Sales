import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

def create_product_insights(df):
    """
    Create comprehensive product insights and analysis
    """
    st.header("📦 Product Insights & Analysis")
    
    if df.empty:
        st.error("No data available for product analysis")
        return
    
    # Product overview
    create_product_overview(df)
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Product Analysis",
        ["Brand Performance", "Product Categories", "Product Profitability", "Sales Volume Analysis", "Product Lifecycle"]
    )
    
    if analysis_type == "Brand Performance":
        create_brand_performance(df)
    elif analysis_type == "Product Categories":
        create_product_categories(df)
    elif analysis_type == "Product Profitability":
        create_product_profitability(df)
    elif analysis_type == "Sales Volume Analysis":
        create_volume_analysis(df)
    elif analysis_type == "Product Lifecycle":
        create_product_lifecycle(df)

def create_product_overview(df):
    """
    Create product overview metrics
    """
    st.subheader("📊 Product Portfolio Overview")
    
    try:
        # Calculate product metrics
        total_products = df['Item Number'].nunique()
        total_brands = df['Brand'].nunique()
        total_categories = df['Item Category'].nunique() if 'Item Category' in df.columns else 0
        
        # Top performing metrics
        top_product = df.groupby('Item Number')['Total Line Amount'].sum().idxmax()
        top_product_revenue = df.groupby('Item Number')['Total Line Amount'].sum().max()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", f"{total_products:,}")
        
        with col2:
            st.metric("Total Brands", f"{total_brands:,}")
        
        with col3:
            st.metric("Product Categories", f"{total_categories:,}")
        
        with col4:
            st.metric("Top Product Revenue", f"${top_product_revenue:,.0f}")
        
        # Product performance distribution
        st.subheader("📈 Product Performance Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by brand
            brand_revenue = df.groupby('Brand')['Total Line Amount'].sum().reset_index()
            brand_revenue = brand_revenue.sort_values('Total Line Amount', ascending=False).head(10)
            
            fig = px.pie(
                brand_revenue,
                values='Total Line Amount',
                names='Brand',
                title='Top 10 Brands by Revenue'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Product concentration
            product_revenue = df.groupby('Item Number')['Total Line Amount'].sum().reset_index()
            
            # Calculate Pareto analysis
            product_revenue = product_revenue.sort_values('Total Line Amount', ascending=False)
            product_revenue['Cumulative_Revenue'] = product_revenue['Total Line Amount'].cumsum()
            product_revenue['Cumulative_Percent'] = (
                product_revenue['Cumulative_Revenue'] / product_revenue['Total Line Amount'].sum() * 100
            )
            product_revenue['Product_Rank'] = range(1, len(product_revenue) + 1)
            product_revenue['Product_Percent'] = (
                product_revenue['Product_Rank'] / len(product_revenue) * 100
            )
            
            # 80/20 analysis
            pareto_80 = product_revenue[product_revenue['Cumulative_Percent'] <= 80]
            pareto_products = len(pareto_80)
            pareto_percent = (pareto_products / total_products) * 100
            
            fig = px.line(
                product_revenue.head(100),  # Top 100 for visibility
                x='Product_Percent',
                y='Cumulative_Percent',
                title='Product Revenue Concentration (Pareto Analysis)',
                labels={
                    'Product_Percent': '% of Products',
                    'Cumulative_Percent': '% of Revenue'
                }
            )
            
            # Add 80/20 line
            fig.add_hline(y=80, line_dash="dash", line_color="red", 
                         annotation_text=f"80% Revenue from {pareto_percent:.1f}% Products")
            fig.add_vline(x=20, line_dash="dash", line_color="red")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Quick insights
        st.info(f"💡 **Key Insight**: {pareto_percent:.1f}% of products ({pareto_products} products) generate 80% of revenue")
        
    except Exception as e:
        st.error(f"Error creating product overview: {e}")

def create_brand_performance(df):
    """
    Create brand performance analysis
    """
    st.subheader("🏷️ Brand Performance Analysis")
    
    try:
        # Calculate brand metrics
        brand_metrics = df.groupby('Brand').agg({
            'Total Line Amount': ['sum', 'mean', 'count'],
            'Total Cost': 'sum',
            'QTY': 'sum',
            'Cust Name': 'nunique',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        # Flatten columns
        brand_metrics.columns = [
            'Brand', 'Total_Revenue', 'Avg_Deal_Size', 'Deal_Count',
            'Total_Cost', 'Total_Quantity', 'Customer_Count', 'Order_Count'
        ]
        
        # Calculate additional metrics
        brand_metrics['Profit'] = brand_metrics['Total_Revenue'] - brand_metrics['Total_Cost']
        brand_metrics['Profit_Margin'] = (
            brand_metrics['Profit'] / brand_metrics['Total_Revenue'] * 100
        ).round(2)
        brand_metrics['Revenue_Per_Customer'] = (
            brand_metrics['Total_Revenue'] / brand_metrics['Customer_Count']
        ).round(0)
        brand_metrics['Units_Per_Order'] = (
            brand_metrics['Total_Quantity'] / brand_metrics['Order_Count']
        ).round(2)
        
        # Remove null/NA brands
        brand_metrics = brand_metrics[brand_metrics['Brand'].notna()]
        brand_metrics = brand_metrics[brand_metrics['Brand'] != 'NA']
        
        # Top brands analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue ranking
            top_brands = brand_metrics.nlargest(15, 'Total_Revenue')
            
            fig = px.bar(
                top_brands,
                x='Total_Revenue',
                y='Brand',
                orientation='h',
                title='Top 15 Brands by Revenue',
                color='Total_Revenue',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Profit margin ranking
            top_margin_brands = brand_metrics.nlargest(15, 'Profit_Margin')
            
            fig = px.bar(
                top_margin_brands,
                x='Profit_Margin',
                y='Brand',
                orientation='h',
                title='Top 15 Brands by Profit Margin',
                color='Profit_Margin',
                color_continuous_scale='greens'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Brand portfolio matrix
        st.subheader("📊 Brand Portfolio Matrix")
        
        # Create portfolio matrix based on revenue and growth
        # Since we don't have historical data for growth, we'll use profit margin as proxy
        fig = px.scatter(
            brand_metrics,
            x='Total_Revenue',
            y='Profit_Margin',
            size='Total_Quantity',
            color='Customer_Count',
            hover_name='Brand',
            title='Brand Portfolio Matrix: Revenue vs Profit Margin',
            labels={
                'Total_Revenue': 'Total Revenue ($)',
                'Profit_Margin': 'Profit Margin (%)'
            },
            color_continuous_scale='viridis'
        )
        
        # Add quadrant lines
        median_revenue = brand_metrics['Total_Revenue'].median()
        median_margin = brand_metrics['Profit_Margin'].median()
        
        fig.add_vline(x=median_revenue, line_dash="dash", line_color="gray", 
                     annotation_text="Median Revenue")
        fig.add_hline(y=median_margin, line_dash="dash", line_color="gray", 
                     annotation_text="Median Margin")
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Brand performance table
        st.subheader("📋 Brand Performance Summary")
        
        # Format for display
        display_brands = brand_metrics.nlargest(20, 'Total_Revenue').copy()
        display_brands['Total_Revenue'] = display_brands['Total_Revenue'].apply(lambda x: f"${x:,.0f}")
        display_brands['Avg_Deal_Size'] = display_brands['Avg_Deal_Size'].apply(lambda x: f"${x:,.0f}")
        display_brands['Profit'] = display_brands['Profit'].apply(lambda x: f"${x:,.0f}")
        display_brands['Revenue_Per_Customer'] = display_brands['Revenue_Per_Customer'].apply(lambda x: f"${x:,.0f}")
        display_brands['Profit_Margin'] = display_brands['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_brands[[
            'Brand', 'Total_Revenue', 'Profit', 'Profit_Margin', 
            'Deal_Count', 'Customer_Count', 'Revenue_Per_Customer'
        ]], use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating brand performance analysis: {e}")

def create_product_categories(df):
    """
    Create product category analysis
    """
    st.subheader("📂 Product Category Analysis")
    
    try:
        # Check if Item Category exists
        if 'Item Category' not in df.columns or df['Item Category'].isna().all():
            st.warning("Item Category data not available. Using Brand as category proxy.")
            category_col = 'Brand'
        else:
            category_col = 'Item Category'
        
        # Calculate category metrics
        category_metrics = df.groupby(category_col).agg({
            'Total Line Amount': ['sum', 'mean'],
            'Total Cost': 'sum',
            'QTY': 'sum',
            'Item Number': 'nunique',
            'Cust Name': 'nunique'
        }).reset_index()
        
        # Flatten columns
        category_metrics.columns = [
            'Category', 'Total_Revenue', 'Avg_Revenue', 'Total_Cost', 
            'Total_Quantity', 'Product_Count', 'Customer_Count'
        ]
        
        # Remove null categories
        category_metrics = category_metrics[category_metrics['Category'].notna()]
        if category_col == 'Brand':
            category_metrics = category_metrics[category_metrics['Category'] != 'NA']
        
        # Calculate additional metrics
        category_metrics['Profit'] = category_metrics['Total_Revenue'] - category_metrics['Total_Cost']
        category_metrics['Profit_Margin'] = (
            category_metrics['Profit'] / category_metrics['Total_Revenue'] * 100
        ).round(2)
        category_metrics['Revenue_Per_Product'] = (
            category_metrics['Total_Revenue'] / category_metrics['Product_Count']
        ).round(0)
        
        # Category performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by category
            top_categories = category_metrics.nlargest(10, 'Total_Revenue')
            
            fig = px.treemap(
                top_categories,
                values='Total_Revenue',
                names='Category',
                title='Revenue by Category (Treemap)',
                color='Profit_Margin',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category diversity
            fig = px.scatter(
                category_metrics,
                x='Product_Count',
                y='Total_Revenue',
                size='Customer_Count',
                color='Profit_Margin',
                hover_name='Category',
                title='Category Diversity: Products vs Revenue',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Category performance trends
        st.subheader("📈 Category Performance Trends")
        
        # Monthly trends by category
        category_monthly = df.groupby(['YearMonth', category_col])['Total Line Amount'].sum().reset_index()
        category_monthly['YearMonth'] = pd.to_datetime(category_monthly['YearMonth'])
        category_monthly.columns = ['YearMonth', 'Category', 'Revenue']
        
        # Top 5 categories for trend analysis
        top_5_categories = category_metrics.nlargest(5, 'Total_Revenue')['Category'].tolist()
        trend_data = category_monthly[category_monthly['Category'].isin(top_5_categories)]
        
        fig = px.line(
            trend_data,
            x='YearMonth',
            y='Revenue',
            color='Category',
            title='Top 5 Categories - Monthly Revenue Trends',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category efficiency analysis
        st.subheader("⚡ Category Efficiency Analysis")
        
        fig = px.bar(
            category_metrics.nlargest(12, 'Revenue_Per_Product'),
            x='Category',
            y='Revenue_Per_Product',
            title='Revenue per Product by Category',
            color='Revenue_Per_Product',
            color_continuous_scale='plasma'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating product category analysis: {e}")

def create_product_profitability(df):
    """
    Create product profitability analysis
    """
    st.subheader("💰 Product Profitability Analysis")
    
    try:
        # Calculate product profitability
        product_profit = df.groupby(['Item Number', 'Brand']).agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'QTY': 'sum',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        # Calculate profit metrics
        product_profit['Profit'] = product_profit['Total Line Amount'] - product_profit['Total Cost']
        product_profit['Profit_Margin'] = (
            product_profit['Profit'] / product_profit['Total Line Amount'] * 100
        ).round(2)
        product_profit['Profit_Per_Unit'] = (
            product_profit['Profit'] / product_profit['QTY']
        ).round(2)
        
        # Remove products with no cost data
        product_profit = product_profit[product_profit['Total_Cost'] > 0]
        
        # Profitability distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit margin distribution
            fig = px.histogram(
                product_profit,
                x='Profit_Margin',
                title='Product Profit Margin Distribution',
                nbins=30,
                labels={'Profit_Margin': 'Profit Margin (%)'}
            )
            fig.add_vline(x=product_profit['Profit_Margin'].mean(), 
                         line_dash="dash", line_color="red", 
                         annotation_text="Average")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue vs Profit scatter
            sample_products = product_profit.sample(min(500, len(product_profit)))
            
            fig = px.scatter(
                sample_products,
                x='Total Line Amount',
                y='Profit',
                color='Profit_Margin',
                size='QTY',
                hover_data=['Brand'],
                title='Revenue vs Profit Analysis',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top and bottom performers
        st.subheader("🏆 Top & Bottom Performers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Most profitable products
            st.write("**Most Profitable Products**")
            top_profit = product_profit.nlargest(10, 'Profit')[
                ['Item Number', 'Brand', 'Profit', 'Profit_Margin', 'Total Line Amount']
            ].copy()
            
            top_profit['Profit'] = top_profit['Profit'].apply(lambda x: f"${x:,.0f}")
            top_profit['Total Line Amount'] = top_profit['Total Line Amount'].apply(lambda x: f"${x:,.0f}")
            top_profit['Profit_Margin'] = top_profit['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(top_profit, use_container_width=True)
        
        with col2:
            # Least profitable products
            st.write("**Least Profitable Products**")
            bottom_profit = product_profit.nsmallest(10, 'Profit')[
                ['Item Number', 'Brand', 'Profit', 'Profit_Margin', 'Total Line Amount']
            ].copy()
            
            bottom_profit['Profit'] = bottom_profit['Profit'].apply(lambda x: f"${x:,.0f}")
            bottom_profit['Total Line Amount'] = bottom_profit['Total Line Amount'].apply(lambda x: f"${x:,.0f}")
            bottom_profit['Profit_Margin'] = bottom_profit['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(bottom_profit, use_container_width=True)
        
        # Profitability by brand
        st.subheader("🏷️ Profitability by Brand")
        
        brand_profitability = product_profit.groupby('Brand').agg({
            'Profit': 'sum',
            'Total Line Amount': 'sum',
            'Item Number': 'nunique'
        }).reset_index()
        
        brand_profitability['Profit_Margin'] = (
            brand_profitability['Profit'] / brand_profitability['Total Line Amount'] * 100
        ).round(2)
        
        # Remove null/NA brands
        brand_profitability = brand_profitability[brand_profitability['Brand'].notna()]
        brand_profitability = brand_profitability[brand_profitability['Brand'] != 'NA']
        
        top_profit_brands = brand_profitability.nlargest(15, 'Profit')
        
        fig = px.bar(
            top_profit_brands,
            x='Brand',
            y='Profit',
            title='Top 15 Brands by Total Profit',
            color='Profit_Margin',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Profit improvement opportunities
        st.subheader("💡 Profit Improvement Opportunities")
        
        # High revenue, low margin products
        improvement_opportunities = product_profit[
            (product_profit['Total Line Amount'] > product_profit['Total Line Amount'].quantile(0.75)) &
            (product_profit['Profit_Margin'] < product_profit['Profit_Margin'].quantile(0.25))
        ]
        
        if not improvement_opportunities.empty:
            st.warning("🔍 **High Revenue, Low Margin Products** - Consider cost optimization or pricing review:")
            
            opportunities = improvement_opportunities.nlargest(5, 'Total Line Amount')[
                ['Item Number', 'Brand', 'Total Line Amount', 'Profit_Margin', 'QTY']
            ].copy()
            
            opportunities['Total Line Amount'] = opportunities['Total Line Amount'].apply(lambda x: f"${x:,.0f}")
            opportunities['Profit_Margin'] = opportunities['Profit_Margin'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(opportunities, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating product profitability analysis: {e}")

def create_volume_analysis(df):
    """
    Create sales volume analysis
    """
    st.subheader("📊 Sales Volume Analysis")
    
    try:
        # Volume metrics by product
        volume_analysis = df.groupby(['Item Number', 'Brand']).agg({
            'QTY': 'sum',
            'Total Line Amount': 'sum',
            'Invoice No.': 'nunique',
            'Cust Name': 'nunique'
        }).reset_index()
        
        volume_analysis['Revenue_Per_Unit'] = (
            volume_analysis['Total Line Amount'] / volume_analysis['QTY']
        ).round(2)
        volume_analysis['Units_Per_Order'] = (
            volume_analysis['QTY'] / volume_analysis['Invoice No.']
        ).round(2)
        
        # Volume distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Quantity distribution
            fig = px.histogram(
                volume_analysis,
                x='QTY',
                title='Product Quantity Distribution',
                nbins=30,
                labels={'QTY': 'Total Quantity Sold'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price vs Volume
            sample_volume = volume_analysis.sample(min(500, len(volume_analysis)))
            
            fig = px.scatter(
                sample_volume,
                x='QTY',
                y='Revenue_Per_Unit',
                color='Brand',
                size='Total Line Amount',
                title='Volume vs Unit Price Analysis',
                labels={
                    'QTY': 'Total Quantity',
                    'Revenue_Per_Unit': 'Revenue per Unit ($)'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # High volume products
        st.subheader("📦 High Volume Products")
        
        top_volume = volume_analysis.nlargest(15, 'QTY')
        
        fig = px.bar(
            top_volume,
            x='QTY',
            y='Brand',
            orientation='h',
            title='Top 15 Products by Volume',
            color='Revenue_Per_Unit',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume trends
        st.subheader("📈 Volume Trends")
        
        # Monthly volume trends
        monthly_volume = df.groupby('YearMonth').agg({
            'QTY': 'sum',
            'Total Line Amount': 'sum'
        }).reset_index()
        
        monthly_volume['YearMonth'] = pd.to_datetime(monthly_volume['YearMonth'])
        monthly_volume['Avg_Unit_Price'] = monthly_volume['Total Line Amount'] / monthly_volume['QTY']
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Volume Trend', 'Average Unit Price Trend'),
            vertical_spacing=0.1
        )
        
        # Volume trend
        fig.add_trace(
            go.Scatter(
                x=monthly_volume['YearMonth'],
                y=monthly_volume['QTY'],
                mode='lines+markers',
                name='Volume',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Price trend
        fig.add_trace(
            go.Scatter(
                x=monthly_volume['YearMonth'],
                y=monthly_volume['Avg_Unit_Price'],
                mode='lines+markers',
                name='Avg Unit Price',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Quantity", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Brand volume analysis
        st.subheader("🏷️ Volume by Brand")
        
        brand_volume = df.groupby('Brand').agg({
            'QTY': 'sum',
            'Total Line Amount': 'sum',
            'Item Number': 'nunique'
        }).reset_index()
        
        brand_volume['Avg_Unit_Price'] = brand_volume['Total Line Amount'] / brand_volume['QTY']
        
        # Remove null/NA brands
        brand_volume = brand_volume[brand_volume['Brand'].notna()]
        brand_volume = brand_volume[brand_volume['Brand'] != 'NA']
        
        top_volume_brands = brand_volume.nlargest(12, 'QTY')
        
        fig = px.bar(
            top_volume_brands,
            x='Brand',
            y='QTY',
            title='Top 12 Brands by Volume',
            color='Avg_Unit_Price',
            color_continuous_scale='plasma'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating volume analysis: {e}")

def create_product_lifecycle(df):
    """
    Create product lifecycle analysis
    """
    st.subheader("🔄 Product Lifecycle Analysis")
    
    try:
        # Product lifecycle metrics
        product_lifecycle = df.groupby(['Item Number', 'Brand']).agg({
            'Invoice Date': ['min', 'max', 'count'],
            'Total Line Amount': 'sum',
            'QTY': 'sum'
        }).reset_index()
        
        # Flatten columns
        product_lifecycle.columns = [
            'Item_Number', 'Brand', 'First_Sale', 'Last_Sale', 'Sale_Count',
            'Total_Revenue', 'Total_Quantity'
        ]
        
        # Calculate lifecycle metrics
        product_lifecycle['Days_Active'] = (
            product_lifecycle['Last_Sale'] - product_lifecycle['First_Sale']
        ).dt.days + 1
        
        product_lifecycle['Sales_Frequency'] = (
            product_lifecycle['Sale_Count'] / product_lifecycle['Days_Active'] * 365
        ).fillna(0).round(2)
        
        product_lifecycle['Days_Since_Last_Sale'] = (
            pd.Timestamp.now() - product_lifecycle['Last_Sale']
        ).dt.days
        
        # Lifecycle stage classification
        def classify_lifecycle_stage(row):
            if row['Days_Since_Last_Sale'] > 365:
                return 'Discontinued'
            elif row['Days_Since_Last_Sale'] > 180:
                return 'Declining'
            elif row['Days_Active'] < 90:
                return 'New/Introduction'
            elif row['Sales_Frequency'] > product_lifecycle['Sales_Frequency'].quantile(0.75):
                return 'Growth'
            else:
                return 'Mature'
        
        product_lifecycle['Lifecycle_Stage'] = product_lifecycle.apply(classify_lifecycle_stage, axis=1)
        
        # Lifecycle distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Stage distribution
            stage_dist = product_lifecycle['Lifecycle_Stage'].value_counts()
            
            fig = px.pie(
                values=stage_dist.values,
                names=stage_dist.index,
                title='Product Lifecycle Stage Distribution'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue by stage
            stage_revenue = product_lifecycle.groupby('Lifecycle_Stage')['Total_Revenue'].sum().reset_index()
            
            fig = px.bar(
                stage_revenue,
                x='Lifecycle_Stage',
                y='Total_Revenue',
                title='Revenue by Lifecycle Stage',
                color='Total_Revenue',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Product age analysis
        st.subheader("📅 Product Age Analysis")
        
        fig = px.scatter(
            product_lifecycle.sample(min(500, len(product_lifecycle))),
            x='Days_Active',
            y='Total_Revenue',
            color='Lifecycle_Stage',
            size='Sale_Count',
            hover_data=['Brand'],
            title='Product Age vs Revenue',
            labels={'Days_Active': 'Days in Market'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Stage-specific insights
        st.subheader("💡 Lifecycle Stage Insights")
        
        # New products
        new_products = product_lifecycle[product_lifecycle['Lifecycle_Stage'] == 'New/Introduction']
        if not new_products.empty:
            st.success(f"🆕 **New Products**: {len(new_products)} products in introduction stage")
            top_new = new_products.nlargest(5, 'Total_Revenue')[['Brand', 'Total_Revenue', 'Days_Active']]
            st.dataframe(top_new, use_container_width=True)
        
        # Declining products
        declining_products = product_lifecycle[product_lifecycle['Lifecycle_Stage'] == 'Declining']
        if not declining_products.empty:
            st.warning(f"📉 **Declining Products**: {len(declining_products)} products showing decline")
            top_declining = declining_products.nlargest(5, 'Total_Revenue')[
                ['Brand', 'Total_Revenue', 'Days_Since_Last_Sale']
            ]
            st.dataframe(top_declining, use_container_width=True)
        
        # Discontinued products
        discontinued_products = product_lifecycle[product_lifecycle['Lifecycle_Stage'] == 'Discontinued']
        if not discontinued_products.empty:
            st.error(f"🚫 **Discontinued Products**: {len(discontinued_products)} products not sold in >1 year")
            
        # Growth products
        growth_products = product_lifecycle[product_lifecycle['Lifecycle_Stage'] == 'Growth']
        if not growth_products.empty:
            st.success(f"🚀 **Growth Products**: {len(growth_products)} products in growth stage")
            top_growth = growth_products.nlargest(5, 'Sales_Frequency')[
                ['Brand', 'Total_Revenue', 'Sales_Frequency']
            ]
            st.dataframe(top_growth, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating product lifecycle analysis: {e}")
