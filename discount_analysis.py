import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

def create_discount_analysis(df):
    """
    Create comprehensive discount and pricing analysis
    """
    st.header("💰 Discount & Pricing Analysis")
    
    if df.empty:
        st.error("No data available for discount analysis")
        return
    
    # Prepare pricing data
    pricing_data = prepare_pricing_data(df)
    
    if pricing_data.empty:
        st.error("Unable to calculate pricing metrics")
        return
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Profit Margin Analysis", "Discount Impact Analysis", "Pricing Strategy", "Customer Pricing Patterns"]
    )
    
    if analysis_type == "Profit Margin Analysis":
        create_profit_margin_analysis(pricing_data)
    elif analysis_type == "Discount Impact Analysis":
        create_discount_impact_analysis(pricing_data)
    elif analysis_type == "Pricing Strategy":
        create_pricing_strategy_analysis(pricing_data)
    elif analysis_type == "Customer Pricing Patterns":
        create_customer_pricing_patterns(pricing_data)

def prepare_pricing_data(df):
    """
    Prepare data with pricing and margin calculations
    """
    try:
        # Create copy to avoid modifying original
        pricing_df = df.copy()
        
        # Clean numeric columns
        numeric_cols = ['Total Line Amount', 'Item Cost', 'Total Cost', 'QTY', 'Tax Amount']
        for col in numeric_cols:
            if col in pricing_df.columns:
                pricing_df[col] = pd.to_numeric(pricing_df[col], errors='coerce').fillna(0)
        
        # Calculate key pricing metrics
        pricing_df['Unit_Price'] = np.where(
            pricing_df['QTY'] > 0,
            pricing_df['Total Line Amount'] / pricing_df['QTY'],
            pricing_df['Total Line Amount']
        )
        
        pricing_df['Unit_Cost'] = np.where(
            pricing_df['QTY'] > 0,
            pricing_df['Total Cost'] / pricing_df['QTY'],
            pricing_df['Total Cost']
        )
        
        # Calculate profit metrics
        pricing_df['Gross_Profit'] = pricing_df['Total Line Amount'] - pricing_df['Total Cost']
        
        pricing_df['Profit_Margin_Percent'] = np.where(
            pricing_df['Total Line Amount'] > 0,
            (pricing_df['Gross_Profit'] / pricing_df['Total Line Amount']) * 100,
            0
        )
        
        pricing_df['Markup_Percent'] = np.where(
            pricing_df['Total Cost'] > 0,
            ((pricing_df['Total Line Amount'] - pricing_df['Total Cost']) / pricing_df['Total Cost']) * 100,
            0
        )
        
        # Calculate implied discount (assuming standard markup)
        # This is an approximation since we don't have list prices
        pricing_df['Revenue_Per_Unit'] = pricing_df['Unit_Price']
        
        # Remove outliers for better analysis
        pricing_df = pricing_df[
            (pricing_df['Profit_Margin_Percent'] >= -100) & 
            (pricing_df['Profit_Margin_Percent'] <= 100)
        ]
        
        return pricing_df
        
    except Exception as e:
        st.error(f"Error preparing pricing data: {e}")
        return pd.DataFrame()

def create_profit_margin_analysis(df):
    """
    Create profit margin analysis
    """
    st.subheader("📊 Profit Margin Analysis")
    
    try:
        # Overall margin metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_margin = df['Profit_Margin_Percent'].mean()
            st.metric("Average Profit Margin", f"{avg_margin:.1f}%")
        
        with col2:
            median_margin = df['Profit_Margin_Percent'].median()
            st.metric("Median Profit Margin", f"{median_margin:.1f}%")
        
        with col3:
            total_revenue = df['Total Line Amount'].sum()
            total_cost = df['Total Cost'].sum()
            overall_margin = ((total_revenue - total_cost) / total_revenue) * 100 if total_revenue > 0 else 0
            st.metric("Overall Profit Margin", f"{overall_margin:.1f}%")
        
        with col4:
            high_margin_deals = len(df[df['Profit_Margin_Percent'] > 30])
            total_deals = len(df)
            high_margin_percent = (high_margin_deals / total_deals) * 100 if total_deals > 0 else 0
            st.metric("High Margin Deals (>30%)", f"{high_margin_percent:.1f}%")
        
        # Margin distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit margin histogram
            fig = px.histogram(
                df,
                x='Profit_Margin_Percent',
                nbins=30,
                title='Profit Margin Distribution',
                labels={'Profit_Margin_Percent': 'Profit Margin (%)'}
            )
            fig.add_vline(x=df['Profit_Margin_Percent'].mean(), line_dash="dash", 
                         line_color="red", annotation_text="Average")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Margin by deal size
            df['Deal_Size_Category'] = pd.cut(
                df['Total Line Amount'],
                bins=[-np.inf, 10000, 50000, 100000, 500000, np.inf],
                labels=['<$10K', '$10K-$50K', '$50K-$100K', '$100K-$500K', '>$500K']
            )
            
            margin_by_size = df.groupby('Deal_Size_Category')['Profit_Margin_Percent'].mean().reset_index()
            
            fig = px.bar(
                margin_by_size,
                x='Deal_Size_Category',
                y='Profit_Margin_Percent',
                title='Average Profit Margin by Deal Size',
                color='Profit_Margin_Percent',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Margin by business unit
        st.subheader("🏢 Margin by Business Unit")
        
        bu_margins = df.groupby('BU Name').agg({
            'Profit_Margin_Percent': 'mean',
            'Total Line Amount': 'sum',
            'Gross_Profit': 'sum'
        }).reset_index()
        
        bu_margins = bu_margins.sort_values('Profit_Margin_Percent', ascending=False)
        
        fig = px.bar(
            bu_margins,
            x='BU Name',
            y='Profit_Margin_Percent',
            title='Average Profit Margin by Business Unit',
            color='Profit_Margin_Percent',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Margin trends over time
        st.subheader("📈 Margin Trends Over Time")
        
        monthly_margins = df.groupby('YearMonth')['Profit_Margin_Percent'].mean().reset_index()
        monthly_margins['YearMonth'] = pd.to_datetime(monthly_margins['YearMonth'])
        
        fig = px.line(
            monthly_margins,
            x='YearMonth',
            y='Profit_Margin_Percent',
            title='Monthly Average Profit Margin Trend',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating profit margin analysis: {e}")

def create_discount_impact_analysis(df):
    """
    Create discount impact analysis
    """
    st.subheader("📉 Discount Impact Analysis")
    
    try:
        # Since we don't have explicit discount data, we'll analyze margin variations
        # as a proxy for discount impact
        
        st.info("Note: Discount analysis is based on profit margin variations as explicit discount data is not available")
        
        # Analyze margin distribution by customer class (proxy for discount levels)
        margin_by_class = df.groupby('Cust Class Code').agg({
            'Profit_Margin_Percent': ['mean', 'std', 'count'],
            'Total Line Amount': 'sum'
        }).round(2)
        
        margin_by_class.columns = ['Avg_Margin', 'Margin_StdDev', 'Deal_Count', 'Total_Revenue']
        margin_by_class = margin_by_class.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average margin by customer class
            fig = px.bar(
                margin_by_class,
                x='Cust Class Code',
                y='Avg_Margin',
                title='Average Margin by Customer Class',
                color='Avg_Margin',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue vs margin scatter
            fig = px.scatter(
                df.sample(min(1000, len(df))),  # Sample for performance
                x='Total Line Amount',
                y='Profit_Margin_Percent',
                color='Cust Class Code',
                title='Revenue vs Profit Margin by Customer Class',
                hover_data=['Cust Name']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Margin variance analysis (proxy for discount patterns)
        st.subheader("📊 Margin Variance Analysis")
        
        # Calculate margin statistics by salesperson
        salesperson_margins = df.groupby('Salesperson Name').agg({
            'Profit_Margin_Percent': ['mean', 'std', 'min', 'max'],
            'Total Line Amount': 'sum'
        }).round(2)
        
        salesperson_margins.columns = ['Avg_Margin', 'Margin_StdDev', 'Min_Margin', 'Max_Margin', 'Total_Revenue']
        salesperson_margins = salesperson_margins.reset_index()
        salesperson_margins['Margin_Range'] = salesperson_margins['Max_Margin'] - salesperson_margins['Min_Margin']
        
        # Top salespeople by revenue
        top_salespeople = salesperson_margins.nlargest(10, 'Total_Revenue')
        
        fig = px.scatter(
            top_salespeople,
            x='Avg_Margin',
            y='Margin_StdDev',
            size='Total_Revenue',
            hover_name='Salesperson Name',
            title='Margin Consistency vs Average Margin (Top 10 Salespeople)',
            labels={
                'Avg_Margin': 'Average Margin (%)',
                'Margin_StdDev': 'Margin Standard Deviation'
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Low margin deals analysis
        st.subheader("⚠️ Low Margin Deals Analysis")
        
        low_margin_threshold = st.slider("Low Margin Threshold (%)", 0, 30, 10)
        low_margin_deals = df[df['Profit_Margin_Percent'] < low_margin_threshold]
        
        if not low_margin_deals.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                low_margin_count = len(low_margin_deals)
                total_deals = len(df)
                low_margin_percent = (low_margin_count / total_deals) * 100
                st.metric("Low Margin Deals", f"{low_margin_count}", f"{low_margin_percent:.1f}% of total")
            
            with col2:
                low_margin_revenue = low_margin_deals['Total Line Amount'].sum()
                total_revenue = df['Total Line Amount'].sum()
                low_margin_revenue_percent = (low_margin_revenue / total_revenue) * 100
                st.metric("Low Margin Revenue", f"${low_margin_revenue:,.0f}", f"{low_margin_revenue_percent:.1f}% of total")
            
            with col3:
                avg_low_margin = low_margin_deals['Profit_Margin_Percent'].mean()
                st.metric("Avg Low Margin", f"{avg_low_margin:.1f}%")
            
            # Low margin deals by customer
            low_margin_customers = low_margin_deals.groupby('Cust Name').agg({
                'Total Line Amount': 'sum',
                'Profit_Margin_Percent': 'mean'
            }).reset_index()
            
            low_margin_customers = low_margin_customers.nlargest(10, 'Total Line Amount')
            
            fig = px.bar(
                low_margin_customers,
                x='Total Line Amount',
                y='Cust Name',
                orientation='h',
                title=f'Top 10 Customers with Low Margin Deals (<{low_margin_threshold}%)',
                color='Profit_Margin_Percent',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info(f"No deals found with margin below {low_margin_threshold}%")
        
    except Exception as e:
        st.error(f"Error creating discount impact analysis: {e}")

def create_pricing_strategy_analysis(df):
    """
    Create pricing strategy analysis
    """
    st.subheader("🎯 Pricing Strategy Analysis")
    
    try:
        # Price-volume analysis
        st.subheader("📈 Price-Volume Analysis")
        
        # Create price buckets
        df['Price_Bucket'] = pd.cut(
            df['Unit_Price'],
            bins=10,
            labels=[f'Bucket {i+1}' for i in range(10)]
        )
        
        price_volume = df.groupby('Price_Bucket').agg({
            'QTY': 'sum',
            'Total Line Amount': 'sum',
            'Unit_Price': 'mean',
            'Profit_Margin_Percent': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price vs Volume
            fig = px.scatter(
                price_volume,
                x='Unit_Price',
                y='QTY',
                size='Total Line Amount',
                title='Price vs Volume Analysis',
                labels={
                    'Unit_Price': 'Average Unit Price ($)',
                    'QTY': 'Total Quantity Sold'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price vs Margin
            fig = px.scatter(
                price_volume,
                x='Unit_Price',
                y='Profit_Margin_Percent',
                size='Total Line Amount',
                title='Price vs Profit Margin Analysis',
                labels={
                    'Unit_Price': 'Average Unit Price ($)',
                    'Profit_Margin_Percent': 'Profit Margin (%)'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Brand pricing analysis
        st.subheader("🏷️ Brand Pricing Analysis")
        
        brand_pricing = df.groupby('Brand').agg({
            'Unit_Price': 'mean',
            'Profit_Margin_Percent': 'mean',
            'Total Line Amount': 'sum',
            'QTY': 'sum'
        }).reset_index()
        
        # Remove null/NA brands
        brand_pricing = brand_pricing[brand_pricing['Brand'].notna()]
        brand_pricing = brand_pricing[brand_pricing['Brand'] != 'NA']
        brand_pricing = brand_pricing.nlargest(15, 'Total Line Amount')
        
        fig = px.scatter(
            brand_pricing,
            x='Unit_Price',
            y='Profit_Margin_Percent',
            size='Total Line Amount',
            color='Brand',
            title='Brand Positioning: Price vs Margin',
            labels={
                'Unit_Price': 'Average Unit Price ($)',
                'Profit_Margin_Percent': 'Profit Margin (%)'
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Pricing recommendations
        st.subheader("💡 Pricing Insights & Recommendations")
        
        # High-value opportunities
        high_volume_low_margin = df[
            (df['QTY'] > df['QTY'].quantile(0.75)) & 
            (df['Profit_Margin_Percent'] < df['Profit_Margin_Percent'].quantile(0.25))
        ]
        
        if not high_volume_low_margin.empty:
            st.warning("🔍 **High Volume, Low Margin Opportunities**")
            st.write("These products/customers have high volume but low margins - consider price optimization:")
            
            opportunities = high_volume_low_margin.groupby(['Brand', 'Cust Class Code']).agg({
                'Total Line Amount': 'sum',
                'QTY': 'sum',
                'Profit_Margin_Percent': 'mean'
            }).reset_index().nlargest(5, 'Total Line Amount')
            
            st.dataframe(opportunities, use_container_width=True)
        
        # Premium pricing opportunities
        high_margin_products = df[df['Profit_Margin_Percent'] > df['Profit_Margin_Percent'].quantile(0.9)]
        
        if not high_margin_products.empty:
            st.success("✨ **Premium Pricing Success Stories**")
            st.write("These products/customers demonstrate successful premium pricing:")
            
            premium_examples = high_margin_products.groupby(['Brand', 'Cust Class Code']).agg({
                'Total Line Amount': 'sum',
                'Profit_Margin_Percent': 'mean',
                'Unit_Price': 'mean'
            }).reset_index().nlargest(5, 'Profit_Margin_Percent')
            
            st.dataframe(premium_examples, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating pricing strategy analysis: {e}")

def create_customer_pricing_patterns(df):
    """
    Create customer pricing patterns analysis
    """
    st.subheader("👥 Customer Pricing Patterns")
    
    try:
        # Customer pricing summary
        customer_pricing = df.groupby(['Cust Name', 'Cust Class Code']).agg({
            'Total Line Amount': ['sum', 'mean'],
            'Profit_Margin_Percent': 'mean',
            'Unit_Price': 'mean',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        customer_pricing.columns = [
            'Customer', 'Customer_Class', 'Total_Revenue', 'Avg_Deal_Size',
            'Avg_Margin', 'Avg_Unit_Price', 'Deal_Count'
        ]
        
        # Customer value vs margin analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Top customers by revenue with their margins
            top_customers = customer_pricing.nlargest(15, 'Total_Revenue')
            
            fig = px.scatter(
                top_customers,
                x='Total_Revenue',
                y='Avg_Margin',
                size='Deal_Count',
                color='Customer_Class',
                hover_name='Customer',
                title='Top Customers: Revenue vs Margin',
                labels={
                    'Total_Revenue': 'Total Revenue ($)',
                    'Avg_Margin': 'Average Margin (%)'
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer class pricing patterns
            class_patterns = df.groupby('Cust Class Code').agg({
                'Unit_Price': 'mean',
                'Profit_Margin_Percent': 'mean',
                'Total Line Amount': 'sum'
            }).reset_index()
            
            fig = px.bar(
                class_patterns,
                x='Cust Class Code',
                y='Profit_Margin_Percent',
                title='Average Margin by Customer Class',
                color='Profit_Margin_Percent',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer loyalty vs pricing
        st.subheader("🤝 Customer Loyalty vs Pricing")
        
        # Define loyalty based on number of orders and time span
        customer_loyalty = df.groupby('Cust Name').agg({
            'Invoice Date': ['min', 'max', 'count'],
            'Total Line Amount': 'sum',
            'Profit_Margin_Percent': 'mean'
        }).reset_index()
        
        customer_loyalty.columns = [
            'Customer', 'First_Order', 'Last_Order', 'Order_Count',
            'Total_Revenue', 'Avg_Margin'
        ]
        
        customer_loyalty['Customer_Lifetime_Days'] = (
            customer_loyalty['Last_Order'] - customer_loyalty['First_Order']
        ).dt.days + 1
        
        customer_loyalty['Loyalty_Score'] = (
            customer_loyalty['Order_Count'] * 0.4 + 
            (customer_loyalty['Customer_Lifetime_Days'] / 365) * 0.6
        ).round(2)
        
        # Loyalty vs margin analysis
        loyalty_sample = customer_loyalty.sample(min(500, len(customer_loyalty)))
        
        fig = px.scatter(
            loyalty_sample,
            x='Loyalty_Score',
            y='Avg_Margin',
            size='Total_Revenue',
            hover_name='Customer',
            title='Customer Loyalty vs Average Margin',
            labels={
                'Loyalty_Score': 'Customer Loyalty Score',
                'Avg_Margin': 'Average Margin (%)'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographic pricing patterns
        st.subheader("🌍 Geographic Pricing Patterns")
        
        geographic_pricing = df.groupby('Country').agg({
            'Unit_Price': 'mean',
            'Profit_Margin_Percent': 'mean',
            'Total Line Amount': 'sum',
            'Cust Name': 'nunique'
        }).reset_index()
        
        geographic_pricing = geographic_pricing.sort_values('Total_Revenue', ascending=False).head(10)
        
        fig = px.bar(
            geographic_pricing,
            x='Country',
            y='Profit_Margin_Percent',
            title='Average Profit Margin by Country',
            color='Profit_Margin_Percent',
            color_continuous_scale='plasma'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating customer pricing patterns: {e}")
