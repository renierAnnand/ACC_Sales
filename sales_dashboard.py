import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data using the app's built-in data loading functions"""
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

def create_kpi_cards(df, col1, col2, col3, col4):
    """Create KPI cards for key metrics"""
    if df.empty:
        return
    
    # Calculate KPIs
    total_revenue = df['Total Sales'].sum() if 'Total Sales' in df.columns else 0
    total_cost = df['Total Cost'].sum() if 'Total Cost' in df.columns else 0
    total_profit = total_revenue - total_cost
    profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    
    # Display KPIs
    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta=f"{len(df)} transactions"
        )
    
    with col2:
        st.metric(
            label="📈 Total Profit",
            value=f"${total_profit:,.0f}",
            delta=f"{profit_margin:.1f}% margin"
        )
    
    with col3:
        avg_order_value = df['Total Sales'].mean() if 'Total Sales' in df.columns else 0
        customer_count = df['Customer Name'].nunique() if 'Customer Name' in df.columns else 0
        st.metric(
            label="🛒 Avg Order Value",
            value=f"${avg_order_value:,.0f}",
            delta=f"{customer_count} customers"
        )
    
    with col4:
        unique_products = df['Item Description'].nunique() if 'Item Description' in df.columns else 0
        unique_bus = df['Business Unit'].nunique() if 'Business Unit' in df.columns else 0
        st.metric(
            label="📦 Products Sold",
            value=f"{unique_products:,}",
            delta=f"{unique_bus} business units"
        )

def create_revenue_trend_chart(df):
    """Create revenue trend over time"""
    if df.empty or 'Invoice Date' not in df.columns:
        st.warning("Invoice Date column not found for trend analysis")
        return None
    
    # Prepare data for trend analysis - fix timezone issue
    df_copy = df.copy()
    df_copy['Invoice Date'] = pd.to_datetime(df_copy['Invoice Date']).dt.tz_localize(None)
    
    monthly_revenue = df_copy.groupby(df_copy['Invoice Date'].dt.to_period('M')).agg({
        'Total Sales': 'sum',
        'Total Cost': 'sum' if 'Total Cost' in df_copy.columns else 'count'
    }).reset_index()
    
    if 'Total Cost' in df_copy.columns:
        monthly_revenue['Profit'] = monthly_revenue['Total Sales'] - monthly_revenue['Total Cost']
    
    monthly_revenue['Invoice Date'] = monthly_revenue['Invoice Date'].dt.to_timestamp()
    
    # Create the chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Revenue line
    fig.add_trace(
        go.Scatter(
            x=monthly_revenue['Invoice Date'],
            y=monthly_revenue['Total Sales'],
            name="Revenue",
            line=dict(color='#1f77b4', width=3),
            mode='lines+markers'
        ),
        secondary_y=False,
    )
    
    # Profit line (if available)
    if 'Total Cost' in df_copy.columns:
        fig.add_trace(
            go.Scatter(
                x=monthly_revenue['Invoice Date'],
                y=monthly_revenue['Profit'],
                name="Profit",
                line=dict(color='#2ca02c', width=3),
                mode='lines+markers'
            ),
            secondary_y=True,
        )
    
    # Update layout
    fig.update_layout(
        title="📈 Revenue & Profit Trend Over Time",
        xaxis_title="Month",
        height=400,
        hovermode='x unified'
    )
    
    fig.update_yaxis(title_text="Revenue ($)", secondary_y=False)
    if 'Total Cost' in df_copy.columns:
        fig.update_yaxis(title_text="Profit ($)", secondary_y=True)
    
    return fig

def create_bu_performance_chart(df):
    """Create business unit performance chart"""
    if df.empty or 'Business Unit' not in df.columns:
        return None
    
    bu_performance = df.groupby('Business Unit').agg({
        'Total Sales': 'sum',
        'Total Cost': 'sum' if 'Total Cost' in df.columns else 'count',
        'Customer Name': 'nunique' if 'Customer Name' in df.columns else 'count'
    }).reset_index()
    
    if 'Total Cost' in df.columns:
        bu_performance['Profit'] = bu_performance['Total Sales'] - bu_performance['Total Cost']
        bu_performance['Profit Margin'] = (bu_performance['Profit'] / bu_performance['Total Sales'] * 100)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue by Business Unit', 'Customer Count by BU', 
                       'Profit Margin by BU' if 'Total Cost' in df.columns else 'Transactions by BU',
                       'Revenue Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Revenue by BU
    fig.add_trace(
        go.Bar(x=bu_performance['Business Unit'], y=bu_performance['Total Sales'],
               name='Revenue', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Customer count by BU
    fig.add_trace(
        go.Bar(x=bu_performance['Business Unit'], y=bu_performance['Customer Name'],
               name='Customers', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Profit margin or transaction count
    if 'Total Cost' in df.columns:
        fig.add_trace(
            go.Bar(x=bu_performance['Business Unit'], y=bu_performance['Profit Margin'],
                   name='Profit Margin %', marker_color='orange'),
            row=2, col=1
        )
    else:
        transaction_count = df.groupby('Business Unit').size().reset_index(name='Transactions')
        fig.add_trace(
            go.Bar(x=transaction_count['Business Unit'], y=transaction_count['Transactions'],
                   name='Transactions', marker_color='purple'),
            row=2, col=1
        )
    
    # Revenue distribution pie chart
    fig.add_trace(
        go.Pie(labels=bu_performance['Business Unit'], values=bu_performance['Total Sales'],
               name="Revenue Share"),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="🏢 Business Unit Performance Analysis")
    return fig

def create_top_performers_charts(df):
    """Create charts for top customers and products"""
    if df.empty:
        return None, None
    
    # Top 10 customers by revenue
    fig_customers = None
    if 'Customer Name' in df.columns:
        top_customers = df.groupby('Customer Name')['Total Sales'].sum().sort_values(ascending=False).head(10)
        
        fig_customers = px.bar(
            x=top_customers.values,
            y=top_customers.index,
            orientation='h',
            title="🥇 Top 10 Customers by Revenue",
            labels={'x': 'Revenue ($)', 'y': 'Customer'},
            color=top_customers.values,
            color_continuous_scale='Blues'
        )
        fig_customers.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    
    # Top 10 products by revenue (if available)
    fig_products = None
    if 'Item Description' in df.columns:
        top_products = df.groupby('Item Description')['Total Sales'].sum().sort_values(ascending=False).head(10)
        
        fig_products = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title="🏆 Top 10 Products by Revenue",
            labels={'x': 'Revenue ($)', 'y': 'Product'},
            color=top_products.values,
            color_continuous_scale='Greens'
        )
        fig_products.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    
    return fig_customers, fig_products

def create_geographic_analysis(df):
    """Create geographic performance analysis"""
    if df.empty or 'Country' not in df.columns:
        return None
    
    geo_performance = df.groupby('Country').agg({
        'Total Sales': 'sum',
        'Customer Name': 'nunique' if 'Customer Name' in df.columns else 'count'
    }).reset_index().sort_values('Total Sales', ascending=False)
    
    fig = px.bar(
        geo_performance.head(10),
        x='Country',
        y='Total Sales',
        title="🌍 Revenue by Country (Top 10)",
        labels={'Total Sales': 'Revenue ($)'},
        color='Total Sales',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

def generate_insights(df):
    """Generate natural language insights"""
    if df.empty:
        return "No data available for insights."
    
    insights = []
    
    # Revenue insights
    total_revenue = df['Total Sales'].sum() if 'Total Sales' in df.columns else 0
    insights.append(f"💰 **Total Revenue**: ${total_revenue:,.0f} across {len(df):,} transactions")
    
    # Top customer insight
    if 'Customer Name' in df.columns:
        top_customer = df.groupby('Customer Name')['Total Sales'].sum().idxmax()
        top_customer_revenue = df.groupby('Customer Name')['Total Sales'].sum().max()
        customer_percentage = (top_customer_revenue / total_revenue) * 100 if total_revenue > 0 else 0
        insights.append(f"🥇 **Top Customer**: {top_customer} contributed ${top_customer_revenue:,.0f} ({customer_percentage:.1f}% of total revenue)")
    
    # Business unit insights
    if 'Business Unit' in df.columns:
        top_bu = df.groupby('Business Unit')['Total Sales'].sum().idxmax()
        bu_revenue = df.groupby('Business Unit')['Total Sales'].sum().max()
        bu_percentage = (bu_revenue / total_revenue) * 100 if total_revenue > 0 else 0
        insights.append(f"🏢 **Leading Business Unit**: {top_bu} generated ${bu_revenue:,.0f} ({bu_percentage:.1f}% of total revenue)")
    
    # Profit insight (if available)
    if 'Total Cost' in df.columns:
        total_cost = df['Total Cost'].sum()
        total_profit = total_revenue - total_cost
        profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
        insights.append(f"📈 **Profitability**: ${total_profit:,.0f} profit with {profit_margin:.1f}% margin")
    
    # Average order value
    if 'Total Sales' in df.columns:
        avg_order = df['Total Sales'].mean()
        insights.append(f"🛒 **Average Order Value**: ${avg_order:,.0f}")
    
    # Product diversity (if available)
    if 'Item Description' in df.columns:
        unique_products = df['Item Description'].nunique()
        insights.append(f"📦 **Product Portfolio**: {unique_products:,} unique products sold")
    
    return "\n\n".join(insights)

def main():
    """Main function for the sales dashboard"""
    st.title("📊 Sales Intelligence Dashboard")
    st.markdown("---")
    
    # Show data source information
    show_data_source()
    
    # Load data
    with st.spinner("Loading sales data..."):
        df = load_data()
    
    if df.empty:
        st.error("No data available. Please check your data source.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Business Unit filter
    if 'Business Unit' in df.columns:
        business_units = ['All'] + sorted(df['Business Unit'].unique().tolist())
        selected_bu = st.sidebar.selectbox("Business Unit", business_units)
        
        if selected_bu != 'All':
            df = df[df['Business Unit'] == selected_bu]
    
    # Year filter (if date column exists)
    if 'Invoice Date' in df.columns:
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date']).dt.tz_localize(None)
        years = ['All'] + sorted(df['Invoice Date'].dt.year.unique().tolist(), reverse=True)
        selected_year = st.sidebar.selectbox("Year", years)
        
        if selected_year != 'All':
            df = df[df['Invoice Date'].dt.year == selected_year]
    
    # Country filter (if available)
    if 'Country' in df.columns:
        countries = ['All'] + sorted(df['Country'].unique().tolist())
        selected_country = st.sidebar.selectbox("Country", countries)
        
        if selected_country != 'All':
            df = df[df['Country'] == selected_country]
    
    # Display filtered data info
    st.sidebar.markdown(f"**Filtered Data**: {len(df):,} records")
    
    # KPI Cards
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    create_kpi_cards(df, col1, col2, col3, col4)
    
    st.markdown("---")
    
    # Revenue Trend
    st.subheader("📊 Performance Trends")
    if 'Invoice Date' in df.columns:
        fig_trend = create_revenue_trend_chart(df)
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Invoice Date not available for trend analysis")
    
    # Business Unit Performance
    st.subheader("🏢 Business Unit Analysis")
    fig_bu = create_bu_performance_chart(df)
    if fig_bu:
        st.plotly_chart(fig_bu, use_container_width=True)
    
    # Top Performers
    st.subheader("🏆 Top Performers")
    col1, col2 = st.columns(2)
    
    fig_customers, fig_products = create_top_performers_charts(df)
    
    with col1:
        if fig_customers:
            st.plotly_chart(fig_customers, use_container_width=True)
        else:
            st.info("Customer information not available")
    
    with col2:
        if fig_products:
            st.plotly_chart(fig_products, use_container_width=True)
        else:
            st.info("Product information not available")
    
    # Geographic Analysis
    fig_geo = create_geographic_analysis(df)
    if fig_geo:
        st.subheader("🌍 Geographic Performance")
        st.plotly_chart(fig_geo, use_container_width=True)
    
    # Insights Section
    st.markdown("---")
    st.subheader("💡 Key Insights")
    insights = generate_insights(df)
    st.markdown(insights)
    
    # Data Summary
    with st.expander("📋 Data Summary"):
        st.dataframe(df.describe())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Shape:**", df.shape)
            if 'Invoice Date' in df.columns:
                st.write("**Date Range:**", 
                        f"{df['Invoice Date'].min().strftime('%Y-%m-%d')} to {df['Invoice Date'].max().strftime('%Y-%m-%d')}")
            else:
                st.write("**Date Range:**", "Date not available")
        
        with col2:
            if 'Business Unit' in df.columns:
                st.write("**Business Units:**", df['Business Unit'].nunique())
            else:
                st.write("**Business Units:**", "N/A")
            if 'Customer Name' in df.columns:
                st.write("**Unique Customers:**", df['Customer Name'].nunique())
            else:
                st.write("**Unique Customers:**", "N/A")

if __name__ == "__main__":
    main()
