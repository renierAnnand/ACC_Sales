import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load customer data using the app's built-in data loading functions"""
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

def calculate_rfm_scores(df, reference_date=None):
    """Calculate RFM (Recency, Frequency, Monetary) scores for each customer"""
    if df.empty or 'Customer Name' not in df.columns:
        return pd.DataFrame()
    
    # Set reference date (latest date in data or today)
    if reference_date is None:
        if 'Invoice Date' in df.columns:
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date']).dt.tz_localize(None)
            reference_date = df['Invoice Date'].max()
        else:
            reference_date = datetime.now()
    
    # Calculate RFM metrics
    rfm_data = []
    
    for customer in df['Customer Name'].unique():
        customer_data = df[df['Customer Name'] == customer]
        
        # Recency: Days since last purchase
        if 'Invoice Date' in df.columns:
            last_purchase = customer_data['Invoice Date'].max()
            recency = (reference_date - last_purchase).days
        else:
            recency = 0  # Default if no date available
        
        # Frequency: Number of transactions
        frequency = len(customer_data)
        
        # Monetary: Total sales amount
        monetary = customer_data['Total Sales'].sum()
        
        # Additional metrics
        avg_order_value = customer_data['Total Sales'].mean()
        total_quantity = customer_data['Quantity'].sum() if 'Quantity' in customer_data.columns else 0
        
        # Profit metrics (if available)
        total_profit = 0
        avg_profit_margin = 0
        if 'Profit' in customer_data.columns:
            total_profit = customer_data['Profit'].sum()
            avg_profit_margin = (total_profit / monetary * 100) if monetary > 0 else 0
        
        rfm_data.append({
            'Customer Name': customer,
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'Avg Order Value': avg_order_value,
            'Total Quantity': total_quantity,
            'Total Profit': total_profit,
            'Avg Profit Margin': avg_profit_margin,
            'First Purchase': customer_data['Invoice Date'].min() if 'Invoice Date' in customer_data.columns else None,
            'Last Purchase': customer_data['Invoice Date'].max() if 'Invoice Date' in customer_data.columns else None
        })
    
    rfm_df = pd.DataFrame(rfm_data)
    
    # Calculate RFM scores (1-5 scale, 5 being the best)
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
    
    # Convert to numeric
    rfm_df['R_Score'] = pd.to_numeric(rfm_df['R_Score'])
    rfm_df['F_Score'] = pd.to_numeric(rfm_df['F_Score'])
    rfm_df['M_Score'] = pd.to_numeric(rfm_df['M_Score'])
    
    # Calculate RFM combined score
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
    rfm_df['RFM_Score_Numeric'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']
    
    return rfm_df

def segment_customers_rfm(rfm_df):
    """Segment customers based on RFM scores using business rules"""
    if rfm_df.empty:
        return rfm_df
    
    def get_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Champions: High value, frequent, recent customers
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        
        # Loyal Customers: High frequency and monetary, moderate recency
        elif f >= 4 and m >= 4:
            return 'Loyal Customers'
        
        # Potential Loyalists: Recent customers with good frequency
        elif r >= 4 and f >= 3:
            return 'Potential Loyalists'
        
        # New Customers: Very recent, low frequency
        elif r >= 4 and f <= 2:
            return 'New Customers'
        
        # Promising: Recent with moderate frequency and monetary
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Promising'
        
        # Need Attention: Moderate recency, frequency, and monetary
        elif r >= 2 and f >= 2 and m >= 2:
            return 'Need Attention'
        
        # About to Sleep: Low recency but had good frequency/monetary
        elif r <= 2 and f >= 3 and m >= 3:
            return 'About to Sleep'
        
        # At Risk: Low recency, was valuable
        elif r <= 2 and m >= 4:
            return 'At Risk'
        
        # Cannot Lose Them: Very low recency but high monetary
        elif r <= 1 and m >= 4:
            return 'Cannot Lose Them'
        
        # Hibernating: Low scores across the board but not lost
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        
        # Lost: Very low recency and frequency
        elif r <= 1 and f <= 1:
            return 'Lost'
        
        else:
            return 'Others'
    
    rfm_df['Segment'] = rfm_df.apply(get_segment, axis=1)
    
    # Add segment priority for action planning
    segment_priority = {
        'Champions': 1,
        'Loyal Customers': 2, 
        'Potential Loyalists': 3,
        'Cannot Lose Them': 4,
        'At Risk': 5,
        'About to Sleep': 6,
        'Need Attention': 7,
        'Promising': 8,
        'New Customers': 9,
        'Hibernating': 10,
        'Lost': 11,
        'Others': 12
    }
    
    rfm_df['Segment Priority'] = rfm_df['Segment'].map(segment_priority)
    
    return rfm_df

def create_segment_summary(rfm_df):
    """Create summary statistics for each customer segment"""
    if rfm_df.empty or 'Segment' not in rfm_df.columns:
        return pd.DataFrame()
    
    summary = rfm_df.groupby('Segment').agg({
        'Customer Name': 'count',
        'Monetary': ['sum', 'mean'],
        'Frequency': 'mean',
        'Recency': 'mean',
        'Total Profit': ['sum', 'mean'],
        'Avg Profit Margin': 'mean',
        'Avg Order Value': 'mean'
    }).round(2)
    
    # Flatten column names
    summary.columns = ['Customer Count', 'Total Revenue', 'Avg Revenue', 
                      'Avg Frequency', 'Avg Recency', 'Total Profit', 
                      'Avg Profit', 'Avg Profit Margin', 'Avg Order Value']
    
    # Calculate percentage of customers
    summary['Customer %'] = (summary['Customer Count'] / summary['Customer Count'].sum() * 100).round(1)
    
    # Calculate revenue percentage
    summary['Revenue %'] = (summary['Total Revenue'] / summary['Total Revenue'].sum() * 100).round(1)
    
    # Sort by segment priority
    segment_order = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'Cannot Lose Them',
                    'At Risk', 'About to Sleep', 'Need Attention', 'Promising', 
                    'New Customers', 'Hibernating', 'Lost', 'Others']
    
    summary = summary.reindex([seg for seg in segment_order if seg in summary.index])
    
    return summary

def create_rfm_visualizations(rfm_df):
    """Create RFM analysis visualizations"""
    
    if rfm_df.empty:
        return None, None, None
    
    # 1. RFM Score Distribution
    fig_scores = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Recency Score', 'Frequency Score', 'Monetary Score'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]]
    )
    
    fig_scores.add_trace(
        go.Histogram(x=rfm_df['R_Score'], nbinsx=5, name='Recency', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig_scores.add_trace(
        go.Histogram(x=rfm_df['F_Score'], nbinsx=5, name='Frequency', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig_scores.add_trace(
        go.Histogram(x=rfm_df['M_Score'], nbinsx=5, name='Monetary', marker_color='lightcoral'),
        row=1, col=3
    )
    
    fig_scores.update_layout(title_text="📊 RFM Score Distributions", showlegend=False, height=400)
    
    # 2. Customer Segments Pie Chart
    if 'Segment' in rfm_df.columns:
        segment_counts = rfm_df['Segment'].value_counts()
        
        fig_segments = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="🎯 Customer Segments Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_segments.update_traces(textposition='inside', textinfo='percent+label')
        fig_segments.update_layout(height=500)
    else:
        fig_segments = None
    
    # 3. RFM 3D Scatter Plot
    fig_3d = px.scatter_3d(
        rfm_df,
        x='Recency',
        y='Frequency', 
        z='Monetary',
        color='Segment' if 'Segment' in rfm_df.columns else 'RFM_Score_Numeric',
        title="🔄 3D RFM Analysis",
        hover_data=['Customer Name'],
        height=600
    )
    
    return fig_scores, fig_segments, fig_3d

def main():
    """Main function for customer segmentation"""
    
    st.title("👥 Customer Segmentation Analysis")
    st.markdown("---")
    
    # Show data source information
    show_data_source()
    
    # Load data
    with st.spinner("Loading customer data..."):
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
    
    # Year filter
    if 'Invoice Date' in df.columns:
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date']).dt.tz_localize(None)
        years = ['All'] + sorted(df['Invoice Date'].dt.year.unique().tolist(), reverse=True)
        selected_year = st.sidebar.selectbox("Year", years)
        
        if selected_year != 'All':
            df = df[df['Invoice Date'].dt.year == selected_year]
    
    # Display filtered data info
    st.sidebar.markdown(f"**Filtered Data**: {len(df):,} records")
    
    # Calculate RFM scores
    with st.spinner("Calculating RFM scores..."):
        rfm_df = calculate_rfm_scores(df)
    
    if rfm_df.empty:
        st.error("Unable to calculate RFM scores. Please check your data.")
        return
    
    # Segment customers
    with st.spinner("Segmenting customers..."):
        rfm_df = segment_customers_rfm(rfm_df)
        summary_df = create_segment_summary(rfm_df)
    
    # Key metrics
    st.subheader("📊 Customer Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(rfm_df):,}")
    
    with col2:
        st.metric("Total Revenue", f"${rfm_df['Monetary'].sum():,.0f}")
    
    with col3:
        avg_clv = rfm_df['Monetary'].mean()
        st.metric("Avg Customer Value", f"${avg_clv:,.0f}")
    
    with col4:
        segments_count = rfm_df['Segment'].nunique()
        st.metric("Customer Segments", f"{segments_count}")
    
    # RFM Analysis Results
    st.subheader("🎯 RFM Analysis Results")
    
    # Create visualizations
    fig_scores, fig_segments, fig_3d = create_rfm_visualizations(rfm_df)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if fig_scores:
            st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        if fig_segments:
            st.plotly_chart(fig_segments, use_container_width=True)
    
    # 3D Visualization
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Customer Segments Summary
    st.subheader("📋 Customer Segments Summary")
    
    if not summary_df.empty:
        # Display summary table
        st.dataframe(summary_df.style.format({
            'Total Revenue': '${:,.0f}',
            'Avg Revenue': '${:,.0f}',
            'Total Profit': '${:,.0f}',
            'Avg Profit': '${:,.0f}',
            'Avg Order Value': '${:,.0f}',
            'Avg Profit Margin': '{:.1f}%',
            'Customer %': '{:.1f}%',
            'Revenue %': '{:.1f}%'
        }))

if __name__ == "__main__":
    main()
