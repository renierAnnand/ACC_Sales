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
    """Load customer data from data_loader"""
    try:
        from modules.data_loader import load_and_merge_data
        return load_and_merge_data()
    except ImportError:
        st.error("Data loader not found. Please ensure data_loader.py is properly configured.")
        return pd.DataFrame()

def calculate_rfm_scores(df, reference_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) scores for each customer
    """
    if df.empty or 'Customer Name' not in df.columns:
        return pd.DataFrame()
    
    # Set reference date (latest date in data or today)
    if reference_date is None:
        if 'Invoice Date' in df.columns:
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
    """
    Segment customers based on RFM scores using business rules
    """
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

def perform_kmeans_clustering(rfm_df, n_clusters=5):
    """
    Perform K-means clustering on RFM data
    """
    if rfm_df.empty or len(rfm_df) < n_clusters:
        return rfm_df, None
    
    # Prepare features for clustering
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm_df[features].copy()
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    rfm_df_clustered = rfm_df.copy()
    rfm_df_clustered['Cluster'] = cluster_labels
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    # Create cluster summary
    cluster_summary = rfm_df_clustered.groupby('Cluster').agg({
        'Customer Name': 'count',
        'Recency': 'mean',
        'Frequency': 'mean', 
        'Monetary': 'mean',
        'Total Profit': 'mean',
        'Avg Profit Margin': 'mean'
    }).round(2)
    
    cluster_summary.columns = ['Customer Count', 'Avg Recency', 'Avg Frequency', 
                              'Avg Monetary', 'Avg Profit', 'Avg Profit Margin']
    
    return rfm_df_clustered, {'summary': cluster_summary, 'silhouette_score': silhouette_avg}

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

def create_segment_performance_chart(summary_df):
    """Create segment performance comparison chart"""
    
    if summary_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue by Segment', 'Customer Count by Segment',
                       'Average Order Value', 'Profit Margin by Segment'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    segments = summary_df.index
    
    # Revenue by segment
    fig.add_trace(
        go.Bar(x=segments, y=summary_df['Total Revenue'],
               name='Revenue', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Customer count by segment
    fig.add_trace(
        go.Bar(x=segments, y=summary_df['Customer Count'],
               name='Customers', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Average order value
    fig.add_trace(
        go.Bar(x=segments, y=summary_df['Avg Order Value'],
               name='AOV', marker_color='orange'),
        row=2, col=1
    )
    
    # Profit margin
    fig.add_trace(
        go.Bar(x=segments, y=summary_df['Avg Profit Margin'],
               name='Margin %', marker_color='red'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="📈 Segment Performance Analysis")
    fig.update_xaxes(tickangle=45)
    
    return fig

def generate_segment_insights(rfm_df, summary_df):
    """Generate actionable insights for each segment"""
    
    if rfm_df.empty or summary_df.empty:
        return "No data available for insights."
    
    insights = []
    
    # Overall insights
    total_customers = len(rfm_df)
    total_revenue = rfm_df['Monetary'].sum()
    
    insights.append(f"🎯 **Customer Portfolio Overview**: {total_customers:,} total customers generating ${total_revenue:,.0f} in revenue")
    
    # Top segment insights
    top_segment = summary_df.iloc[0]
    insights.append(f"👑 **Champions Segment**: {top_segment.name} represents {top_segment['Customer %']:.1f}% of customers but drives {top_segment['Revenue %']:.1f}% of revenue")
    
    # At-risk insights
    at_risk_segments = ['At Risk', 'About to Sleep', 'Cannot Lose Them']
    at_risk_customers = 0
    at_risk_revenue = 0
    
    for segment in at_risk_segments:
        if segment in summary_df.index:
            at_risk_customers += summary_df.loc[segment, 'Customer Count']
            at_risk_revenue += summary_df.loc[segment, 'Total Revenue']
    
    if at_risk_customers > 0:
        insights.append(f"⚠️ **Risk Alert**: {at_risk_customers} customers (${at_risk_revenue:,.0f} revenue) are at risk of churning")
    
    # New customer insights
    if 'New Customers' in summary_df.index:
        new_customers = summary_df.loc['New Customers', 'Customer Count']
        insights.append(f"🌟 **Growth Opportunity**: {new_customers} new customers with average order value of ${summary_df.loc['New Customers', 'Avg Order Value']:,.0f}")
    
    # Profitability insights
    most_profitable_segment = summary_df['Avg Profit Margin'].idxmax()
    profit_margin = summary_df.loc[most_profitable_segment, 'Avg Profit Margin']
    insights.append(f"💰 **Most Profitable**: {most_profitable_segment} segment has the highest profit margin at {profit_margin:.1f}%")
    
    return "\n\n".join(insights)

def get_segment_recommendations(segment):
    """Get specific recommendations for each customer segment"""
    
    recommendations = {
        'Champions': [
            "🎁 Reward with exclusive offers and early access to new products",
            "📞 Maintain regular communication and personal service",
            "🤝 Turn them into brand advocates and referral sources",
            "📊 Use them for product feedback and testimonials"
        ],
        'Loyal Customers': [
            "⬆️ Upsell and cross-sell premium products",
            "🎯 Recommend products based on purchase history", 
            "💎 Offer loyalty program benefits and rewards",
            "🔄 Increase purchase frequency with targeted campaigns"
        ],
        'Potential Loyalists': [
            "📧 Nurture with personalized email campaigns",
            "💰 Offer membership or loyalty program enrollment",
            "🎯 Provide targeted offers to increase purchase frequency",
            "📱 Engage through multiple channels"
        ],
        'New Customers': [
            "👋 Welcome with onboarding campaigns",
            "🎁 Provide starter discounts and incentives",
            "📚 Educate about product range and benefits",
            "🔄 Follow up after first purchase to encourage repeat buying"
        ],
        'At Risk': [
            "🚨 Launch win-back campaigns immediately",
            "📞 Personal outreach to understand issues",
            "💸 Offer special discounts or incentives",
            "🔍 Survey to identify pain points"
        ],
        'Cannot Lose Them': [
            "🆘 Urgent personal intervention required",
            "💰 Significant discount offers and exclusive deals",
            "👥 Executive-level relationship management",
            "🔧 Address any service issues immediately"
        ],
        'About to Sleep': [
            "⏰ Time-sensitive re-engagement campaigns",
            "📧 Personalized 'We miss you' messages",
            "🎯 Targeted offers based on past preferences",
            "📱 Multi-channel engagement strategy"
        ],
        'Need Attention': [
            "📊 Analyze purchase patterns for insights",
            "🎯 Targeted campaigns to increase engagement",
            "💰 Limited-time offers to drive action",
            "📞 Proactive customer service outreach"
        ]
    }
    
    return recommendations.get(segment, ["No specific recommendations available"])

def main():
    """Main function for customer segmentation"""
    
    st.title("👥 Customer Segmentation Analysis")
    st.markdown("---")
    
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
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
        years = ['All'] + sorted(df['Invoice Date'].dt.year.unique().tolist(), reverse=True)
        selected_year = st.sidebar.selectbox("Year", years)
        
        if selected_year != 'All':
            df = df[df['Invoice Date'].dt.year == selected_year]
    
    # Analysis options
    st.sidebar.header("⚙️ Analysis Options")
    
    use_clustering = st.sidebar.checkbox("Enable K-Means Clustering", value=False)
    
    if use_clustering:
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=3, max_value=8, value=5)
    
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
        
        # Segment performance chart
        fig_performance = create_segment_performance_chart(summary_df)
        if fig_performance:
            st.plotly_chart(fig_performance, use_container_width=True)
    
    # K-Means Clustering (Optional)
    if use_clustering:
        st.subheader("🤖 Machine Learning Clustering")
        
        with st.spinner("Performing K-means clustering..."):
            rfm_clustered, cluster_info = perform_kmeans_clustering(rfm_df, n_clusters)
        
        if cluster_info:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Cluster Summary:**")
                st.dataframe(cluster_info['summary'])
            
            with col2:
                st.metric("Silhouette Score", f"{cluster_info['silhouette_score']:.3f}")
                st.caption("Higher scores indicate better clustering (max: 1.0)")
            
            # Cluster visualization
            fig_cluster = px.scatter_3d(
                rfm_clustered,
                x='Recency',
                y='Frequency',
                z='Monetary',
                color='Cluster',
                title="K-Means Customer Clusters",
                hover_data=['Customer Name'],
                height=500
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Insights and Recommendations
    st.subheader("💡 Key Insights & Recommendations")
    
    insights = generate_segment_insights(rfm_df, summary_df)
    st.markdown(insights)
    
    # Detailed segment recommendations
    st.subheader("🎯 Segment-Specific Action Plans")
    
    # Create tabs for each major segment
    segments_with_customers = summary_df[summary_df['Customer Count'] > 0].index.tolist()
    
    if segments_with_customers:
        tabs = st.tabs(segments_with_customers[:6])  # Show top 6 segments
        
        for i, segment in enumerate(segments_with_customers[:6]):
            with tabs[i]:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    segment_data = summary_df.loc[segment]
                    st.metric("Customers", f"{int(segment_data['Customer Count']):,}")
                    st.metric("Total Revenue", f"${segment_data['Total Revenue']:,.0f}")
                    st.metric("Avg Order Value", f"${segment_data['Avg Order Value']:,.0f}")
                    st.metric("Profit Margin", f"{segment_data['Avg Profit Margin']:.1f}%")
                
                with col2:
                    st.write(f"**Action Plan for {segment}:**")
                    recommendations = get_segment_recommendations(segment)
                    for rec in recommendations:
                        st.write(f"• {rec}")
    
    # Customer List by Segment
    with st.expander("👥 Customer Details by Segment"):
        selected_segment = st.selectbox(
            "Select Segment to View Customers",
            options=segments_with_customers
        )
        
        if selected_segment:
            segment_customers = rfm_df[rfm_df['Segment'] == selected_segment].copy()
            segment_customers = segment_customers.sort_values('Monetary', ascending=False)
            
            # Display customer list
            display_columns = ['Customer Name', 'Recency', 'Frequency', 'Monetary', 
                             'Avg Order Value', 'Total Profit', 'Avg Profit Margin']
            
            st.dataframe(
                segment_customers[display_columns].style.format({
                    'Monetary': '${:,.0f}',
                    'Avg Order Value': '${:,.0f}',
                    'Total Profit': '${:,.0f}',
                    'Avg Profit Margin': '{:.1f}%'
                })
            )

if __name__ == "__main__":
    main()
