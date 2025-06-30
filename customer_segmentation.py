import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def create_customer_segmentation(df):
    """
    Create comprehensive customer segmentation analysis
    """
    st.header("👥 Customer Segmentation Analysis")
    st.markdown("*All amounts in Saudi Riyal (SAR)*")
    
    if df.empty:
        st.error("No data available for customer segmentation")
        return
    
    # Create customer summary
    customer_summary = create_customer_summary(df)
    
    if customer_summary.empty:
        st.error("Unable to create customer summary")
        return
    
    # Segmentation options
    segmentation_type = st.selectbox(
        "Select Segmentation Method",
        ["RFM Analysis", "Value-Based Segmentation", "Geographic Segmentation", "Behavioral Segmentation"]
    )
    
    if segmentation_type == "RFM Analysis":
        create_rfm_analysis(df, customer_summary)
    elif segmentation_type == "Value-Based Segmentation":
        create_value_based_segmentation(customer_summary)
    elif segmentation_type == "Geographic Segmentation":
        create_geographic_segmentation(df)
    elif segmentation_type == "Behavioral Segmentation":
        create_behavioral_segmentation(df, customer_summary)

def create_customer_summary(df):
    """
    Create customer summary with key metrics
    """
    try:
        # Calculate customer metrics
        customer_metrics = df.groupby(['Cust No.', 'Cust Name', 'Country', 'Local', 'Cust Class Code']).agg({
            'Total Line Amount': ['sum', 'mean', 'count'],
            'Invoice Date': ['min', 'max'],
            'QTY': 'sum',
            'Profit': 'sum' if 'Profit' in df.columns else 'count'
        }).reset_index()
        
        # Flatten column names
        customer_metrics.columns = [
            'Customer_No', 'Customer_Name', 'Country', 'Local_International', 'Customer_Class',
            'Total_Revenue', 'Avg_Order_Value', 'Order_Count', 'First_Purchase', 'Last_Purchase',
            'Total_Quantity', 'Total_Profit'
        ]
        
        # Calculate additional metrics
        customer_metrics['Days_Since_Last_Purchase'] = (
            pd.Timestamp.now() - customer_metrics['Last_Purchase']
        ).dt.days
        
        customer_metrics['Customer_Lifetime_Days'] = (
            customer_metrics['Last_Purchase'] - customer_metrics['First_Purchase']
        ).dt.days + 1
        
        customer_metrics['Purchase_Frequency'] = (
            customer_metrics['Order_Count'] / customer_metrics['Customer_Lifetime_Days'] * 365
        ).fillna(0)
        
        return customer_metrics
        
    except Exception as e:
        st.error(f"Error creating customer summary: {e}")
        return pd.DataFrame()

def create_rfm_analysis(df, customer_summary):
    """
    Create RFM (Recency, Frequency, Monetary) analysis
    """
    st.subheader("📊 RFM Analysis")
    
    try:
        # Calculate RFM scores
        rfm_data = customer_summary.copy()
        
        # Recency (lower is better)
        rfm_data['R_Score'] = pd.qcut(
            rfm_data['Days_Since_Last_Purchase'].rank(method='first', ascending=False),
            5, labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        # Frequency (higher is better)
        rfm_data['F_Score'] = pd.qcut(
            rfm_data['Order_Count'].rank(method='first'),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Monetary (higher is better)
        rfm_data['M_Score'] = pd.qcut(
            rfm_data['Total_Revenue'].rank(method='first'),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Combined RFM Score
        rfm_data['RFM_Score'] = (
            rfm_data['R_Score'].astype(str) + 
            rfm_data['F_Score'].astype(str) + 
            rfm_data['M_Score'].astype(str)
        )
        
        # Customer segments based on RFM
        rfm_data['Customer_Segment'] = rfm_data.apply(classify_rfm_segment, axis=1)
        
        # Display RFM metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM segment distribution
            segment_counts = rfm_data['Customer_Segment'].value_counts()
            
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title='Customer Segment Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # RFM scatter plot
            fig = px.scatter(
                rfm_data,
                x='F_Score',
                y='M_Score',
                color='Customer_Segment',
                size='Total_Revenue',
                title='RFM Scatter Plot (Frequency vs Monetary)',
                hover_data=['Customer_Name', 'R_Score']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # RFM segment summary table
        st.subheader("📋 RFM Segment Summary")
        
        segment_summary = rfm_data.groupby('Customer_Segment').agg({
            'Customer_Name': 'count',
            'Total_Revenue': ['sum', 'mean'],
            'Order_Count': 'mean',
            'Days_Since_Last_Purchase': 'mean'
        }).round(2)
        
        segment_summary.columns = [
            'Customer_Count', 'Total_Revenue_SAR', 'Avg_Revenue_Per_Customer_SAR',
            'Avg_Order_Count', 'Avg_Days_Since_Last_Purchase'
        ]
        
        # Format currency columns
        segment_summary['Total_Revenue_SAR'] = segment_summary['Total_Revenue_SAR'].apply(lambda x: f"{x:,.0f} SAR")
        segment_summary['Avg_Revenue_Per_Customer_SAR'] = segment_summary['Avg_Revenue_Per_Customer_SAR'].apply(lambda x: f"{x:,.0f} SAR")
        
        st.dataframe(segment_summary, use_container_width=True)
        
        # Top customers by segment
        st.subheader("🏆 Top Customers by Segment")
        
        selected_segment = st.selectbox(
            "Select segment to view top customers",
            rfm_data['Customer_Segment'].unique()
        )
        
        top_customers = rfm_data[
            rfm_data['Customer_Segment'] == selected_segment
        ].nlargest(10, 'Total_Revenue')[
            ['Customer_Name', 'Total_Revenue', 'Order_Count', 'Days_Since_Last_Purchase', 'RFM_Score']
        ].copy()
        
        # Format revenue column
        top_customers['Total_Revenue'] = top_customers['Total_Revenue'].apply(lambda x: f"{x:,.0f} SAR")
        
        st.dataframe(top_customers, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in RFM analysis: {e}")

def classify_rfm_segment(row):
    """
    Classify customers into segments based on RFM scores
    """
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2 and m >= 3:
        return 'Cannot Lose Them'
    elif r <= 2 and f <= 2 and m <= 2:
        return 'Lost Customers'
    elif f >= 3 and m <= 2:
        return 'Price Sensitive'
    else:
        return 'Others'

def create_value_based_segmentation(customer_summary):
    """
    Create value-based customer segmentation
    """
    st.subheader("💰 Value-Based Segmentation")
    
    try:
        # Define value segments based on total revenue
        revenue_percentiles = customer_summary['Total_Revenue'].quantile([0.8, 0.95, 1.0])
        
        def classify_value_segment(revenue):
            if revenue >= revenue_percentiles[0.95]:
                return 'VIP Customers'
            elif revenue >= revenue_percentiles[0.8]:
                return 'High Value'
            elif revenue >= customer_summary['Total_Revenue'].median():
                return 'Medium Value'
            else:
                return 'Low Value'
        
        customer_summary['Value_Segment'] = customer_summary['Total_Revenue'].apply(classify_value_segment)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Value segment distribution
            segment_dist = customer_summary['Value_Segment'].value_counts()
            
            fig = px.bar(
                x=segment_dist.index,
                y=segment_dist.values,
                title='Customer Count by Value Segment',
                color=segment_dist.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue by segment
            revenue_by_segment = customer_summary.groupby('Value_Segment')['Total_Revenue'].sum()
            
            fig = px.pie(
                values=revenue_by_segment.values,
                names=revenue_by_segment.index,
                title='Revenue Distribution by Value Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment comparison
        st.subheader("📊 Segment Comparison")
        
        segment_comparison = customer_summary.groupby('Value_Segment').agg({
            'Customer_Name': 'count',
            'Total_Revenue': ['sum', 'mean'],
            'Avg_Order_Value': 'mean',
            'Order_Count': 'mean'
        }).round(2)
        
        segment_comparison.columns = [
            'Customer_Count', 'Total_Revenue_SAR', 'Avg_Revenue_Per_Customer_SAR',
            'Avg_Order_Value_SAR', 'Avg_Order_Count'
        ]
        
        # Format currency columns
        for col in ['Total_Revenue_SAR', 'Avg_Revenue_Per_Customer_SAR', 'Avg_Order_Value_SAR']:
            segment_comparison[col] = segment_comparison[col].apply(lambda x: f"{x:,.0f} SAR")
        
        st.dataframe(segment_comparison, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in value-based segmentation: {e}")

def create_geographic_segmentation(df):
    """
    Create geographic segmentation analysis
    """
    st.subheader("🌍 Geographic Segmentation")
    
    try:
        # Country analysis
        col1, col2 = st.columns(2)
        
        with col1:
            country_metrics = df.groupby('Country').agg({
                'Total Line Amount': 'sum',
                'Cust Name': 'nunique',
                'Invoice No.': 'nunique'
            }).reset_index()
            
            country_metrics.columns = ['Country', 'Total_Revenue', 'Customer_Count', 'Order_Count']
            country_metrics = country_metrics.sort_values('Total_Revenue', ascending=False)
            
            fig = px.bar(
                country_metrics.head(10),
                x='Country',
                y='Total_Revenue',
                title='Revenue by Country',
                color='Total_Revenue',
                color_continuous_scale='blues'
            )
            fig.update_layout(xaxis_tickangle=-45, yaxis_title="Revenue (SAR)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Local vs International
            local_intl = df.groupby('Local').agg({
                'Total Line Amount': 'sum',
                'Cust Name': 'nunique'
            }).reset_index()
            
            fig = px.pie(
                local_intl,
                values='Total Line Amount',
                names='Local',
                title='Local vs International Revenue'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic summary table
        st.subheader("📋 Geographic Summary")
        
        # Format the revenue column
        display_country = country_metrics.copy()
        display_country['Total_Revenue'] = display_country['Total_Revenue'].apply(lambda x: f"{x:,.0f} SAR")
        
        st.dataframe(display_country, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in geographic segmentation: {e}")

def create_behavioral_segmentation(df, customer_summary):
    """
    Create behavioral segmentation based on purchase patterns
    """
    st.subheader("🎯 Behavioral Segmentation")
    
    try:
        # Purchase frequency analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Frequency distribution
            fig = px.histogram(
                customer_summary,
                x='Purchase_Frequency',
                nbins=20,
                title='Purchase Frequency Distribution',
                labels={'Purchase_Frequency': 'Purchases per Year'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Order value distribution
            fig = px.histogram(
                customer_summary,
                x='Avg_Order_Value',
                nbins=20,
                title='Average Order Value Distribution',
                labels={'Avg_Order_Value': 'Average Order Value (SAR)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer class analysis
        st.subheader("🏢 Customer Class Analysis")
        
        class_analysis = df.groupby('Cust Class Code').agg({
            'Total Line Amount': ['sum', 'mean'],
            'Cust Name': 'nunique',
            'QTY': 'sum'
        }).round(2)
        
        class_analysis.columns = [
            'Total_Revenue', 'Avg_Order_Value', 'Customer_Count', 'Total_Quantity'
        ]
        
        # Format currency columns
        display_class = class_analysis.copy()
        display_class['Total_Revenue'] = display_class['Total_Revenue'].apply(lambda x: f"{x:,.0f} SAR")
        display_class['Avg_Order_Value'] = display_class['Avg_Order_Value'].apply(lambda x: f"{x:,.0f} SAR")
        
        st.dataframe(display_class, use_container_width=True)
        
        # Class distribution chart
        fig = px.bar(
            x=class_analysis.index,
            y=class_analysis['Total_Revenue'],
            title='Revenue by Customer Class',
            color=class_analysis['Total_Revenue'],
            color_continuous_scale='viridis'
        )
        fig.update_layout(yaxis_title="Revenue (SAR)")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in behavioral segmentation: {e}")
