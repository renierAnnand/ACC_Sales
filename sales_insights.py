import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import all modules directly (since they're in the same directory)
try:
    import sales_dashboard
    import customer_segmentation
    import sales_forecasting
    import salesperson_performance
    import discount_analysis
    import bu_benchmarking
    import product_insights
    from sales_insights import SalesInsights  # Import the core insights class
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import io
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please make sure all module files are in the same directory as app.py")
    st.error("Make sure you have installed: pip install plotly xlsxwriter")

# Page configuration
st.set_page_config(
    page_title="ACC Sales Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_and_process_data(uploaded_file):
    """
    Load and preprocess the Excel data to match the expected structure
    """
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Handle the duplicate BU columns (BU_1, BU Name_1) by keeping the first ones
        if 'BU_1' in df.columns:
            df = df.drop(['BU_1'], axis=1, errors='ignore')
        if 'BU Name_1' in df.columns:
            df = df.drop(['BU Name_1'], axis=1, errors='ignore')
        
        # Data type conversions
        try:
            # Convert date columns
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
            if 'Created On' in df.columns:
                df['Created On'] = pd.to_datetime(df['Created On'], errors='coerce')
            
            # Convert numeric columns - handle the space in column name
            numeric_columns = ['Line Amount', 'Tax Amount', 'Total Line Amount', 'Applied Amount',
                             'Item Cost', 'Total Cost', 'QTY']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                # Also check for column with spaces
                col_with_space = f' {col} '
                if col_with_space in df.columns:
                    df[col_with_space] = pd.to_numeric(df[col_with_space], errors='coerce')
            
            # Convert ID columns to int where possible
            id_columns = ['Invoice No.', 'Inv Line No.', 'Sales Order No.', 'Cust No.',
                         'Item Number', 'Account', 'Act.', 'Site', 'BU']
            for col in id_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
        except Exception as e:
            st.warning(f"Some data type conversions failed: {e}")
        
        # Add calculated fields
        try:
            # Profit calculation
            if 'Total Line Amount' in df.columns and 'Total Cost' in df.columns:
                df['Profit'] = df['Total Line Amount'] - df['Total Cost'].fillna(0)
                df['Profit Margin %'] = np.where(
                    df['Total Line Amount'] != 0,
                    (df['Profit'] / df['Total Line Amount'] * 100).round(2),
                    0
                )
            
            # Date components
            if 'Invoice Date' in df.columns:
                df['Year'] = df['Invoice Date'].dt.year
                df['Month'] = df['Invoice Date'].dt.month
                df['Quarter'] = df['Invoice Date'].dt.quarter
                df['YearMonth'] = df['Invoice Date'].dt.to_period('M').astype(str)
                df['MonthName'] = df['Invoice Date'].dt.month_name()
        
        except Exception as e:
            st.warning(f"Error adding calculated fields: {e}")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_advanced_insights(df):
    """Create the Advanced Sales Insights dashboard using the sales_insights module."""
    
    st.title("🧠 Advanced Sales Insights")
    st.markdown("**Comprehensive business intelligence and analytics dashboard**")
    
    # Initialize the insights module with DataFrame
    try:
        # Create a copy of the DataFrame and handle timezone issues
        df_copy = df.copy()
        
        # Convert timezone-aware datetimes to timezone-naive
        datetime_columns = df_copy.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns
        for col in datetime_columns:
            if df_copy[col].dt.tz is not None:
                df_copy[col] = df_copy[col].dt.tz_localize(None)
        
        # Create a temporary Excel file from the DataFrame for the SalesInsights class
        temp_file = "temp_data.xlsx"
        df_copy.to_excel(temp_file, index=False)
        insights = SalesInsights(temp_file)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    except Exception as e:
        st.error(f"Error initializing insights module: {str(e)}")
        st.error("Please check your data format and try again.")
        return
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📊 Analytics Menu")
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Executive Summary",
                "Revenue Analytics", 
                "Customer Intelligence",
                "Product Performance",
                "Sales Team Analytics",
                "Business Unit Analysis",
                "Geographic Analysis",
                "Trend Analysis",
                "Advanced Analytics"
            ]
        )
    
    # Main content based on selection
    if analysis_type == "Executive Summary":
        show_executive_summary(insights, df)
    elif analysis_type == "Revenue Analytics":
        show_revenue_analytics(insights, df)
    elif analysis_type == "Customer Intelligence":
        show_customer_intelligence(insights, df)
    elif analysis_type == "Product Performance":
        show_product_performance(insights, df)
    elif analysis_type == "Sales Team Analytics":
        show_sales_team_analytics(insights, df)
    elif analysis_type == "Business Unit Analysis":
        show_business_unit_analysis(insights, df)
    elif analysis_type == "Geographic Analysis":
        show_geographic_analysis(insights, df)
    elif analysis_type == "Trend Analysis":
        show_trend_analysis(insights, df)
    elif analysis_type == "Advanced Analytics":
        show_advanced_analytics(insights, df)

def show_executive_summary(insights, df):
    """Display executive summary dashboard."""
    st.header("📈 Executive Summary")
    
    try:
        # Get executive summary data
        summary = insights.generate_executive_summary()
        revenue_overview = insights.revenue_overview()
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Revenue",
                f"{revenue_overview['total_revenue']:,.0f} SAR",
                delta=None
            )
            
        with col2:
            st.metric(
                "Total Profit", 
                f"{revenue_overview['total_profit']:,.0f} SAR",
                delta=f"{revenue_overview['profit_margin_pct']:.1f}% margin"
            )
            
        with col3:
            st.metric(
                "Total Orders",
                f"{revenue_overview['total_orders']:,}",
                delta=None
            )
            
        with col4:
            st.metric(
                "Avg Order Value",
                f"{revenue_overview['avg_order_value']:,.0f} SAR",
                delta=None
            )
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Unique Customers", f"{revenue_overview['total_customers']:,}")
        with col6:
            st.metric("Revenue per Customer", f"{revenue_overview['revenue_per_customer']:,.0f} SAR")
        with col7:
            top_customers = insights.top_customers_analysis(1)
            if not top_customers.empty:
                st.metric("Top Customer", top_customers.iloc[0]['Customer_Name'][:20])
        with col8:
            sales_team = insights.salesperson_performance()
            if not sales_team.empty:
                st.metric("Top Salesperson", sales_team.iloc[0]['Salesperson Name'][:20])
        
        # Revenue trend chart
        st.subheader("📈 Monthly Revenue Trend")
        monthly_data = insights.monthly_revenue_trend()
        
        fig = px.line(
            monthly_data, 
            x='Date', 
            y='Total Line Amount',
            title="Monthly Revenue Trend",
            labels={'Total Line Amount': 'Revenue (SAR)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("🎯 Key Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Period Coverage:** {summary['period_covered']}
            
            **Performance Highlights:**
            - 💰 Total Revenue: {summary['total_revenue']}
            - 📊 Profit Margin: {summary['profit_margin']}
            - 🏆 Top Customer: {summary['top_customer']}
            - 🌟 Best Performer: {summary['best_salesperson']}
            """)
            
        with col2:
            # Quick stats
            bu_performance = insights.business_unit_performance()
            if not bu_performance.empty:
                top_bu = bu_performance.iloc[0]
                st.markdown(f"""
                **Business Unit Performance:**
                - 🏢 Top BU: {top_bu['BU Name']}
                - 💵 BU Revenue: {top_bu['Total Line Amount']:,.0f} SAR
                - 📈 BU Margin: {top_bu['Profit_Margin']:.1f}%
                
                **Customer Distribution:**
                - 👥 Total Customers: {revenue_overview['total_customers']:,}
                - 📦 Average Orders per Customer: {revenue_overview['total_orders'] / revenue_overview['total_customers']:.1f}
                """)
        
    except Exception as e:
        st.error(f"Error generating executive summary: {str(e)}")

def show_revenue_analytics(insights, df):
    """Display revenue analytics."""
    st.header("💰 Revenue Analytics")
    
    # Revenue overview
    st.subheader("📊 Revenue Overview")
    overview = insights.revenue_overview()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue breakdown pie chart
        fig = px.pie(
            values=[overview['total_profit'], overview['total_cost']], 
            names=['Profit', 'Cost'],
            title="Revenue Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown(f"""
        **Revenue Metrics:**
        - 💰 Total Revenue: {overview['total_revenue']:,.2f} SAR
        - 💵 Total Cost: {overview['total_cost']:,.2f} SAR
        - 📈 Total Profit: {overview['total_profit']:,.2f} SAR
        - 📊 Profit Margin: {overview['profit_margin_pct']:.2f}%
        
        **Order Metrics:**
        - 🛒 Total Orders: {overview['total_orders']:,}
        - 💳 Average Order Value: {overview['avg_order_value']:,.2f} SAR
        - 👥 Revenue per Customer: {overview['revenue_per_customer']:,.2f} SAR
        """)
    
    # Monthly and quarterly trends
    st.subheader("📈 Trend Analysis")
    
    tab1, tab2 = st.tabs(["Monthly Trends", "Quarterly Performance"])
    
    with tab1:
        monthly_data = insights.monthly_revenue_trend()
        
        # Create subplot with revenue and growth rate
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Revenue', 'Growth Rate %'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_data['Date'], y=monthly_data['Total Line Amount'], 
                      name='Revenue', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_data['Date'], y=monthly_data['Growth_Rate'], 
                      name='Growth Rate %', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.dataframe(monthly_data[['Date', 'Total Line Amount', 'Growth_Rate', 'Invoice No.']].tail(12))
    
    with tab2:
        quarterly_data = insights.quarterly_performance()
        
        fig = px.bar(
            quarterly_data,
            x='Quarter',
            y='Total Line Amount',
            color='Year',
            title='Quarterly Revenue Performance',
            labels={'Total Line Amount': 'Revenue (SAR)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(quarterly_data)

def show_customer_intelligence(insights, df):
    """Display customer intelligence dashboard."""
    st.header("👥 Customer Intelligence")
    
    # Top customers analysis
    st.subheader("🏆 Top Customers")
    
    top_n = st.slider("Number of top customers to display", 5, 50, 20)
    top_customers = insights.top_customers_analysis(top_n)
    
    # Top customers chart
    fig = px.bar(
        top_customers.head(10),
        x='Customer_Name',
        y='Total_Revenue',
        title=f'Top 10 Customers by Revenue',
        labels={'Total_Revenue': 'Revenue (SAR)'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            top_customers[['Customer_Name', 'Total_Revenue', 'Order_Count', 'Avg_Order_Value']].head(10),
            use_container_width=True
        )
    
    with col2:
        # Customer segmentation
        st.subheader("🎯 Customer Segmentation")
        segments = insights.customer_segmentation()
        segment_counts = segments['Segment'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Segmentation Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn risk analysis
    st.subheader("⚠️ Churn Risk Analysis")
    churn_data = insights.customer_churn_risk()
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_summary = churn_data['Churn_Risk'].value_counts()
        fig = px.bar(
            x=risk_summary.index,
            y=risk_summary.values,
            title="Customers by Churn Risk Level",
            labels={'x': 'Risk Level', 'y': 'Number of Customers'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # High risk customers
        high_risk = churn_data[churn_data['Churn_Risk'].isin(['High Risk', 'Critical Risk'])]
        st.markdown(f"**High Risk Customers: {len(high_risk)}**")
        if not high_risk.empty:
            st.dataframe(
                high_risk[['Cust Name', 'Days_Since_Last_Purchase', 'Total Line Amount', 'Churn_Risk']].head(10),
                use_container_width=True
            )

def show_product_performance(insights, df):
    """Display product performance analysis."""
    st.header("🛍️ Product Performance")
    
    # Top products
    st.subheader("🏆 Top Products by Revenue")
    
    top_n = st.slider("Number of top products to display", 5, 50, 20)
    top_products = insights.product_performance(top_n)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            top_products.head(10),
            x='Total Line Amount',
            y='Line Description',
            orientation='h',
            title='Top 10 Products by Revenue',
            labels={'Total Line Amount': 'Revenue (SAR)'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            top_products[['Line Description', 'Total Line Amount', 'Profit_Margin']].head(10),
            use_container_width=True
        )
    
    # Brand analysis
    st.subheader("🏷️ Brand Performance")
    brand_analysis = insights.brand_analysis()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            brand_analysis.head(8),
            values='Total Line Amount',
            names='Brand',
            title="Revenue Distribution by Brand"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(brand_analysis.head(10), use_container_width=True)

def show_sales_team_analytics(insights, df):
    """Display sales team analytics."""
    st.header("👨‍💼 Sales Team Analytics")
    
    # Sales performance
    sales_performance = insights.salesperson_performance()
    
    st.subheader("🌟 Sales Team Performance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            sales_performance.head(10),
            x='Salesperson Name',
            y='Total Line Amount',
            title='Top 10 Salespeople by Revenue',
            labels={'Total Line Amount': 'Revenue (SAR)'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            sales_performance[['Salesperson Name', 'Total Line Amount', 'Invoice No.', 'Profit_Margin']].head(10),
            use_container_width=True
        )
    
    # Sales efficiency metrics
    st.subheader("⚡ Sales Efficiency")
    efficiency_metrics = insights.sales_efficiency_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            efficiency_metrics,
            x='Avg_Monthly_Revenue',
            y='Consistency_Score',
            size='Avg_Monthly_Orders',
            hover_name='Salesperson',
            title='Sales Efficiency: Revenue vs Consistency'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(efficiency_metrics, use_container_width=True)

def show_business_unit_analysis(insights, df):
    """Display business unit analysis."""
    st.header("🏢 Business Unit Analysis")
    
    bu_performance = insights.business_unit_performance()
    
    # BU performance overview
    st.subheader("📊 BU Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            bu_performance,
            x='BU Name',
            y='Total Line Amount',
            title='Revenue by Business Unit',
            labels={'Total Line Amount': 'Revenue (SAR)'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            bu_performance,
            x='Total Line Amount',
            y='Profit_Margin',
            size='Invoice No.',
            hover_name='BU Name',
            title='Revenue vs Profit Margin by BU'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # BU details table
    st.subheader("📋 BU Performance Details")
    st.dataframe(bu_performance, use_container_width=True)

def show_geographic_analysis(insights, df):
    """Display geographic analysis."""
    st.header("🌍 Geographic Analysis")
    
    geo_analysis = insights.geographical_analysis()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            geo_analysis,
            x='Country',
            y='Total Line Amount',
            color='Local',
            title='Revenue by Geography',
            labels={'Total Line Amount': 'Revenue (SAR)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            geo_analysis,
            values='Total Line Amount',
            names='Local',
            title='Local vs International Revenue'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(geo_analysis, use_container_width=True)

def show_trend_analysis(insights, df):
    """Display trend analysis."""
    st.header("📈 Trend Analysis")
    
    # Seasonal analysis
    st.subheader("🗓️ Seasonal Analysis")
    seasonal_data = insights.seasonal_analysis()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            seasonal_data,
            x='Month_Name',
            y='Total Line Amount',
            title='Revenue by Month',
            labels={'Total Line Amount': 'Revenue (SAR)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            seasonal_data,
            x='Month_Num',
            y='Invoice No.',
            title='Orders by Month',
            labels={'Invoice No.': 'Number of Orders'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment terms analysis
    st.subheader("💳 Payment Terms Analysis")
    payment_analysis = insights.payment_terms_analysis()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            payment_analysis,
            values='Total Line Amount',
            names='Term Name',
            title='Revenue by Payment Terms'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(payment_analysis, use_container_width=True)

def show_advanced_analytics(insights, df):
    """Display advanced analytics and insights."""
    st.header("🔬 Advanced Analytics")
    
    st.subheader("📊 Comprehensive Analysis")
    
    # Export functionality
    st.subheader("💾 Export Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Export Executive Summary"):
            summary = insights.generate_executive_summary()
            summary_df = pd.DataFrame([summary])
            
            # Convert to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            st.download_button(
                label="Download Executive Summary",
                data=output.getvalue(),
                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("👥 Export Customer Analysis"):
            customers = insights.top_customers_analysis(100)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                customers.to_excel(writer, sheet_name='Top Customers', index=False)
                
                segments = insights.customer_segmentation()
                segments.to_excel(writer, sheet_name='Customer Segments', index=False)
            
            st.download_button(
                label="Download Customer Analysis",
                data=output.getvalue(),
                file_name=f"customer_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("📈 Export All Reports"):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Executive summary
                summary_df = pd.DataFrame([insights.generate_executive_summary()])
                summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
                
                # Top customers
                insights.top_customers_analysis(50).to_excel(writer, sheet_name='Top Customers', index=False)
                
                # Product performance
                insights.product_performance(100).to_excel(writer, sheet_name='Product Performance', index=False)
                
                # Sales team
                insights.salesperson_performance().to_excel(writer, sheet_name='Sales Team', index=False)
                
                # Business units
                insights.business_unit_performance().to_excel(writer, sheet_name='Business Units', index=False)
                
                # Monthly trends
                insights.monthly_revenue_trend().to_excel(writer, sheet_name='Monthly Trends', index=False)
            
            st.download_button(
                label="Download All Reports",
                data=output.getvalue(),
                file_name=f"comprehensive_sales_report_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Data insights
    st.subheader("🎯 Key Data Insights")
    
    try:
        summary = insights.generate_executive_summary()
        revenue_overview = insights.revenue_overview()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **📈 Business Performance:**
            - Revenue Growth Opportunity: Focus on top 20% of customers
            - Profit Optimization: Current margin is {revenue_overview['profit_margin_pct']:.1f}%
            - Customer Efficiency: {revenue_overview['revenue_per_customer']:,.0f} SAR revenue per customer
            
            **🎯 Recommendations:**
            - Increase engagement with medium-risk churn customers
            - Expand successful product lines
            - Optimize sales team territories
            """)
        
        with col2:
            # Quick insights from data
            top_customers = insights.top_customers_analysis(10)
            top_20_pct_revenue = top_customers['Total_Revenue'].sum()
            total_revenue = revenue_overview['total_revenue']
            concentration = (top_20_pct_revenue / total_revenue) * 100
            
            st.markdown(f"""
            **📊 Data Insights:**
            - Top 10 customers contribute {concentration:.1f}% of total revenue
            - Average order value: {revenue_overview['avg_order_value']:,.0f} SAR
            - Customer base: {revenue_overview['total_customers']:,} unique customers
            
            **🔍 Analysis Period:**
            - From: {summary['period_covered'].split(' to ')[0]}
            - To: {summary['period_covered'].split(' to ')[1]}
            """)
    
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")

def main():
    """
    Main application function
    """
    # Title and description
    st.title("🏢 ACC Sales Intelligence System")
    st.markdown("**Advanced Analytics Dashboard for Sales Performance & Business Intelligence**")
    st.markdown("*All amounts displayed in Saudi Riyal (SAR)*")
    
    # Sidebar for navigation and file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload your sales data Excel file (same format as sample.xlsx)"
        )
        
        if uploaded_file is not None:
            # Load and process data
            with st.spinner('Loading and processing data...'):
                df = load_and_process_data(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Data loaded successfully!")
                st.info(f"📊 {len(df):,} rows loaded")
                
                # Store data in session state
                st.session_state.df = df
                
                # Navigation menu
                st.header("🧭 Navigation")
                page = st.selectbox(
                    "Select Analysis",
                    [
                        "Sales Dashboard",
                        "Advanced Sales Insights",  # New comprehensive insights module
                        "Customer Segmentation", 
                        "Sales Forecasting",
                        "Salesperson Performance",
                        "Discount Analysis",
                        "BU Benchmarking",
                        "Product Insights"
                    ]
                )
                
                # Data overview in sidebar
                with st.expander("📈 Data Overview"):
                    st.write(f"**Date Range:** {df['Invoice Date'].min().strftime('%Y-%m-%d')} to {df['Invoice Date'].max().strftime('%Y-%m-%d')}")
                    st.write(f"**Total Revenue:** {df['Total Line Amount'].sum():,.2f} SAR")
                    st.write(f"**Business Units:** {df['BU Name'].nunique()}")
                    st.write(f"**Customers:** {df['Cust Name'].nunique()}")
                    st.write(f"**Salespeople:** {df['Salesperson Name'].nunique()}")
        else:
            st.info("👆 Please upload an Excel file to begin analysis")
            page = None
    
    # Main content area
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Route to selected page
        if page == "Sales Dashboard":
            sales_dashboard.create_sales_dashboard(df)
        elif page == "Advanced Sales Insights":
            create_advanced_insights(df)  # Call function within this file
        elif page == "Customer Segmentation":
            customer_segmentation.create_customer_segmentation(df)
        elif page == "Sales Forecasting":
            sales_forecasting.create_sales_forecast(df)
        elif page == "Salesperson Performance":
            salesperson_performance.create_performance_analysis(df)
        elif page == "Discount Analysis":
            discount_analysis.create_discount_analysis(df)
        elif page == "BU Benchmarking":
            bu_benchmarking.create_bu_benchmark(df)
        elif page == "Product Insights":
            product_insights.create_product_insights(df)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to ACC Sales Intelligence System
        
        This comprehensive analytics platform provides insights into:
        
        ### 📊 **Sales Dashboard**
        - Key performance metrics
        - Monthly trends and patterns
        - Business unit performance comparison
        
        ### 🧠 **Advanced Sales Insights**
        - Executive summary and KPIs
        - Revenue trend analysis
        - Customer intelligence and segmentation
        - Profitability analytics
        - Churn risk assessment
        - Sales team efficiency metrics
        
        ### 👥 **Customer Segmentation**
        - Customer classification and analysis
        - Geographic distribution
        - Customer value analysis
        
        ### 📈 **Sales Forecasting**
        - Predictive analytics
        - Trend analysis
        - Future sales projections
        
        ### 🏆 **Salesperson Performance**
        - Individual performance metrics
        - Comparative analysis
        - Commission and target tracking
        
        ### 💰 **Discount Analysis**
        - Profit margin analysis
        - Discount impact assessment
        - Pricing optimization insights
        
        ### 🏢 **BU Benchmarking**
        - Business unit comparison
        - Performance benchmarking
        - Cross-unit analytics
        
        ### 📦 **Product Insights**
        - Product performance analysis
        - Brand comparison
        - Category insights
        
        ---
        
        **Note:** All financial amounts are displayed in Saudi Riyal (SAR)
        
        **To get started:** Upload your Excel file using the sidebar file uploader.
        """)

if __name__ == "__main__":
    main()
