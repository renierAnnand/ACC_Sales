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
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please make sure all module files are in the same directory as app.py")

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
