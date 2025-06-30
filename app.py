import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the current directory to Python path to ensure modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the modules directory to the Python path
modules_dir = os.path.join(current_dir, 'modules')
if modules_dir not in sys.path:
    sys.path.insert(0, modules_dir)

# Page configuration
st.set_page_config(
    page_title="ACC Sales Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.streamlit.io/community',
        'Report a bug': None,
        'About': "# ACC Sales Intelligence System\nBuilt with Streamlit for comprehensive sales analytics and insights."
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .module-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .stSelectbox > div > div > div {
        background-color: white;
    }
    
    .error-container {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-container {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# DATA LOADING FUNCTIONS (Built-in)
# =====================================

@st.cache_data
def load_and_merge_data_from_file(uploaded_file):
    """
    Optimized loading for large Excel or CSV files.
    Returns a cleaned and standardized DataFrame.
    """
    
    if uploaded_file is None:
        return pd.DataFrame()
    
    try:
        file_name = uploaded_file.name.lower()
        
        # Handle CSV files
        if file_name.endswith('.csv'):
            with st.spinner("📊 Processing CSV file..."):
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ CSV loaded successfully ({len(df):,} records)")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        st.error(f"Error reading CSV: {str(e)}")
                        continue
                
                if df is None:
                    st.error("❌ Could not read CSV file. Please check the file format.")
                    return pd.DataFrame()
                
                # Add metadata for CSV
                df['Source_Sheet'] = 'CSV_Data'
                df['Data_Year'] = datetime.now().year
                
                # Clean and standardize the data
                with st.spinner("🔄 Cleaning and standardizing data..."):
                    cleaned_df = clean_and_standardize_data_optimized(df)
                
                st.success(f"🎉 Successfully processed {len(cleaned_df):,} records")
                return cleaned_df
        
        # Handle Excel files with optimization
        elif file_name.endswith(('.xlsx', '.xls')):
            with st.spinner("📊 Reading Excel file..."):
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                st.info(f"📊 Processing {len(sheet_names)} sheets...")
                
                # Use progress bar for large files
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                dataframes = []
                total_records = 0
                
                for i, sheet_name in enumerate(sheet_names):
                    try:
                        # Update progress
                        progress = (i + 1) / len(sheet_names)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing sheet {i+1}/{len(sheet_names)}: {sheet_name}")
                        
                        # Read sheet with optimizations
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                        
                        # Skip empty sheets
                        if df.empty:
                            continue
                        
                        # Add metadata efficiently
                        df['Source_Sheet'] = sheet_name
                        
                        # Extract year from sheet name efficiently
                        year = datetime.now().year
                        for year_candidate in [2022, 2023, 2024, 2025]:
                            if str(year_candidate) in sheet_name or str(year_candidate)[-2:] in sheet_name:
                                year = year_candidate
                                break
                        df['Data_Year'] = year
                        
                        dataframes.append(df)
                        total_records += len(df)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error loading sheet '{sheet_name}': {str(e)}")
                        continue
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if not dataframes:
                    st.error("❌ No data could be loaded from any sheet")
                    return pd.DataFrame()
                
                # Combine dataframes efficiently
                with st.spinner(f"🔄 Combining {len(dataframes)} sheets ({total_records:,} records)..."):
                    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
                
                # Clean and standardize the data
                with st.spinner("🔄 Cleaning and standardizing data..."):
                    cleaned_df = clean_and_standardize_data_optimized(combined_df)
                
                st.success(f"🎉 Successfully processed {len(cleaned_df):,} records from {len(dataframes)} sheets")
                return cleaned_df
                
        else:
            st.error("❌ Unsupported file format. Please upload .xlsx, .xls, or .csv files.")
            return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

def clean_and_standardize_data_optimized(df):
    """
    Optimized cleaning and standardization for large datasets.
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying original data
    df = df.copy()
    
    # Optimize column name standardization
    column_mapping = {
        'customername': 'Customer Name',
        'customer': 'Customer Name',
        'invoicenumber': 'Invoice Number',
        'itemdescription': 'Item Description',
        'product': 'Item Description',
        'quantity': 'Quantity',
        'unitprice': 'Unit Price',
        'totalsales': 'Total Sales',
        'sales': 'Total Sales',
        'revenue': 'Total Sales',
        'businessunit': 'Business Unit',
        'bu': 'Business Unit',
        'salesperson': 'Salesperson',
        'salesrep': 'Salesperson',
        'totalcost': 'Total Cost',
        'discount': 'Discount',
        'profit': 'Profit',
        'country': 'Country',
        'location': 'Location',
        'invoicedate': 'Invoice Date',
        'date': 'Invoice Date'
    }
    
    # Create a mapping of actual column names to standardized names
    actual_mapping = {}
    df_columns_lower = [col.lower().replace(' ', '').replace('_', '') for col in df.columns]
    
    for i, col_lower in enumerate(df_columns_lower):
        if col_lower in column_mapping:
            actual_mapping[df.columns[i]] = column_mapping[col_lower]
    
    # Apply column renaming
    df = df.rename(columns=actual_mapping)
    
    # Optimize data type conversions
    numeric_columns = ['Quantity', 'Unit Price', 'Total Sales', 'Total Cost', 'Discount', 'Profit']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Optimize date conversion
    if 'Invoice Date' in df.columns:
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    
    # Optimize string cleaning
    string_columns = ['Customer Name', 'Item Description', 'Business Unit', 'Salesperson', 'Country', 'Location']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', ''], np.nan)
    
    # Calculate derived fields efficiently
    if 'Profit' not in df.columns and 'Total Sales' in df.columns and 'Total Cost' in df.columns:
        df['Profit'] = df['Total Sales'] - df['Total Cost']
    
    if 'Total Sales' not in df.columns and 'Quantity' in df.columns and 'Unit Price' in df.columns:
        df['Total Sales'] = df['Quantity'] * df['Unit Price']
    
    if 'Total Sales' in df.columns and 'Profit' in df.columns:
        df['Profit Margin %'] = np.where(df['Total Sales'] > 0, 
                                        (df['Profit'] / df['Total Sales'] * 100).round(2), 0)
    
    # Add date-based fields efficiently
    if 'Invoice Date' in df.columns:
        df['Invoice Year'] = df['Invoice Date'].dt.year
        df['Invoice Month'] = df['Invoice Date'].dt.month
        df['Invoice Quarter'] = df['Invoice Date'].dt.quarter
    
    # Clean text fields efficiently
    if 'Business Unit' in df.columns:
        df['Business Unit'] = df['Business Unit'].str.title()
    
    if 'Customer Name' in df.columns:
        df['Customer Name'] = df['Customer Name'].str.title()
    
    # Remove rows with missing critical data
    critical_columns = ['Customer Name', 'Total Sales']
    existing_critical = [col for col in critical_columns if col in df.columns]
    if existing_critical:
        df = df.dropna(subset=existing_critical)
    
    # Remove duplicates efficiently
    df = df.drop_duplicates()
    
    # Sort by date if available
    if 'Invoice Date' in df.columns:
        df = df.sort_values('Invoice Date', ignore_index=True)
    
    return df

def clean_and_standardize_data(df):
    """
    Clean and standardize the merged dataframe.
    """
    
    # Make a copy to avoid modifying original data
    df = df.copy()
    
    # Standardize column names (remove spaces, make consistent)
    column_mapping = {
        'Customer Name': 'Customer Name',
        'Invoice Number': 'Invoice Number',
        'Item Description': 'Item Description',
        'Quantity': 'Quantity',
        'Unit Price': 'Unit Price',
        'Total Sales': 'Total Sales',
        'Business Unit': 'Business Unit',
        'Salesperson': 'Salesperson',
        'Total Cost': 'Total Cost',
        'Discount': 'Discount',
        'Profit': 'Profit',
        'Country': 'Country',
        'Location': 'Location',
        'Invoice Date': 'Invoice Date'
    }
    
    # Rename columns if they exist (case-insensitive matching)
    for old_name, new_name in column_mapping.items():
        # Find matching column (case-insensitive)
        matching_cols = [col for col in df.columns if col.lower().replace(' ', '').replace('_', '') == 
                        old_name.lower().replace(' ', '').replace('_', '')]
        
        if matching_cols:
            df = df.rename(columns={matching_cols[0]: new_name})
    
    # Clean and convert data types
    
    # Numeric columns
    numeric_columns = ['Quantity', 'Unit Price', 'Total Sales', 'Total Cost', 'Discount', 'Profit']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Date columns
    if 'Invoice Date' in df.columns:
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    
    # String columns - clean up
    string_columns = ['Customer Name', 'Item Description', 'Business Unit', 'Salesperson', 'Country', 'Location']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace 'nan' string with actual NaN
            df[col] = df[col].replace('nan', np.nan)
    
    # Calculate derived fields if missing
    
    # Calculate Profit if not available but Total Sales and Total Cost are
    if 'Profit' not in df.columns and 'Total Sales' in df.columns and 'Total Cost' in df.columns:
        df['Profit'] = df['Total Sales'] - df['Total Cost']
    
    # Calculate Total Sales if not available but Quantity and Unit Price are
    if 'Total Sales' not in df.columns and 'Quantity' in df.columns and 'Unit Price' in df.columns:
        df['Total Sales'] = df['Quantity'] * df['Unit Price']
    
    # Calculate Profit Margin
    if 'Total Sales' in df.columns and 'Profit' in df.columns:
        df['Profit Margin %'] = (df['Profit'] / df['Total Sales'] * 100).round(2)
    
    # Add additional derived fields
    
    # Month and Year from Invoice Date
    if 'Invoice Date' in df.columns:
        df['Invoice Year'] = df['Invoice Date'].dt.year
        df['Invoice Month'] = df['Invoice Date'].dt.month
        df['Invoice Quarter'] = df['Invoice Date'].dt.quarter
        df['Month-Year'] = df['Invoice Date'].dt.to_period('M')
    
    # Clean Business Unit names (standardize)
    if 'Business Unit' in df.columns:
        df['Business Unit'] = df['Business Unit'].str.title()
    
    # Clean Customer Names
    if 'Customer Name' in df.columns:
        df['Customer Name'] = df['Customer Name'].str.title()
    
    # Remove rows with missing critical data
    critical_columns = ['Customer Name', 'Total Sales']
    for col in critical_columns:
        if col in df.columns:
            df = df.dropna(subset=[col])
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Sort by Invoice Date if available
    if 'Invoice Date' in df.columns:
        df = df.sort_values('Invoice Date')
    
    return df

def generate_sample_data():
    """
    Generate realistic sample data for testing purposes.
    """
    
    np.random.seed(42)
    
    # Generate sample data
    n_records = 2000  # Substantial dataset
    
    # Realistic sample data for Middle East context
    customers = [
        "Saudi Aramco", "SABIC", "STC - Saudi Telecom", "Al Rajhi Bank", "SAMBA Bank",
        "NCB - National Commercial Bank", "Saudi Electricity Company", "Ma'aden", "ACWA Power",
        "Saudi Airlines", "Etihad Airways", "Emirates Group", "Qatar Airways", "Kuwait Airways",
        "ADNOC", "Dubai Petroleum", "Qatar Petroleum", "Borouge", "Emirates Steel",
        "Almarai Company", "Savola Group", "Abdullah Al Othaim Markets", "BinDawood Holding",
        "Jarir Marketing", "Extra Stores", "Al Futtaim Group", "Majid Al Futtaim",
        "Emaar Properties", "DAMAC Properties", "Kingdom Holding", "Al Hokair Group",
        "Olayan Group", "Al Muhaidib Group", "Abdul Latif Jameel", "Chalhoub Group",
        "Al Habtoor Group", "Al Naboodah Group", "Al Ghurair Group", "Al Fares Group",
        "Modern Electronics", "Jahez Platform", "Noon Academy", "Careem Networks",
        "Talabat", "Hungerstation", "Wadi.com", "Souq.com", "Namshi", "Fikra Space"
    ]
    
    products = [
        "Enterprise Resource Planning (ERP)", "Customer Relationship Management (CRM)",
        "Business Intelligence Platform", "Data Analytics Suite", "Cloud Infrastructure",
        "Cybersecurity Solutions", "Digital Transformation Services", "AI & Machine Learning",
        "IoT Solutions", "Blockchain Technology", "Mobile App Development", "Web Development",
        "E-commerce Platform", "Payment Gateway Integration", "Supply Chain Management",
        "Human Resource Management", "Financial Management System", "Project Management Tools",
        "Document Management System", "Workflow Automation", "API Integration Services",
        "Database Management", "Server Infrastructure", "Network Security", "Backup Solutions",
        "Disaster Recovery", "Cloud Migration Services", "DevOps Implementation",
        "Quality Assurance Testing", "Performance Monitoring", "Technical Support Services",
        "Training & Certification", "Consulting Services", "System Integration",
        "Software Licensing", "Hardware Procurement", "Maintenance Contracts",
        "Upgrade Services", "Custom Software Development", "Mobile Security Solutions",
        "Email Security", "Endpoint Protection", "Identity Management", "Compliance Software",
        "Audit & Risk Management", "Business Continuity Planning", "Green IT Solutions",
        "Digital Marketing Platform", "Social Media Management", "Content Management System"
    ]
    
    salespeople = [
        "Ahmed Al-Rashid", "Fatima Al-Zahra", "Mohammed Al-Saud", "Noor Al-Fahad",
        "Omar Al-Mahmoud", "Sarah Al-Khalil", "Khaled Al-Mutairi", "Layla Al-Dosari",
        "Abdulaziz Al-Othman", "Maryam Al-Harbi", "Faisal Al-Qahtani", "Huda Al-Shammari",
        "Nasser Al-Enezi", "Reem Al-Sulaimani", "Tariq Al-Balawi", "Aisha Al-Ghamdi",
        "Hamad Al-Ajmi", "Noura Al-Khatib", "Saleh Al-Thuwaini", "Dalal Al-Sabah",
        "Yousef Al-Ahmad", "Hanan Al-Rashid", "Badr Al-Tamimi", "Lina Al-Mansouri"
    ]
    
    business_units = [
        'Digital Solutions', 'Enterprise Services', 'Cloud Technologies', 
        'Cybersecurity Division', 'Consulting Services', 'Infrastructure Solutions'
    ]
    
    countries = ['Saudi Arabia', 'UAE', 'Kuwait', 'Qatar', 'Bahrain', 'Oman']
    
    data = []
    
    # Generate data with realistic patterns
    for i in range(n_records):
        # Generate date with seasonal patterns
        base_date = pd.Timestamp('2022-01-01')
        days_offset = np.random.randint(0, 1095)  # 3 years
        date = base_date + pd.Timedelta(days=days_offset)
        
        # Add seasonal variation (higher sales in Q4, lower in Q1)
        seasonal_multiplier = 1.0
        if date.month in [10, 11, 12]:  # Q4
            seasonal_multiplier = 1.4
        elif date.month in [1, 2, 3]:   # Q1
            seasonal_multiplier = 0.7
        elif date.month in [4, 5, 6]:   # Q2
            seasonal_multiplier = 0.9
        else:  # Q3
            seasonal_multiplier = 1.1
        
        # Generate quantities and prices with realistic ranges
        quantity = np.random.randint(1, 500)
        base_unit_price = np.random.uniform(100, 10000)
        unit_price = base_unit_price * seasonal_multiplier
        total_sales = quantity * unit_price
        
        # Generate costs with realistic margins
        cost_margin = np.random.uniform(0.45, 0.75)  # 45-75% cost ratio
        total_cost = total_sales * cost_margin
        
        # Generate discounts with realistic patterns (0-20% of sales)
        discount_rate = np.random.beta(2, 8) * 0.20
        discount = total_sales * discount_rate
        
        # Calculate profit
        profit = total_sales - total_cost - discount
        
        # Select business unit and adjust pricing accordingly
        business_unit = np.random.choice(business_units)
        
        # Different BUs have different average deal sizes
        if business_unit == 'Enterprise Services':
            unit_price *= 1.8  # Higher value services
        elif business_unit == 'Cybersecurity Division':
            unit_price *= 1.6  # Premium security solutions
        elif business_unit == 'Digital Solutions':
            unit_price *= 1.3  # Moderate premium
        elif business_unit == 'Cloud Technologies':
            unit_price *= 1.4  # Growing market premium
        
        # Recalculate dependent values
        total_sales = quantity * unit_price
        total_cost = total_sales * cost_margin
        discount = total_sales * discount_rate
        profit = total_sales - total_cost - discount
        
        data.append({
            'Customer Name': np.random.choice(customers),
            'Invoice Number': f"ACC-{date.year}-{i+1:06d}",
            'Item Description': np.random.choice(products),
            'Quantity': quantity,
            'Unit Price': round(unit_price, 2),
            'Total Sales': round(total_sales, 2),
            'Business Unit': business_unit,
            'Salesperson': np.random.choice(salespeople),
            'Total Cost': round(total_cost, 2),
            'Discount': round(discount, 2),
            'Profit': round(profit, 2),
            'Country': np.random.choice(countries),
            'Location': f"Office-{np.random.randint(1, 25):02d}",
            'Invoice Date': date
        })
    
    df = pd.DataFrame(data)
    df = clean_and_standardize_data(df)
    
    return df

@st.cache_data
def get_data_summary(df):
    """
    Get a summary of the loaded data.
    """
    
    if df.empty:
        return {}
    
    summary = {
        'total_records': len(df),
        'date_range': None,
        'total_revenue': df['Total Sales'].sum() if 'Total Sales' in df.columns else 0,
        'unique_customers': df['Customer Name'].nunique() if 'Customer Name' in df.columns else 0,
        'unique_products': df['Item Description'].nunique() if 'Item Description' in df.columns else 0,
        'business_units': df['Business Unit'].nunique() if 'Business Unit' in df.columns else 0,
        'salespeople': df['Salesperson'].nunique() if 'Salesperson' in df.columns else 0,
        'countries': df['Country'].nunique() if 'Country' in df.columns else 0,
    }
    
    if 'Invoice Date' in df.columns:
        summary['date_range'] = {
            'start': df['Invoice Date'].min(),
            'end': df['Invoice Date'].max()
        }
    
    if 'Total Cost' in df.columns:
        summary['total_cost'] = df['Total Cost'].sum()
        summary['total_profit'] = summary['total_revenue'] - summary['total_cost']
        summary['profit_margin'] = (summary['total_profit'] / summary['total_revenue'] * 100) if summary['total_revenue'] > 0 else 0
    
    return summary

def load_data():
    """
    Main data loading function that handles uploaded files or sample data.
    """
    
    # Check if there's an uploaded file in session state
    uploaded_file = st.session_state.get('uploaded_file', None)
    
    if uploaded_file is not None:
        # Load from uploaded file
        return load_and_merge_data_from_file(uploaded_file)
    else:
        # No file uploaded, use sample data
        return generate_sample_data()

def show_data_source_info():
    """
    Display information about the current data source
    """
    if st.session_state.get('uploaded_file') is not None:
        filename = st.session_state['uploaded_file'].name
        st.success(f"📊 **Data Source**: {filename}")
    else:
        st.info("📝 **Data Source**: Sample data (upload your Excel file using the sidebar)")

# =====================================
# MODULE AVAILABILITY CHECK
# =====================================

def check_module_availability():
    """Check which modules are available and return status"""
    module_status = {}
    
    modules_to_check = [
        'sales_dashboard', 
        'customer_segmentation',
        'sales_forecasting',
        'salesperson_performance',
        'discount_analysis',
        'bu_benchmarking',
        'product_insights'
    ]
    
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            module_status[module_name] = True
        except ImportError:
            module_status[module_name] = False
    
    return module_status

# =====================================
# HOME PAGE FUNCTION
# =====================================

def show_home_page():
    """Display the home page with system overview"""
    
    # Main header
    st.markdown('<h1 class="main-header">🚀 ACC Sales Intelligence System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Comprehensive sales analytics platform powered by advanced data science and machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload instructions
    if st.session_state.get('uploaded_file') is None:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem; border-left: 4px solid #2196f3;">
            <h4 style="color: #1976d2; margin-top: 0;">📁 Get Started</h4>
            <p style="margin-bottom: 0;">
                <strong>Upload your Excel file</strong> using the sidebar to analyze your actual sales data, 
                or continue with the sample data to explore the system capabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded_filename = st.session_state['uploaded_file'].name
        st.markdown(f"""
        <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem; border-left: 4px solid #4caf50;">
            <h4 style="color: #2e7d32; margin-top: 0;">✅ Data Loaded</h4>
            <p style="margin-bottom: 0;">
                Successfully analyzing data from: <strong>{uploaded_filename}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check module availability
    module_status = check_module_availability()
    
    # Load and display data overview
    with st.spinner("Loading data overview..."):
        df = load_data()
        summary = get_data_summary(df)
    
    if summary:
        # Key metrics overview
        st.subheader("📈 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #1f77b4; margin: 0;">📊</h3>
                <h2 style="margin: 0.5rem 0;">{:,}</h2>
                <p style="margin: 0; color: #666;">Total Records</p>
            </div>
            """.format(summary.get('total_records', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #2ca02c; margin: 0;">💰</h3>
                <h2 style="margin: 0.5rem 0;">${:,.0f}</h2>
                <p style="margin: 0; color: #666;">Total Revenue</p>
            </div>
            """.format(summary.get('total_revenue', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #ff7f0e; margin: 0;">👥</h3>
                <h2 style="margin: 0.5rem 0;">{:,}</h2>
                <p style="margin: 0; color: #666;">Customers</p>
            </div>
            """.format(summary.get('unique_customers', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #d62728; margin: 0;">🏢</h3>
                <h2 style="margin: 0.5rem 0;">{:,}</h2>
                <p style="margin: 0; color: #666;">Business Units</p>
            </div>
            """.format(summary.get('business_units', 0)), unsafe_allow_html=True)
        
        # Date range info
        if summary.get('date_range'):
            st.info(f"📅 Data Range: {summary['date_range']['start'].strftime('%B %Y')} - {summary['date_range']['end'].strftime('%B %Y')}")
    
    # System modules overview
    st.subheader("🎯 Available Modules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales Dashboard
        status_icon = "✅" if module_status.get('sales_dashboard', False) else "⚠️"
        st.markdown(f"""
        <div class="module-card">
            <h4>{status_icon} 📊 Sales Dashboard</h4>
            <p>Real-time sales performance monitoring with KPIs, trends, and business unit analysis.</p>
            <strong>Features:</strong> Revenue tracking, profit analysis, geographic insights
        </div>
        """, unsafe_allow_html=True)
        
        # Customer Segmentation
        status_icon = "✅" if module_status.get('customer_segmentation', False) else "⚠️"
        st.markdown(f"""
        <div class="module-card">
            <h4>{status_icon} 👥 Customer Segmentation</h4>
            <p>Advanced RFM analysis and ML-powered customer clustering for targeted strategies.</p>
            <strong>Features:</strong> RFM scoring, customer lifetime value, segment profiling
        </div>
        """, unsafe_allow_html=True)
        
        # Sales Forecasting
        status_icon = "✅" if module_status.get('sales_forecasting', False) else "⚠️"
        st.markdown(f"""
        <div class="module-card">
            <h4>{status_icon} 📈 Sales Forecasting</h4>
            <p>AI-powered forecasting using Prophet and XGBoost for accurate sales predictions.</p>
            <strong>Features:</strong> Time series analysis, seasonal trends, demand planning
        </div>
        """, unsafe_allow_html=True)
        
        # Salesperson Performance
        status_icon = "✅" if module_status.get('salesperson_performance', False) else "⚠️"
        st.markdown(f"""
        <div class="module-card">
            <h4>{status_icon} 🏆 Salesperson Performance</h4>
            <p>Comprehensive performance analytics and KPI benchmarking for sales team optimization.</p>
            <strong>Features:</strong> Performance rankings, quota analysis, territory insights
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Discount Analysis
        status_icon = "✅" if module_status.get('discount_analysis', False) else "⚠️"
        st.markdown(f"""
        <div class="module-card">
            <h4>{status_icon} 💰 Discount Analysis</h4>
            <p>Deep dive into discount strategies and their impact on profitability and customer behavior.</p>
            <strong>Features:</strong> Margin analysis, discount effectiveness, pricing optimization
        </div>
        """, unsafe_allow_html=True)
        
        # BU Benchmarking
        status_icon = "✅" if module_status.get('bu_benchmarking', False) else "⚠️"
        st.markdown(f"""
        <div class="module-card">
            <h4>{status_icon} 🏢 BU Benchmarking</h4>
            <p>Cross-business unit performance comparison with strategic recommendations.</p>
            <strong>Features:</strong> Comparative analytics, market share analysis, growth opportunities
        </div>
        """, unsafe_allow_html=True)
        
        # Product Insights
        status_icon = "✅" if module_status.get('product_insights', False) else "⚠️"
        st.markdown(f"""
        <div class="module-card">
            <h4>{status_icon} 📦 Product Insights</h4>
            <p>Product portfolio analysis with cross-sell opportunities and inventory optimization.</p>
            <strong>Features:</strong> Product lifecycle, basket analysis, recommendation engine
        </div>
        """, unsafe_allow_html=True)
    
    # Module status summary
    working_modules = sum(1 for status in module_status.values() if status)
    total_modules = len(module_status)
    
    if working_modules == total_modules:
        st.markdown(f"""
        <div class="success-container">
            <h4>🎉 All Systems Operational!</h4>
            <p>All {total_modules} modules are loaded and ready to use.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-container">
            <h4>⚠️ Module Status</h4>
            <p>{working_modules} of {total_modules} modules loaded successfully.</p>
            <p>Some modules may need to be placed in the 'modules' directory.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    with st.expander("How to use this system", expanded=False):
        st.markdown("""
        **1. Upload Your Data**: Use the sidebar file uploader to upload your Excel file
        
        **2. Navigation**: Use the sidebar to switch between different analytical modules
        
        **3. Filters**: Each module provides filtering options for business units, time periods, and other dimensions
        
        **4. Interactivity**: Click on charts and graphs to drill down into specific data points
        
        **5. Insights**: Look for the 💡 insights sections in each module for automated analysis
        
        **6. Export**: Most visualizations can be downloaded using the menu in the top-right corner of each chart
        
        **7. Data Refresh**: Upload a new file anytime to analyze different data
        """)
    
    # File format guide
    with st.expander("📁 Excel File Format Guide & Upload Test", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **📊 File Requirements:**
            
            **Supported Formats:**
            - Excel files: `.xlsx`, `.xls`
            - CSV files: `.csv`
            - Maximum file size: 200MB
            
            **Required Columns** (any of these variations):
            - `Customer Name` or `Customer`
            - `Total Sales` or `Sales` or `Revenue`
            - `Invoice Date` or `Date`
            
            **Optional Columns** (for enhanced analytics):
            - `Business Unit` or `BU`
            - `Salesperson` or `Sales Rep`
            - `Item Description` or `Product`
            - `Quantity`, `Unit Price`, `Total Cost`, `Discount`, `Profit`
            - `Country`, `Location`, `Invoice Number`
            
            **📝 Data Tips:**
            - Date formats: Excel date format or YYYY-MM-DD
            - Numbers: No special formatting needed
            - Text: Any text format is fine
            - Missing optional columns will be handled automatically
            """)
            
        with col2:
            st.markdown("**🧪 Test File Upload:**")
            
            # Quick upload test
            test_file = st.file_uploader(
                "Test upload here",
                type=['xlsx', 'xls', 'csv'],
                help="Upload a test file to check compatibility",
                key="test_uploader"
            )
            
            if test_file is not None:
                st.success("✅ Upload working!")
                st.info(f"**File:** {test_file.name}")
                st.info(f"**Size:** {test_file.size / (1024*1024):.1f} MB")
                st.info(f"**Type:** {test_file.type}")
                
                # Basic file validation
                if test_file.name.lower().endswith(('.xlsx', '.xls', '.csv')):
                    st.success("🎯 File format supported!")
                else:
                    st.error("❌ Unsupported file format")
                
                if test_file.size < 200 * 1024 * 1024:  # 200MB
                    st.success("📏 File size OK!")
                else:
                    st.error("📏 File too large (max 200MB)")
                    
                st.info("💡 If this test works, the main upload should work too!")
            else:
                st.info("🔄 Upload a test file above")
                
        # Common issues and solutions
        st.markdown("---")
        st.markdown("**🔧 Common Upload Issues:**")
        
        issues_col1, issues_col2 = st.columns(2)
        
        with issues_col1:
            st.markdown("""
            **File Selection Issues:**
            - **Can't click "Browse files"** → Refresh page, try different browser
            - **File dialog doesn't open** → Disable ad blockers, check browser permissions
            - **File appears but doesn't upload** → Check file size (<200MB)
            - **Upload button missing** → Scroll up to sidebar, check screen resolution
            """)
            
        with issues_col2:
            st.markdown("""
            **File Processing Issues:**
            - **"Error loading data"** → Check file format, ensure headers in first row
            - **"No sheets found"** → Ensure Excel file has data sheets
            - **"Empty dataframe"** → Check that file contains actual data
            - **"Encoding error"** → Save as .xlsx instead of .xls
            """)
            
        # Browser compatibility
        st.markdown("**🌐 Browser Compatibility:**")
        st.markdown("✅ **Recommended:** Chrome, Firefox, Safari, Edge (latest versions)")
        st.markdown("⚠️ **Limited support:** Internet Explorer, very old browser versions")
        
        # Sample file download
        st.markdown("---")
        st.markdown("**📥 Sample Template:**")
        
        # Create a sample Excel template for download
        sample_data = {
            'Customer Name': ['ABC Corp', 'XYZ Ltd', 'DEF Inc', 'GHI Co', 'JKL Corp'],
            'Invoice Date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
            'Total Sales': [15000, 8500, 12000, 9800, 11200],
            'Business Unit': ['Alpha', 'Beta', 'Alpha', 'Gamma', 'Beta'],
            'Salesperson': ['John Smith', 'Jane Doe', 'John Smith', 'Bob Johnson', 'Jane Doe'],
            'Item Description': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
            'Quantity': [100, 50, 80, 60, 70],
            'Unit Price': [150, 170, 150, 163, 160],
            'Total Cost': [12000, 6800, 9600, 7840, 8960],
            'Discount': [500, 200, 400, 300, 350],
            'Profit': [2500, 1500, 2000, 1660, 1890]
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        # Convert to CSV for download
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv,
            file_name="ACC_Sales_Template.csv",
            mime="text/csv"
        )

# =====================================
# MAIN APPLICATION FUNCTION
# =====================================

def main():
    """Main application function"""
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    # Module selection
    modules = {
        "🏠 Home": "home",
        "📊 Sales Dashboard": "sales_dashboard", 
        "👥 Customer Segmentation": "customer_segmentation",
        "📈 Sales Forecasting": "sales_forecasting",
        "🏆 Salesperson Performance": "salesperson_performance",
        "💰 Discount Analysis": "discount_analysis",
        "🏢 BU Benchmarking": "bu_benchmarking",
        "📦 Product Insights": "product_insights"
    }
    
    selected_module = st.sidebar.selectbox(
        "Select Module",
        list(modules.keys()),
        index=0
    )
    
    module_key = modules[selected_module]
    
    # Add some spacing
    st.sidebar.markdown("---")
    
    # File Upload Section (Optimized for large files)
    st.sidebar.subheader("📁 Data Upload")
    
    # More efficient file uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose your Excel file",
        type=['xlsx', 'xls', 'csv'],
        help="Supports: Excel (.xlsx, .xls) or CSV (.csv) | Max: 200MB",
        accept_multiple_files=False,
        key="file_uploader"
    )
    
    # Optimized file processing status
    if uploaded_file is not None:
        # Only process if it's a new file
        if st.session_state.get('uploaded_file_name') != uploaded_file.name:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.session_state['file_processed'] = False
        
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        
        # Show file info
        st.sidebar.success("✅ **File ready for processing**")
        st.sidebar.info(f"📄 **{uploaded_file.name}**")
        st.sidebar.info(f"📊 **Size:** {file_size:.1f} MB")
        
        # Performance tips for large files
        if file_size > 50:
            st.sidebar.warning("⚡ **Large file detected**")
            st.sidebar.info("💡 Processing may take 30-60 seconds")
        elif file_size > 20:
            st.sidebar.info("⚡ **Medium file** - Processing ~15-30 seconds")
        else:
            st.sidebar.info("⚡ **Small file** - Quick processing")
        
        # Show file type info
        if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            st.sidebar.info("📋 Excel file - will process all sheets")
        elif uploaded_file.name.lower().endswith('.csv'):
            st.sidebar.info("📋 CSV file - single table")
            
    elif 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
        st.session_state['uploaded_file_name'] = None
        st.sidebar.info("📝 **No file uploaded**")
        st.sidebar.info("🔄 Using sample data")
    
    # Efficient clear data function
    if st.session_state.get('uploaded_file') is not None:
        if st.sidebar.button("🗑️ Clear Data", type="secondary"):
            # Clear all file-related session state
            for key in ['uploaded_file', 'uploaded_file_name', 'file_processed']:
                if key in st.session_state:
                    del st.session_state[key]
            st.sidebar.success("✅ Data cleared")
            st.rerun()
    
    # Quick performance tips
    with st.sidebar.expander("⚡ Performance Tips", expanded=False):
        st.markdown("""
        **For Large Files (>50MB):**
        - Wait for progress bar to complete
        - Don't refresh during processing
        - Processing can take 1-2 minutes
        
        **Speed Up Processing:**
        - Use CSV format when possible
        - Remove empty sheets from Excel
        - Close other browser tabs
        - Ensure stable internet connection
        
        **Memory Usage:**
        - Files >100MB may use significant RAM
        - Close other applications if needed
        - Consider splitting very large files
        """)
    
    # Troubleshooting for slow uploads
    if st.sidebar.button("🐌 Upload Too Slow?"):
        st.sidebar.markdown("""
        **Speed up your uploads:**
        
        1. **Convert to CSV** - Much faster than Excel
        2. **Split large files** - Process in smaller chunks
        3. **Remove empty columns/rows** - Clean data first
        4. **Use Chrome/Firefox** - Better file handling
        5. **Clear browser cache** - Free up memory
        6. **Close other tabs** - Reduce RAM usage
        
        **File size recommendations:**
        - **< 20MB**: Fast processing
        - **20-50MB**: Moderate speed
        - **> 50MB**: Slower but works
        """)
    
    st.sidebar.markdown("💡 **Tip:** CSV files process 3-5x faster than Excel files")
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.subheader("⚡ System Status")
    
    # Check module status
    module_status = check_module_availability()
    working_modules = sum(1 for status in module_status.values() if status)
    total_modules = len(module_status)
    
    if working_modules == total_modules:
        st.sidebar.success("✅ All modules loaded")
    else:
        st.sidebar.warning(f"⚠️ {working_modules}/{total_modules} modules loaded")
    
    # Data status
    try:
        df = load_data()
        if not df.empty:
            if st.session_state.get('uploaded_file') is not None:
                st.sidebar.success("✅ Your data loaded")
                st.sidebar.info(f"📊 {len(df):,} records from uploaded file")
            else:
                st.sidebar.success("✅ Sample data loaded")
                st.sidebar.info(f"📊 {len(df):,} sample records")
        else:
            st.sidebar.warning("⚠️ No data available")
    except Exception as e:
        st.sidebar.error("❌ Data loading error")
        st.sidebar.caption(str(e)[:50] + "..." if len(str(e)) > 50 else str(e))
    
    # Footer info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <small>
    🏢 **ACC Sales Intelligence**<br>
    📅 Last Updated: {}<br>
    🔧 Version: 2.0.0
    </small>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
    
    # Route to appropriate module
    try:
        if module_key == "home":
            show_home_page()
        
        elif module_key == "sales_dashboard":
            try:
                import sales_dashboard
                sales_dashboard.main()
            except ImportError:
                st.error("📊 Sales Dashboard module not found")
                st.info("Please ensure `sales_dashboard.py` is in the `modules/` directory")
        
        elif module_key == "customer_segmentation":
            try:
                import customer_segmentation
                customer_segmentation.main()
            except ImportError:
                st.error("👥 Customer Segmentation module not found")
                st.info("Please ensure `customer_segmentation.py` is in the `modules/` directory")
        
        elif module_key == "sales_forecasting":
            try:
                import sales_forecasting
                sales_forecasting.main()
            except ImportError:
                st.error("📈 Sales Forecasting module not found")
                st.info("Please ensure `sales_forecasting.py` is in the `modules/` directory")
        
        elif module_key == "salesperson_performance":
            try:
                import salesperson_performance
                salesperson_performance.main()
            except ImportError:
                st.error("🏆 Salesperson Performance module not found")
                st.info("This module is under development. Please ensure `salesperson_performance.py` is in the `modules/` directory")
        
        elif module_key == "discount_analysis":
            try:
                import discount_analysis
                discount_analysis.main()
            except ImportError:
                st.error("💰 Discount Analysis module not found")
                st.info("This module is under development. Please ensure `discount_analysis.py` is in the `modules/` directory")
        
        elif module_key == "bu_benchmarking":
            try:
                import bu_benchmarking
                bu_benchmarking.main()
            except ImportError:
                st.error("🏢 BU Benchmarking module not found")
                st.info("This module is under development. Please ensure `bu_benchmarking.py` is in the `modules/` directory")
        
        elif module_key == "product_insights":
            try:
                import product_insights
                product_insights.main()
            except ImportError:
                st.error("📦 Product Insights module not found")
                st.info("This module is under development. Please ensure `product_insights.py` is in the `modules/` directory")
    
    except Exception as e:
        st.error(f"Error loading module: {str(e)}")
        st.info("Please check the module implementation or contact the system administrator.")
        
        # Show error details in expandable section
        with st.expander("🔍 Error Details"):
            st.code(str(e))
            st.markdown("**Possible Solutions:**")
            st.markdown("- Check if the module file exists in the `modules/` directory")
            st.markdown("- Verify the file name matches exactly (case-sensitive)")
            st.markdown("- Ensure the module has a `main()` function")
            st.markdown("- Check for any syntax errors in the module")

if __name__ == "__main__":
    main()
