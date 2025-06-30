import streamlit as st
import pandas as pd
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

def check_module_availability():
    """Check which modules are available and return status"""
    module_status = {}
    
    modules_to_check = [
        'data_loader',
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

def load_sample_data():
    """Load sample data for the home page overview"""
    try:
        import data_loader
        df = data_loader.load_and_merge_data()
        summary = data_loader.get_data_summary(df)
        return df, summary
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), {}

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
    
    # Check module availability
    module_status = check_module_availability()
    
    # Load and display data overview
    with st.spinner("Loading data overview..."):
        df, summary = load_sample_data()
    
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
        **1. Navigation**: Use the sidebar to switch between different analytical modules
        
        **2. Filters**: Each module provides filtering options for business units, time periods, and other dimensions
        
        **3. Interactivity**: Click on charts and graphs to drill down into specific data points
        
        **4. Insights**: Look for the 💡 insights sections in each module for automated analysis
        
        **5. Export**: Most visualizations can be downloaded using the menu in the top-right corner of each chart
        
        **6. Data Refresh**: The system automatically caches data for performance, but you can refresh by reloading the page
        """)
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting", expanded=False):
        st.markdown("""
        **Common Issues:**
        
        **📁 Module Not Found Errors:**
        - Ensure all `.py` files are in the `modules/` directory
        - Make sure `modules/__init__.py` exists (can be empty)
        - Check that file names match exactly (case-sensitive)
        
        **📊 Data Loading Issues:**
        - Verify `data/ACC Sales 22~24.xlsx` exists
        - Check Excel file has the expected sheet names
        - Ensure data has required columns (Customer Name, Total Sales, etc.)
        
        **📈 Library Missing Errors:**
        - Install missing packages: `pip install -r requirements.txt`
        - For advanced forecasting: `pip install prophet xgboost`
        
        **🔄 Performance Issues:**
        - Clear browser cache and reload
        - Restart Streamlit server
        - Check available system memory
        """)

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
        import data_loader
        df = data_loader.load_and_merge_data()
        if not df.empty:
            st.sidebar.success("✅ Data loaded successfully")
            st.sidebar.info(f"📊 {len(df):,} records available")
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
    🔧 Version: 1.0.0
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
