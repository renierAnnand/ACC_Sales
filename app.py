import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

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
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample data for the home page overview"""
    try:
        from modules.data_loader import load_and_merge_data, get_data_summary
        df = load_and_merge_data()
        summary = get_data_summary(df)
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
        st.markdown("""
        <div class="module-card">
            <h4>📊 Sales Dashboard</h4>
            <p>Real-time sales performance monitoring with KPIs, trends, and business unit analysis.</p>
            <strong>Features:</strong> Revenue tracking, profit analysis, geographic insights
        </div>
        
        <div class="module-card">
            <h4>👥 Customer Segmentation</h4>
            <p>Advanced RFM analysis and ML-powered customer clustering for targeted strategies.</p>
            <strong>Features:</strong> RFM scoring, customer lifetime value, segment profiling
        </div>
        
        <div class="module-card">
            <h4>📈 Sales Forecasting</h4>
            <p>AI-powered forecasting using Prophet and XGBoost for accurate sales predictions.</p>
            <strong>Features:</strong> Time series analysis, seasonal trends, demand planning
        </div>
        
        <div class="module-card">
            <h4>🏆 Salesperson Performance</h4>
            <p>Comprehensive performance analytics and KPI benchmarking for sales team optimization.</p>
            <strong>Features:</strong> Performance rankings, quota analysis, territory insights
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <h4>💰 Discount Analysis</h4>
            <p>Deep dive into discount strategies and their impact on profitability and customer behavior.</p>
            <strong>Features:</strong> Margin analysis, discount effectiveness, pricing optimization
        </div>
        
        <div class="module-card">
            <h4>🏢 BU Benchmarking</h4>
            <p>Cross-business unit performance comparison with strategic recommendations.</p>
            <strong>Features:</strong> Comparative analytics, market share analysis, growth opportunities
        </div>
        
        <div class="module-card">
            <h4>📦 Product Insights</h4>
            <p>Product portfolio analysis with cross-sell opportunities and inventory optimization.</p>
            <strong>Features:</strong> Product lifecycle, basket analysis, recommendation engine
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
    
    # System information
    with st.expander("System Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Data Sources:**
            - ACC Sales 2022-2024 Excel Files
            - Multiple sheets with comprehensive sales data
            - Real-time data processing and validation
            """)
        
        with col2:
            st.markdown("""
            **🔧 Technology Stack:**
            - Streamlit for interactive web interface
            - Pandas for data processing
            - Plotly for interactive visualizations
            - Scikit-learn & Prophet for ML models
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
    try:
        from modules.data_loader import load_and_merge_data
        df = load_and_merge_data()
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
            from modules.sales_dashboard import main as dashboard_main
            dashboard_main()
        
        elif module_key == "customer_segmentation":
            try:
                from modules.customer_segmentation import main as segmentation_main
                segmentation_main()
            except ImportError:
                st.error("Customer Segmentation module not yet implemented")
                st.info("This module is under development. Please select another module.")
        
        elif module_key == "sales_forecasting":
            try:
                from modules.sales_forecasting import main as forecasting_main
                forecasting_main()
            except ImportError:
                st.error("Sales Forecasting module not yet implemented")
                st.info("This module is under development. Please select another module.")
        
        elif module_key == "salesperson_performance":
            try:
                from modules.salesperson_performance import main as performance_main
                performance_main()
            except ImportError:
                st.error("Salesperson Performance module not yet implemented")
                st.info("This module is under development. Please select another module.")
        
        elif module_key == "discount_analysis":
            try:
                from modules.discount_analysis import main as discount_main
                discount_main()
            except ImportError:
                st.error("Discount Analysis module not yet implemented")
                st.info("This module is under development. Please select another module.")
        
        elif module_key == "bu_benchmarking":
            try:
                from modules.bu_benchmarking import main as benchmarking_main
                benchmarking_main()
            except ImportError:
                st.error("BU Benchmarking module not yet implemented")
                st.info("This module is under development. Please select another module.")
        
        elif module_key == "product_insights":
            try:
                from modules.product_insights import main as insights_main
                insights_main()
            except ImportError:
                st.error("Product Insights module not yet implemented")
                st.info("This module is under development. Please select another module.")
    
    except Exception as e:
        st.error(f"Error loading module: {str(e)}")
        st.info("Please check the module implementation or contact the system administrator.")
        
        # Show error details in expandable section
        with st.expander("Error Details"):
            st.code(str(e))

if __name__ == "__main__":
    main()
