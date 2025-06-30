import streamlit as st

st.set_page_config(page_title='ACC Sales Intelligence', layout='wide')

st.title('ACC Sales Intelligence System')

st.sidebar.title('Modules')
module = st.sidebar.radio('Select Module', [
    '1. Sales Analytics Dashboard',
    '2. Customer Segmentation',
    '3. Sales Forecasting',
    '4. Salesperson Performance',
    '5. Discount Analysis',
    '6. BU Benchmarking',
    '7. Product Insights'
])

if module == '1. Sales Analytics Dashboard':
    import modules.sales_dashboard as mod
    mod.run()
elif module == '2. Customer Segmentation':
    import modules.customer_segmentation as mod
    mod.run()
elif module == '3. Sales Forecasting':
    import modules.sales_forecasting as mod
    mod.run()
elif module == '4. Salesperson Performance':
    import modules.salesperson_performance as mod
    mod.run()
elif module == '5. Discount Analysis':
    import modules.discount_analysis as mod
    mod.run()
elif module == '6. BU Benchmarking':
    import modules.bu_benchmarking as mod
    mod.run()
elif module == '7. Product Insights':
    import modules.product_insights as mod
    mod.run()
