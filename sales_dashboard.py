import streamlit as st
import pandas as pd
import plotly.express as px

def run():
    st.header("Sales Analytics Dashboard")
    st.write("This module will show total sales, costs, and margins by BU, salesperson, customer, and more.")
    
    # Example demo dataframe
    df = pd.DataFrame({
        'Business Unit': ['A', 'B', 'C'],
        'Total Sales': [100000, 150000, 120000],
        'Total COGS': [60000, 80000, 75000]
    })
    df['Profit'] = df['Total Sales'] - df['Total COGS']

    fig = px.bar(df, x='Business Unit', y='Profit', title="Profit by Business Unit")
    st.plotly_chart(fig)
