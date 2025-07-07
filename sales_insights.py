"""
Sales Insights Module
====================
A comprehensive module for analyzing sales data and generating key business insights.

Features:
- Revenue analysis and trends
- Customer segmentation and insights
- Product performance analysis
- Sales team performance metrics
- Business unit analysis
- Profitability insights
- Forecasting capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SalesInsights:
    def __init__(self, data_path: str):
        """Initialize the Sales Insights module with sales data."""
        self.data = pd.read_excel(data_path)
        self.data['Invoice Date'] = pd.to_datetime(self.data['Invoice Date'])
        self.data['Year'] = self.data['Invoice Date'].dt.year
        self.data['Month'] = self.data['Invoice Date'].dt.month
        self.data['Quarter'] = self.data['Invoice Date'].dt.quarter
        self.data['Profit'] = self.data['Total Line Amount'] - self.data['Total Cost']
        self.data['Profit Margin'] = (self.data['Profit'] / self.data['Total Line Amount']) * 100
        
    # ====================== REVENUE ANALYSIS ======================
    
    def revenue_overview(self) -> Dict:
        """Generate comprehensive revenue overview."""
        total_revenue = self.data['Total Line Amount'].sum()
        total_cost = self.data['Total Cost'].sum()
        total_profit = total_revenue - total_cost
        profit_margin = (total_profit / total_revenue) * 100
        
        avg_order_value = self.data.groupby('Invoice No.')['Total Line Amount'].sum().mean()
        total_orders = self.data['Invoice No.'].nunique()
        total_customers = self.data['Cust No.'].nunique()
        
        return {
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'profit_margin_pct': profit_margin,
            'avg_order_value': avg_order_value,
            'total_orders': total_orders,
            'total_customers': total_customers,
            'revenue_per_customer': total_revenue / total_customers
        }
    
    def monthly_revenue_trend(self) -> pd.DataFrame:
        """Analyze monthly revenue trends."""
        monthly_data = self.data.groupby(['Year', 'Month']).agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Profit': 'sum',
            'Invoice No.': 'nunique',
            'QTY': 'sum'
        }).reset_index()
        
        monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
        monthly_data['Growth_Rate'] = monthly_data['Total Line Amount'].pct_change() * 100
        
        return monthly_data.sort_values('Date')
    
    def quarterly_performance(self) -> pd.DataFrame:
        """Analyze quarterly performance."""
        return self.data.groupby(['Year', 'Quarter']).agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Profit': 'sum',
            'Profit Margin': 'mean',
            'Invoice No.': 'nunique',
            'Cust No.': 'nunique'
        }).reset_index()
    
    # ====================== CUSTOMER ANALYSIS ======================
    
    def top_customers_analysis(self, top_n: int = 20) -> pd.DataFrame:
        """Identify top customers by revenue and other metrics."""
        customer_metrics = self.data.groupby(['Cust No.', 'Cust Name']).agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Profit': 'sum',
            'Invoice No.': 'nunique',
            'QTY': 'sum',
            'Invoice Date': ['min', 'max']
        }).reset_index()
        
        customer_metrics.columns = ['Cust_No', 'Customer_Name', 'Total_Revenue', 'Total_Cost', 
                                  'Total_Profit', 'Order_Count', 'Total_Quantity', 
                                  'First_Order', 'Last_Order']
        
        customer_metrics['Avg_Order_Value'] = customer_metrics['Total_Revenue'] / customer_metrics['Order_Count']
        customer_metrics['Profit_Margin'] = (customer_metrics['Total_Profit'] / customer_metrics['Total_Revenue']) * 100
        customer_metrics['Customer_Lifetime_Days'] = (customer_metrics['Last_Order'] - customer_metrics['First_Order']).dt.days
        
        return customer_metrics.nlargest(top_n, 'Total_Revenue')
    
    def customer_segmentation(self) -> pd.DataFrame:
        """Segment customers using RFM analysis."""
        # Calculate RFM metrics
        today = self.data['Invoice Date'].max()
        
        rfm = self.data.groupby('Cust No.').agg({
            'Invoice Date': lambda x: (today - x.max()).days,  # Recency
            'Invoice No.': 'nunique',  # Frequency
            'Total Line Amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['Cust_No', 'Recency', 'Frequency', 'Monetary']
        
        # Create quintiles for each metric
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Combine scores
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # Define customer segments
        def segment_customers(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['533', '532', '531', '523', '522', '521', '515', '514', '513', '425','424', '413','414','415', '315', '314', '313']:
                return 'New Customers'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'Champions'
            elif row['RFM_Score'] in ['331', '321', '231', '241', '251']:
                return 'Need Attention'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115']:
                return 'Cannot Lose Them'
            elif row['RFM_Score'] in ['332', '322', '231', '241', '251', '233', '232']:
                return 'At Risk'
            else:
                return 'Others'
        
        rfm['Segment'] = rfm.apply(segment_customers, axis=1)
        return rfm
    
    def customer_churn_risk(self) -> pd.DataFrame:
        """Identify customers at risk of churn."""
        last_purchase = self.data.groupby('Cust No.')['Invoice Date'].max().reset_index()
        last_purchase['Days_Since_Last_Purchase'] = (datetime.now() - last_purchase['Invoice Date']).dt.days
        
        # Define risk levels based on days since last purchase
        def risk_level(days):
            if days <= 30:
                return 'Low Risk'
            elif days <= 90:
                return 'Medium Risk'
            elif days <= 180:
                return 'High Risk'
            else:
                return 'Critical Risk'
        
        last_purchase['Churn_Risk'] = last_purchase['Days_Since_Last_Purchase'].apply(risk_level)
        
        return last_purchase.merge(
            self.data.groupby('Cust No.').agg({
                'Cust Name': 'first',
                'Total Line Amount': 'sum',
                'Invoice No.': 'nunique'
            }).reset_index(),
            on='Cust No.'
        )
    
    # ====================== PRODUCT ANALYSIS ======================
    
    def product_performance(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze product performance."""
        product_data = self.data.groupby(['Item Number', 'Line Description', 'Brand']).agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Profit': 'sum',
            'QTY': 'sum',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        product_data['Profit_Margin'] = (product_data['Profit'] / product_data['Total Line Amount']) * 100
        product_data['Avg_Unit_Price'] = product_data['Total Line Amount'] / product_data['QTY']
        
        return product_data.nlargest(top_n, 'Total Line Amount')
    
    def brand_analysis(self) -> pd.DataFrame:
        """Analyze performance by brand."""
        return self.data.groupby('Brand').agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Profit': 'sum',
            'QTY': 'sum',
            'Invoice No.': 'nunique',
            'Cust No.': 'nunique'
        }).reset_index().sort_values('Total Line Amount', ascending=False)
    
    # ====================== SALES TEAM ANALYSIS ======================
    
    def salesperson_performance(self) -> pd.DataFrame:
        """Analyze sales team performance."""
        sales_performance = self.data.groupby('Salesperson Name').agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Profit': 'sum',
            'Invoice No.': 'nunique',
            'Cust No.': 'nunique',
            'QTY': 'sum'
        }).reset_index()
        
        sales_performance['Profit_Margin'] = (sales_performance['Profit'] / sales_performance['Total Line Amount']) * 100
        sales_performance['Avg_Deal_Size'] = sales_performance['Total Line Amount'] / sales_performance['Invoice No.']
        sales_performance['Customer_Coverage'] = sales_performance['Cust No.']
        
        return sales_performance.sort_values('Total Line Amount', ascending=False)
    
    def sales_efficiency_metrics(self) -> pd.DataFrame:
        """Calculate sales efficiency metrics by salesperson."""
        monthly_sales = self.data.groupby(['Salesperson Name', 'Year', 'Month']).agg({
            'Total Line Amount': 'sum',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        efficiency_metrics = monthly_sales.groupby('Salesperson Name').agg({
            'Total Line Amount': ['mean', 'std'],
            'Invoice No.': 'mean'
        }).reset_index()
        
        efficiency_metrics.columns = ['Salesperson', 'Avg_Monthly_Revenue', 'Revenue_Consistency', 'Avg_Monthly_Orders']
        efficiency_metrics['Consistency_Score'] = 100 - (efficiency_metrics['Revenue_Consistency'] / efficiency_metrics['Avg_Monthly_Revenue'] * 100)
        
        return efficiency_metrics
    
    # ====================== BUSINESS UNIT ANALYSIS ======================
    
    def business_unit_performance(self) -> pd.DataFrame:
        """Analyze performance by business unit."""
        bu_performance = self.data.groupby(['BU', 'BU Name']).agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Profit': 'sum',
            'Invoice No.': 'nunique',
            'Cust No.': 'nunique',
            'QTY': 'sum'
        }).reset_index()
        
        bu_performance['Profit_Margin'] = (bu_performance['Profit'] / bu_performance['Total Line Amount']) * 100
        bu_performance['Market_Share'] = (bu_performance['Total Line Amount'] / bu_performance['Total Line Amount'].sum()) * 100
        
        return bu_performance.sort_values('Total Line Amount', ascending=False)
    
    def geographical_analysis(self) -> pd.DataFrame:
        """Analyze sales by geography."""
        geo_analysis = self.data.groupby(['Country', 'Local']).agg({
            'Total Line Amount': 'sum',
            'Total Cost': 'sum',
            'Profit': 'sum',
            'Invoice No.': 'nunique',
            'Cust No.': 'nunique'
        }).reset_index()
        
        geo_analysis['Profit_Margin'] = (geo_analysis['Profit'] / geo_analysis['Total Line Amount']) * 100
        
        return geo_analysis.sort_values('Total Line Amount', ascending=False)
    
    # ====================== ADVANCED ANALYTICS ======================
    
    def payment_terms_analysis(self) -> pd.DataFrame:
        """Analyze performance by payment terms."""
        return self.data.groupby('Term Name').agg({
            'Total Line Amount': 'sum',
            'Applied Amount': 'sum',
            'Invoice No.': 'nunique',
            'Cust No.': 'nunique'
        }).reset_index()
    
    def seasonal_analysis(self) -> pd.DataFrame:
        """Analyze seasonal trends."""
        self.data['Month_Name'] = self.data['Invoice Date'].dt.month_name()
        
        seasonal_data = self.data.groupby('Month_Name').agg({
            'Total Line Amount': 'sum',
            'Profit': 'sum',
            'Invoice No.': 'nunique'
        }).reset_index()
        
        # Add month number for proper ordering
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        seasonal_data['Month_Num'] = seasonal_data['Month_Name'].apply(lambda x: month_order.index(x) + 1)
        
        return seasonal_data.sort_values('Month_Num')
    
    def generate_executive_summary(self) -> Dict:
        """Generate executive summary with key insights."""
        overview = self.revenue_overview()
        top_customers = self.top_customers_analysis(5)
        top_products = self.product_performance(5)
        sales_team = self.salesperson_performance()
        
        return {
            'period_covered': f"{self.data['Invoice Date'].min().strftime('%Y-%m-%d')} to {self.data['Invoice Date'].max().strftime('%Y-%m-%d')}",
            'total_revenue': f"${overview['total_revenue']:,.2f}",
            'total_profit': f"${overview['total_profit']:,.2f}",
            'profit_margin': f"{overview['profit_margin_pct']:.1f}%",
            'top_customer': top_customers.iloc[0]['Customer_Name'],
            'top_customer_revenue': f"${top_customers.iloc[0]['Total_Revenue']:,.2f}",
            'best_salesperson': sales_team.iloc[0]['Salesperson Name'],
            'best_salesperson_revenue': f"${sales_team.iloc[0]['Total Line Amount']:,.2f}",
            'total_orders': overview['total_orders'],
            'avg_order_value': f"${overview['avg_order_value']:,.2f}",
            'unique_customers': overview['total_customers']
        }
    
    # ====================== VISUALIZATION HELPERS ======================
    
    def plot_revenue_trend(self, save_path: Optional[str] = None):
        """Plot monthly revenue trend."""
        monthly_data = self.monthly_revenue_trend()
        
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_data['Date'], monthly_data['Total Line Amount'], marker='o', linewidth=2)
        plt.title('Monthly Revenue Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Revenue ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_customer_segmentation(self, save_path: Optional[str] = None):
        """Plot customer segmentation distribution."""
        segments = self.customer_segmentation()
        segment_counts = segments['Segment'].value_counts()
        
        plt.figure(figsize=(10, 6))
        plt.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Customer Segmentation Distribution', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ====================== USAGE EXAMPLE ======================

def main():
    """Example usage of the Sales Insights module."""
    
    # Initialize the insights module
    insights = SalesInsights('sample.xlsx')
    
    # Generate executive summary
    summary = insights.generate_executive_summary()
    print("=== EXECUTIVE SUMMARY ===")
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Get revenue overview
    print("\n=== REVENUE OVERVIEW ===")
    revenue_data = insights.revenue_overview()
    for key, value in revenue_data.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Top customers analysis
    print("\n=== TOP 10 CUSTOMERS ===")
    top_customers = insights.top_customers_analysis(10)
    print(top_customers[['Customer_Name', 'Total_Revenue', 'Order_Count', 'Profit_Margin']].to_string(index=False))
    
    # Sales team performance
    print("\n=== SALES TEAM PERFORMANCE ===")
    sales_performance = insights.salesperson_performance()
    print(sales_performance[['Salesperson Name', 'Total Line Amount', 'Invoice No.', 'Profit_Margin']].head(10).to_string(index=False))


if __name__ == "__main__":
    main()