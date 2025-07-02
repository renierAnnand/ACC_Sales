#!/usr/bin/env python3
"""
ACC Sales Intelligence System - Customer Segmentation Analysis
Advanced Analytics Dashboard for Sales Performance & Business Intelligence

This module handles customer segmentation with proper timezone management
and Saudi Riyal currency formatting.

Author: ACC Sales Intelligence Team
Date: July 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from dataclasses import dataclass
import json

# Optional imports - will work without sklearn and seaborn
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not available - clustering features disabled")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - plotting features disabled")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠️  seaborn not available - advanced plotting disabled")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Saudi Arabia timezone
SAUDI_TZ = pytz.timezone('Asia/Riyadh')
UTC_TZ = pytz.UTC

@dataclass
class CustomerMetrics:
    """Data class for customer metrics"""
    customer_id: str
    recency_days: int
    frequency: int
    monetary_sar: float
    avg_order_value_sar: float
    total_orders: int
    days_since_first_purchase: int
    segment: str = ""

class DateTimeHandler:
    """
    Utility class to handle timezone-aware and timezone-naive datetime operations
    Fixes the common error: Cannot subtract tz-naive and tz-aware datetime-like objects
    """
    
    @staticmethod
    def normalize_datetime(dt: Union[datetime, pd.Timestamp, str], target_tz: pytz.BaseTzInfo = SAUDI_TZ) -> datetime:
        """
        Normalize datetime to specified timezone, handling both naive and aware datetimes
        
        Args:
            dt: Input datetime (can be naive, aware, string, or pandas Timestamp)
            target_tz: Target timezone (default: Saudi Arabia)
            
        Returns:
            Timezone-aware datetime in target timezone
        """
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        
        if dt is None:
            return None
            
        # If datetime is naive, assume it's in target timezone
        if dt.tzinfo is None:
            dt = target_tz.localize(dt)
        else:
            # Convert to target timezone
            dt = dt.astimezone(target_tz)
            
        return dt
    
    @staticmethod
    def make_naive(dt: datetime) -> datetime:
        """Convert timezone-aware datetime to naive datetime"""
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt
    
    @staticmethod
    def calculate_days_between(start_dt: datetime, end_dt: datetime) -> int:
        """
        Calculate days between two datetimes, handling timezone issues
        
        Args:
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            Number of days between dates
        """
        try:
            # Normalize both datetimes to the same timezone
            start_normalized = DateTimeHandler.normalize_datetime(start_dt)
            end_normalized = DateTimeHandler.normalize_datetime(end_dt)
            
            # Calculate difference
            diff = end_normalized - start_normalized
            return abs(diff.days)
            
        except Exception as e:
            logger.error(f"Error calculating days between dates: {e}")
            # Fallback: convert to naive datetimes
            start_naive = DateTimeHandler.make_naive(DateTimeHandler.normalize_datetime(start_dt))
            end_naive = DateTimeHandler.make_naive(DateTimeHandler.normalize_datetime(end_dt))
            diff = end_naive - start_naive
            return abs(diff.days)

class CurrencyFormatter:
    """Handle Saudi Riyal currency formatting"""
    
    @staticmethod
    def format_sar(amount: float, decimals: int = 2) -> str:
        """Format amount in Saudi Riyal"""
        return f"SAR {amount:,.{decimals}f}"
    
    @staticmethod
    def format_sar_short(amount: float) -> str:
        """Format large amounts with K/M abbreviations"""
        if amount >= 1000000:
            return f"SAR {amount/1000000:.1f}M"
        elif amount >= 1000:
            return f"SAR {amount/1000:.1f}K"
        else:
            return f"SAR {amount:.0f}"

class CustomerSegmentation:
    """
    Main customer segmentation class with advanced analytics
    """
    
    def __init__(self, timezone: str = 'Asia/Riyadh'):
        self.timezone = pytz.timezone(timezone)
        self.dt_handler = DateTimeHandler()
        self.currency_formatter = CurrencyFormatter()
        self.segments = {}
        self.customer_metrics = []
        
    def load_sample_data(self) -> pd.DataFrame:
        """
        Generate sample customer data for demonstration
        In production, replace this with your actual data loading logic
        """
        np.random.seed(42)
        n_customers = 1000
        
        # Generate sample data with mixed timezone scenarios
        current_time = datetime.now(self.timezone)
        
        customers = []
        for i in range(n_customers):
            customer_id = f"CUST_{i+1:04d}"
            
            # Random dates - some timezone-aware, some naive (common real-world scenario)
            days_since_first = np.random.randint(30, 365*2)
            first_purchase = current_time - timedelta(days=days_since_first)
            
            # Sometimes create naive datetime (common data issue)
            if np.random.random() > 0.7:
                first_purchase = first_purchase.replace(tzinfo=None)
            
            last_purchase_days = np.random.randint(1, min(days_since_first, 90))
            last_purchase = current_time - timedelta(days=last_purchase_days)
            
            # Random order data
            total_orders = np.random.poisson(5) + 1
            total_spent = np.random.exponential(1000) * np.random.uniform(0.5, 3.0)
            
            customers.append({
                'customer_id': customer_id,
                'first_purchase_date': first_purchase,
                'last_purchase_date': last_purchase,
                'total_orders': total_orders,
                'total_spent_sar': total_spent,
                'avg_order_value_sar': total_spent / total_orders,
                'customer_lifetime_days': days_since_first
            })
        
        return pd.DataFrame(customers)
    
    def calculate_rfm_metrics(self, df: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics with proper datetime handling
        
        Args:
            df: Customer data DataFrame
            reference_date: Reference date for recency calculation (default: now)
            
        Returns:
            DataFrame with RFM metrics
        """
        if reference_date is None:
            reference_date = datetime.now(self.timezone)
        else:
            reference_date = self.dt_handler.normalize_datetime(reference_date)
        
        logger.info("Calculating RFM metrics...")
        
        rfm_data = []
        
        for _, row in df.iterrows():
            try:
                # Handle datetime operations safely
                last_purchase = row['last_purchase_date']
                first_purchase = row['first_purchase_date']
                
                # Calculate recency (days since last purchase)
                recency_days = self.dt_handler.calculate_days_between(last_purchase, reference_date)
                
                # Calculate customer lifetime
                lifetime_days = self.dt_handler.calculate_days_between(first_purchase, reference_date)
                
                # Create customer metrics
                metrics = CustomerMetrics(
                    customer_id=row['customer_id'],
                    recency_days=recency_days,
                    frequency=row['total_orders'],
                    monetary_sar=row['total_spent_sar'],
                    avg_order_value_sar=row['avg_order_value_sar'],
                    total_orders=row['total_orders'],
                    days_since_first_purchase=lifetime_days
                )
                
                rfm_data.append({
                    'customer_id': metrics.customer_id,
                    'recency_days': metrics.recency_days,
                    'frequency': metrics.frequency,
                    'monetary_sar': metrics.monetary_sar,
                    'avg_order_value_sar': metrics.avg_order_value_sar,
                    'customer_lifetime_days': metrics.days_since_first_purchase
                })
                
                self.customer_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error processing customer {row['customer_id']}: {e}")
                continue
        
        return pd.DataFrame(rfm_data)
    
    def assign_rfm_scores(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign RFM scores (1-5) based on quintiles
        """
        logger.info("Assigning RFM scores...")
        
        # Calculate quintiles for scoring (1 = worst, 5 = best)
        rfm_df['R_Score'] = pd.qcut(rfm_df['recency_days'], 5, labels=[5,4,3,2,1])  # Lower recency = better
        rfm_df['F_Score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])  # Higher frequency = better
        rfm_df['M_Score'] = pd.qcut(rfm_df['monetary_sar'], 5, labels=[1,2,3,4,5])  # Higher monetary = better
        
        # Create RFM segment
        rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
        
        return rfm_df
    
    def create_business_segments(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create business-meaningful customer segments based on RFM scores
        """
        logger.info("Creating business segments...")
        
        def assign_segment(row):
            R, F, M = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
            
            # Champions: High value, frequent, recent customers
            if R >= 4 and F >= 4 and M >= 4:
                return "Champions"
            
            # Loyal Customers: High frequency, good monetary
            elif F >= 4 and M >= 3:
                return "Loyal Customers"
            
            # Potential Loyalists: Recent customers with potential
            elif R >= 4 and F >= 2 and M >= 2:
                return "Potential Loyalists"
            
            # New Customers: Recent but low frequency/monetary
            elif R >= 4 and F <= 2:
                return "New Customers"
            
            # Promising: Recent, decent monetary but low frequency
            elif R >= 3 and M >= 3 and F <= 2:
                return "Promising"
            
            # Need Attention: Declining recent customers
            elif R >= 3 and F >= 3 and M >= 3:
                return "Need Attention"
            
            # About to Sleep: Declining recency but were good customers
            elif R <= 3 and F >= 3 and M >= 3:
                return "About to Sleep"
            
            # At Risk: Were good customers but haven't purchased recently
            elif R <= 2 and F >= 3 and M >= 3:
                return "At Risk"
            
            # Cannot Lose Them: High value but very low recency
            elif R <= 2 and F >= 4 and M >= 4:
                return "Cannot Lose Them"
            
            # Hibernating: Low recency, frequency, but decent monetary
            elif R <= 2 and F <= 2 and M >= 3:
                return "Hibernating"
            
            # Lost: Very low scores across all metrics
            else:
                return "Lost"
        
        rfm_df['Segment'] = rfm_df.apply(assign_segment, axis=1)
        return rfm_df
    
    def perform_kmeans_clustering(self, rfm_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Perform K-means clustering on RFM data (if sklearn is available)
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available - skipping clustering analysis")
            # Create simple quintile-based clusters as fallback
            rfm_df['Cluster'] = pd.qcut(rfm_df['monetary_sar'], n_clusters, labels=range(n_clusters))
            logger.info(f"Using monetary value quintiles for {n_clusters} clusters")
            return rfm_df
        
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        # Prepare data for clustering
        features = ['recency_days', 'frequency', 'monetary_sar']
        X = rfm_df[features].copy()
        
        # Handle any missing values
        X = X.fillna(X.median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        rfm_df['Cluster'] = cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        logger.info(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return rfm_df
    
    def generate_segment_summary(self, rfm_df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive segment summary with Saudi Riyal formatting
        """
        logger.info("Generating segment summary...")
        
        summary = {}
        
        # Business segments summary
        segment_summary = rfm_df.groupby('Segment').agg({
            'customer_id': 'count',
            'recency_days': 'mean',
            'frequency': 'mean',
            'monetary_sar': ['mean', 'sum'],
            'avg_order_value_sar': 'mean'
        }).round(2)
        
        segment_summary.columns = ['Count', 'Avg_Recency_Days', 'Avg_Frequency', 'Avg_Monetary_SAR', 'Total_Revenue_SAR', 'Avg_Order_Value_SAR']
        
        # Add percentage of total customers
        total_customers = len(rfm_df)
        segment_summary['Percentage'] = (segment_summary['Count'] / total_customers * 100).round(2)
        
        # Format currency columns
        currency_cols = ['Avg_Monetary_SAR', 'Total_Revenue_SAR', 'Avg_Order_Value_SAR']
        for col in currency_cols:
            segment_summary[f'{col}_Formatted'] = segment_summary[col].apply(self.currency_formatter.format_sar)
        
        summary['business_segments'] = segment_summary
        
        # Overall statistics
        summary['total_customers'] = total_customers
        summary['total_revenue_sar'] = rfm_df['monetary_sar'].sum()
        summary['avg_customer_value_sar'] = rfm_df['monetary_sar'].mean()
        summary['total_orders'] = rfm_df['frequency'].sum()
        
        # Format overall stats
        summary['total_revenue_sar_formatted'] = self.currency_formatter.format_sar(summary['total_revenue_sar'])
        summary['avg_customer_value_sar_formatted'] = self.currency_formatter.format_sar(summary['avg_customer_value_sar'])
        
        return summary
    
    def create_segment_recommendations(self, summary: Dict) -> Dict[str, List[str]]:
        """
        Create actionable recommendations for each segment
        """
        recommendations = {
            "Champions": [
                "Reward with exclusive offers and early access to new products",
                "Request referrals and testimonials",
                "Implement VIP customer service",
                "Cross-sell premium products"
            ],
            "Loyal Customers": [
                "Maintain engagement with regular communication",
                "Offer loyalty program benefits",
                "Suggest complementary products",
                "Prevent churn with personalized offers"
            ],
            "Potential Loyalists": [
                "Encourage more frequent purchases with incentives",
                "Provide product education and recommendations",
                "Offer membership programs",
                "Send targeted promotions"
            ],
            "New Customers": [
                "Welcome series with onboarding content",
                "Introduce product range gradually",
                "Offer new customer discounts for repeat purchases",
                "Collect feedback to improve experience"
            ],
            "Promising": [
                "Increase purchase frequency with targeted campaigns",
                "Offer bundle deals and volume discounts",
                "Share success stories and use cases",
                "Provide exceptional customer service"
            ],
            "Need Attention": [
                "Send personalized offers to re-engage",
                "Investigate reasons for declining activity",
                "Offer customer service support",
                "Create win-back campaigns"
            ],
            "About to Sleep": [
                "Immediate re-engagement campaigns",
                "Special discounts and limited-time offers",
                "Survey to understand issues",
                "Personalized recommendations"
            ],
            "At Risk": [
                "Urgent retention campaigns",
                "Deep discounts and free shipping",
                "Personal outreach from account managers",
                "Address service issues immediately"
            ],
            "Cannot Lose Them": [
                "Priority customer service intervention",
                "Exclusive offers they cannot refuse",
                "Personal contact from senior management",
                "Investigate and resolve any issues"
            ],
            "Hibernating": [
                "Strong win-back campaigns",
                "Show new product innovations",
                "Limited-time reactivation offers",
                "Survey for feedback and improvements"
            ],
            "Lost": [
                "Final win-back attempt with best offers",
                "Feedback collection for future improvements",
                "Remove from active marketing to reduce costs",
                "Focus resources on more promising segments"
            ]
        }
        
        return recommendations
    
    def run_complete_analysis(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run complete customer segmentation analysis
        
        Args:
            df: Customer data DataFrame (if None, uses sample data)
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info("Starting complete customer segmentation analysis...")
            
            # Load data
            if df is None:
                logger.info("Loading sample data...")
                df = self.load_sample_data()
            
            # Calculate RFM metrics
            rfm_df = self.calculate_rfm_metrics(df)
            
            # Assign RFM scores
            rfm_df = self.assign_rfm_scores(rfm_df)
            
            # Create business segments
            rfm_df = self.create_business_segments(rfm_df)
            
            # Perform clustering
            rfm_df = self.perform_kmeans_clustering(rfm_df)
            
            # Generate summary
            summary = self.generate_segment_summary(rfm_df)
            
            # Create recommendations
            recommendations = self.create_segment_recommendations(summary)
            
            # Compile results
            results = {
                'rfm_data': rfm_df,
                'summary': summary,
                'recommendations': recommendations,
                'analysis_date': datetime.now(self.timezone).isoformat(),
                'total_customers_analyzed': len(rfm_df)
            }
            
            logger.info("Customer segmentation analysis completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise
    
    def print_analysis_report(self, results: Dict):
        """
        Print formatted analysis report
        """
        print("\n" + "="*80)
        print("🏢 ACC SALES INTELLIGENCE SYSTEM")
        print("📊 CUSTOMER SEGMENTATION ANALYSIS REPORT")
        print("="*80)
        
        summary = results['summary']
        
        print(f"\n📈 OVERVIEW")
        print(f"Analysis Date: {results['analysis_date']}")
        print(f"Total Customers Analyzed: {summary['total_customers']:,}")
        print(f"Total Revenue: {summary['total_revenue_sar_formatted']}")
        print(f"Average Customer Value: {summary['avg_customer_value_sar_formatted']}")
        print(f"Total Orders: {summary['total_orders']:,}")
        
        print(f"\n🎯 CUSTOMER SEGMENTS")
        print("-" * 80)
        
        segments_df = summary['business_segments']
        for segment, data in segments_df.iterrows():
            print(f"\n{segment.upper()}")
            print(f"  • Customers: {data['Count']:,} ({data['Percentage']:.1f}%)")
            print(f"  • Avg Days Since Last Purchase: {data['Avg_Recency_Days']:.0f}")
            print(f"  • Avg Orders per Customer: {data['Avg_Frequency']:.1f}")
            print(f"  • Avg Customer Value: {data['Avg_Monetary_SAR_Formatted']}")
            print(f"  • Total Segment Revenue: {data['Total_Revenue_SAR_Formatted']}")
            print(f"  • Avg Order Value: {data['Avg_Order_Value_SAR_Formatted']}")
        
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 80)
        
        recommendations = results['recommendations']
        for segment, recs in recommendations.items():
            if segment in segments_df.index:
                print(f"\n{segment.upper()}:")
                for i, rec in enumerate(recs, 1):
                    print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        print("✅ Analysis Complete - Ready for Action!")
        print("="*80)

def install_dependencies():
    """
    Function to help install missing dependencies
    """
    missing_packages = []
    
    try:
        import pandas
    except ImportError:
        missing_packages.append('pandas')
    
    try:
        import numpy
    except ImportError:
        missing_packages.append('numpy')
        
    try:
        import pytz
    except ImportError:
        missing_packages.append('pytz')
    
    if missing_packages:
        print("❌ Missing required packages:", ", ".join(missing_packages))
        print("\n📦 To install required packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Optional packages
    optional_packages = []
    if not SKLEARN_AVAILABLE:
        optional_packages.append('scikit-learn')
    if not MATPLOTLIB_AVAILABLE:
        optional_packages.append('matplotlib')
    if not SEABORN_AVAILABLE:
        optional_packages.append('seaborn')
    
    if optional_packages:
        print("⚠️  Optional packages for enhanced features:")
        print(f"pip install {' '.join(optional_packages)}")
    
    return True

def create_customer_segmentation(df=None):
    """
    Standalone function to create customer segmentation
    This can be called directly from app.py
    """
    try:
        # Check dependencies
        if not install_dependencies():
            return None
        
        # Initialize the segmentation system
        segmentation = CustomerSegmentation()
        
        # Run complete analysis
        results = segmentation.run_complete_analysis(df)
        
        # Print report
        segmentation.print_analysis_report(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in customer segmentation: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all required packages are installed")
        print("2. Check that your data has the required columns")
        print("3. Verify datetime formats are correct")
        return None

def main():
    """
    Main function to demonstrate the customer segmentation system
    """
    print("🏢 ACC Sales Intelligence System - Customer Segmentation")
    print("=" * 60)
    
    # Check if we can run the analysis
    if not install_dependencies():
        print("\n❌ Cannot run analysis - missing required dependencies")
        return
    
    try:
        # Run the segmentation analysis
        results = create_customer_segmentation()
        
        if results:
            print("\n✅ Analysis completed successfully!")
            
            # Save results (optional)
            try:
                results['rfm_data'].to_csv('customer_segments.csv', index=False)
                print("📄 Results saved to 'customer_segments.csv'")
            except Exception as e:
                print(f"⚠️  Could not save CSV file: {e}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\n❌ Error: {e}")
        print("\nThis error has been logged. Please check:")
        print("1. All required packages are installed")
        print("2. Data formats are correct")
        print("3. File permissions for saving results")

# Alternative entry points for different environments
def run_analysis_safe(customer_data=None):
    """
    Safe entry point that handles all errors gracefully
    Use this function if you're having import issues
    """
    try:
        return create_customer_segmentation(customer_data)
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None

def get_sample_analysis():
    """
    Get a sample analysis without running the full system
    Useful for testing and demonstration
    """
    try:
        segmentation = CustomerSegmentation()
        sample_df = segmentation.load_sample_data()
        
        print(f"📊 Sample data generated: {len(sample_df)} customers")
        print(f"Columns: {list(sample_df.columns)}")
        print(f"Date range: {sample_df['first_purchase_date'].min()} to {sample_df['last_purchase_date'].max()}")
        
        return sample_df
    except Exception as e:
        print(f"Could not generate sample data: {e}")
        return None

if __name__ == "__main__":
    main()

# Quick test function - uncomment to test
# if __name__ == "__main__":
#     print("Running quick test...")
#     sample_data = get_sample_analysis()
#     if sample_data is not None:
#         results = run_analysis_safe(sample_data)

# Example usage for integration with app.py:
"""
# In your app.py file, use this simple import:

try:
    from customer_segmentation import create_customer_segmentation, run_analysis_safe
    
    # Use the safe function
    results = run_analysis_safe(your_dataframe)
    
    if results:
        print("Segmentation completed!")
        # Use results['rfm_data'] for your dashboard
        # Use results['summary'] for key metrics
        # Use results['recommendations'] for actionable insights
    else:
        print("Segmentation failed - check logs")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure customer_segmentation.py is in the same directory")

# Required DataFrame columns for your data:
# - customer_id: Unique customer identifier
# - first_purchase_date: Date of first purchase
# - last_purchase_date: Date of last purchase  
# - total_orders: Number of orders
# - total_spent_sar: Total amount spent in SAR
# - avg_order_value_sar: Average order value in SAR
"""
"""
# Common datetime fixes:

# 1. Fix timezone-naive and timezone-aware mixing:
from datetime import datetime
import pytz

# Instead of this (causes error):
# naive_dt = datetime(2024, 1, 1)
# aware_dt = datetime.now(pytz.timezone('Asia/Riyadh'))
# diff = aware_dt - naive_dt  # ERROR!

# Do this:
dt_handler = DateTimeHandler()
naive_dt = datetime(2024, 1, 1)
aware_dt = datetime.now(pytz.timezone('Asia/Riyadh'))

# Normalize both to same timezone
normalized_naive = dt_handler.normalize_datetime(naive_dt)
normalized_aware = dt_handler.normalize_datetime(aware_dt)
diff = normalized_aware - normalized_naive  # Works!

# 2. Or make both naive:
naive_dt1 = dt_handler.make_naive(normalized_naive)
naive_dt2 = dt_handler.make_naive(normalized_aware)
diff = naive_dt2 - naive_dt1  # Works!

# 3. Use the safe calculation method:
days_between = dt_handler.calculate_days_between(naive_dt, aware_dt)
"""
