#!/usr/bin/env python3
"""
ACC Sales Intelligence System - Customer Segmentation Module
Advanced Analytics for Sales Performance & Business Intelligence

This module handles customer segmentation analysis for 110K+ customer records
with proper timezone management and Saudi Riyal currency formatting.

Compatible with Streamlit and handles large datasets efficiently.

Author: ACC Sales Intelligence Team
Date: July 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import os
from pathlib import Path

# Handle timezone operations safely
try:
    import pytz
    PYTZ_AVAILABLE = True
    SAUDI_TZ = pytz.timezone('Asia/Riyadh')
    UTC_TZ = pytz.UTC
except ImportError:
    PYTZ_AVAILABLE = False
    print("⚠️  pytz not available - using basic datetime operations")

# Optional advanced analytics libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DateTimeHandler:
    """
    Utility class to handle timezone-aware and timezone-naive datetime operations
    Fixes the common error: Cannot subtract tz-naive and tz-aware datetime-like objects
    """
    
    @staticmethod
    def normalize_datetime(dt: Union[datetime, pd.Timestamp, str, None], 
                          target_tz=None) -> Optional[datetime]:
        """
        Normalize datetime to specified timezone, handling both naive and aware datetimes
        
        Args:
            dt: Input datetime (can be naive, aware, string, or pandas Timestamp)
            target_tz: Target timezone (default: Saudi Arabia if pytz available)
            
        Returns:
            Normalized datetime or None if input is invalid
        """
        if dt is None or pd.isna(dt):
            return None
            
        try:
            # Convert string to datetime
            if isinstance(dt, str):
                dt = pd.to_datetime(dt)
            
            # Convert pandas timestamp to datetime
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            
            # If no timezone support, return as-is
            if not PYTZ_AVAILABLE:
                return dt.replace(tzinfo=None) if hasattr(dt, 'tzinfo') and dt.tzinfo else dt
            
            # Set default timezone
            if target_tz is None:
                target_tz = SAUDI_TZ
            
            # Handle timezone conversion
            if dt.tzinfo is None:
                # Naive datetime - assume it's in target timezone
                dt = target_tz.localize(dt)
            else:
                # Timezone-aware datetime - convert to target timezone
                dt = dt.astimezone(target_tz)
                
            return dt
            
        except Exception as e:
            logger.warning(f"Error normalizing datetime {dt}: {e}")
            # Return naive datetime as fallback
            if hasattr(dt, 'replace') and hasattr(dt, 'tzinfo'):
                return dt.replace(tzinfo=None)
            return None
    
    @staticmethod
    def make_naive(dt: datetime) -> Optional[datetime]:
        """Convert timezone-aware datetime to naive datetime"""
        if dt is None:
            return None
        try:
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt
        except Exception:
            return dt
    
    @staticmethod
    def calculate_days_between(start_dt: Any, end_dt: Any) -> int:
        """
        Calculate days between two datetimes, handling timezone issues safely
        
        Args:
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            Number of days between dates (always positive)
        """
        try:
            # Handle None values
            if start_dt is None or end_dt is None:
                return 0
            
            # Normalize both datetimes
            start_normalized = DateTimeHandler.normalize_datetime(start_dt)
            end_normalized = DateTimeHandler.normalize_datetime(end_dt)
            
            if start_normalized is None or end_normalized is None:
                return 0
            
            # Calculate difference
            diff = end_normalized - start_normalized
            return abs(diff.days)
            
        except Exception as e:
            logger.warning(f"Error calculating days between {start_dt} and {end_dt}: {e}")
            # Fallback: try with naive datetimes
            try:
                start_naive = DateTimeHandler.make_naive(
                    DateTimeHandler.normalize_datetime(start_dt)
                )
                end_naive = DateTimeHandler.make_naive(
                    DateTimeHandler.normalize_datetime(end_dt)
                )
                if start_naive and end_naive:
                    diff = end_naive - start_naive
                    return abs(diff.days)
            except Exception:
                pass
            return 0

class CurrencyFormatter:
    """Handle Saudi Riyal currency formatting for the ACC system"""
    
    @staticmethod
    def format_sar(amount: float, decimals: int = 2, short: bool = False) -> str:
        """
        Format amount in Saudi Riyal
        
        Args:
            amount: Amount to format
            decimals: Number of decimal places
            short: Use short format (K/M abbreviations)
            
        Returns:
            Formatted currency string
        """
        try:
            if pd.isna(amount) or amount is None:
                return "SAR 0.00"
            
            amount = float(amount)
            
            if short:
                if amount >= 1000000:
                    return f"SAR {amount/1000000:.1f}M"
                elif amount >= 1000:
                    return f"SAR {amount/1000:.1f}K"
                else:
                    return f"SAR {amount:.0f}"
            else:
                return f"SAR {amount:,.{decimals}f}"
                
        except (ValueError, TypeError):
            return "SAR 0.00"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format percentage values"""
        try:
            if pd.isna(value) or value is None:
                return "0.0%"
            return f"{float(value):.{decimals}f}%"
        except (ValueError, TypeError):
            return "0.0%"

class CustomerSegmentation:
    """
    Main customer segmentation class for ACC Sales Intelligence System
    Handles large datasets (110K+ records) efficiently
    """
    
    def __init__(self, timezone: str = 'Asia/Riyadh'):
        """Initialize the segmentation system"""
        self.timezone_str = timezone
        self.dt_handler = DateTimeHandler()
        self.currency_formatter = CurrencyFormatter()
        self.segments = {}
        self.analysis_results = {}
        
        # Set timezone if available
        if PYTZ_AVAILABLE:
            try:
                self.timezone = pytz.timezone(timezone)
            except:
                self.timezone = pytz.UTC
                logger.warning(f"Invalid timezone {timezone}, using UTC")
        else:
            self.timezone = None
            
        logger.info("ACC Customer Segmentation System initialized")
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data for customer segmentation
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if df is None or df.empty:
            errors.append("DataFrame is empty or None")
            return False, errors
        
        # Check required columns (flexible naming)
        required_patterns = {
            'customer_id': ['customer_id', 'customer', 'cust_id', 'id', 'customer_number'],
            'date': ['date', 'purchase_date', 'order_date', 'transaction_date', 'created_date'],
            'amount': ['amount', 'total', 'revenue', 'sales', 'value', 'price']
        }
        
        found_columns = {}
        for field, patterns in required_patterns.items():
            found = False
            for pattern in patterns:
                matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
                if matching_cols:
                    found_columns[field] = matching_cols[0]
                    found = True
                    break
            
            if not found:
                errors.append(f"No column found for {field}. Expected one of: {patterns}")
        
        # Check data types and basic validation
        if len(errors) == 0:
            try:
                # Test date parsing
                if 'date' in found_columns:
                    test_dates = pd.to_datetime(df[found_columns['date']].head(), errors='coerce')
                    if test_dates.isna().all():
                        errors.append(f"Date column '{found_columns['date']}' contains invalid dates")
                
                # Test amount parsing
                if 'amount' in found_columns:
                    test_amounts = pd.to_numeric(df[found_columns['amount']].head(), errors='coerce')
                    if test_amounts.isna().all():
                        errors.append(f"Amount column '{found_columns['amount']}' contains invalid numbers")
                        
            except Exception as e:
                errors.append(f"Data validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect column mappings based on common patterns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping field types to column names
        """
        column_mapping = {}
        
        # Column detection patterns
        patterns = {
            'customer_id': [
                'customer_id', 'customer', 'cust_id', 'customer_number', 'id',
                'client_id', 'account_id', 'user_id'
            ],
            'date': [
                'date', 'purchase_date', 'order_date', 'transaction_date',
                'created_date', 'timestamp', 'datetime'
            ],
            'amount': [
                'amount', 'total', 'revenue', 'sales', 'value', 'price',
                'spend', 'purchase_amount', 'order_total'
            ]
        }
        
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                matching_cols = [col for col in df.columns 
                               if pattern.lower() in col.lower()]
                if matching_cols:
                    column_mapping[field] = matching_cols[0]
                    break
        
        return column_mapping
    
    def prepare_data(self, df: pd.DataFrame, column_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """
        Prepare and clean data for customer segmentation analysis
        
        Args:
            df: Raw input DataFrame
            column_mapping: Optional column mapping dictionary
            
        Returns:
            Cleaned and prepared DataFrame
        """
        logger.info(f"Preparing data: {len(df)} rows")
        
        # Auto-detect columns if mapping not provided
        if column_mapping is None:
            column_mapping = self.auto_detect_columns(df)
            logger.info(f"Auto-detected columns: {column_mapping}")
        
        # Create working copy
        data = df.copy()
        
        # Rename columns to standard names
        rename_map = {}
        if 'customer_id' in column_mapping:
            rename_map[column_mapping['customer_id']] = 'customer_id'
        if 'date' in column_mapping:
            rename_map[column_mapping['date']] = 'purchase_date'
        if 'amount' in column_mapping:
            rename_map[column_mapping['amount']] = 'amount_sar'
        
        data = data.rename(columns=rename_map)
        
        # Clean and convert data types
        try:
            # Clean customer IDs
            if 'customer_id' in data.columns:
                data['customer_id'] = data['customer_id'].astype(str)
                data['customer_id'] = data['customer_id'].str.strip()
            
            # Parse dates
            if 'purchase_date' in data.columns:
                data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
                # Remove rows with invalid dates
                data = data.dropna(subset=['purchase_date'])
            
            # Clean amounts
            if 'amount_sar' in data.columns:
                data['amount_sar'] = pd.to_numeric(data['amount_sar'], errors='coerce')
                # Remove negative amounts and zeros
                data = data[data['amount_sar'] > 0]
            
            # Remove duplicates
            initial_rows = len(data)
            data = data.drop_duplicates()
            if len(data) < initial_rows:
                logger.info(f"Removed {initial_rows - len(data)} duplicate rows")
            
            logger.info(f"Data prepared: {len(data)} clean rows remaining")
            return data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def calculate_rfm_metrics(self, df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for customer segmentation
        
        Args:
            df: Prepared customer transaction data
            reference_date: Reference date for recency calculation
            
        Returns:
            DataFrame with RFM metrics per customer
        """
        logger.info("Calculating RFM metrics...")
        
        if reference_date is None:
            if PYTZ_AVAILABLE and self.timezone:
                reference_date = datetime.now(self.timezone)
            else:
                reference_date = datetime.now()
        
        # Aggregate customer data
        customer_agg = df.groupby('customer_id').agg({
            'purchase_date': ['min', 'max', 'count'],
            'amount_sar': ['sum', 'mean']
        }).round(2)
        
        # Flatten column names
        customer_agg.columns = [
            'first_purchase_date', 'last_purchase_date', 'frequency',
            'monetary_sar', 'avg_order_value_sar'
        ]
        
        # Calculate recency (days since last purchase)
        customer_agg['recency_days'] = customer_agg['last_purchase_date'].apply(
            lambda x: self.dt_handler.calculate_days_between(x, reference_date)
        )
        
        # Calculate customer lifetime (days since first purchase)
        customer_agg['customer_lifetime_days'] = customer_agg['first_purchase_date'].apply(
            lambda x: self.dt_handler.calculate_days_between(x, reference_date)
        )
        
        # Reset index to make customer_id a column
        customer_agg = customer_agg.reset_index()
        
        logger.info(f"RFM metrics calculated for {len(customer_agg)} customers")
        return customer_agg
    
    def assign_rfm_scores(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign RFM scores (1-5) based on quintiles
        
        Args:
            rfm_df: DataFrame with RFM metrics
            
        Returns:
            DataFrame with RFM scores added
        """
        logger.info("Assigning RFM scores...")
        
        df = rfm_df.copy()
        
        # Calculate quintiles for scoring (1 = worst, 5 = best)
        # Recency: Lower is better (recent purchases)
        df['R_Score'] = pd.qcut(df['recency_days'], 5, labels=[5,4,3,2,1], duplicates='drop')
        
        # Frequency: Higher is better
        df['F_Score'] = pd.qcut(df['frequency'].rank(method='first'), 5, 
                               labels=[1,2,3,4,5], duplicates='drop')
        
        # Monetary: Higher is better
        df['M_Score'] = pd.qcut(df['monetary_sar'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Handle cases where qcut might fail due to duplicate values
        for score_col in ['R_Score', 'F_Score', 'M_Score']:
            df[score_col] = df[score_col].fillna(3)  # Default to middle score
        
        # Create combined RFM score
        df['RFM_Score'] = (df['R_Score'].astype(str) + 
                          df['F_Score'].astype(str) + 
                          df['M_Score'].astype(str))
        
        return df
    
    def create_business_segments(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create business-meaningful customer segments based on RFM scores
        
        Args:
            rfm_df: DataFrame with RFM scores
            
        Returns:
            DataFrame with business segments assigned
        """
        logger.info("Creating business segments...")
        
        df = rfm_df.copy()
        
        def assign_segment(row):
            """Assign business segment based on RFM scores"""
            try:
                R = int(float(row['R_Score']))
                F = int(float(row['F_Score']))
                M = int(float(row['M_Score']))
            except (ValueError, TypeError):
                return "Undefined"
            
            # Business segment logic
            if R >= 4 and F >= 4 and M >= 4:
                return "Champions"
            elif F >= 4 and M >= 3:
                return "Loyal Customers"
            elif R >= 4 and F >= 2 and M >= 2:
                return "Potential Loyalists"
            elif R >= 4 and F <= 2:
                return "New Customers"
            elif R >= 3 and M >= 3 and F <= 2:
                return "Promising"
            elif R >= 3 and F >= 3 and M >= 3:
                return "Need Attention"
            elif R <= 3 and F >= 3 and M >= 3:
                return "About to Sleep"
            elif R <= 2 and F >= 3 and M >= 3:
                return "At Risk"
            elif R <= 2 and F >= 4 and M >= 4:
                return "Cannot Lose Them"
            elif R <= 2 and F <= 2 and M >= 3:
                return "Hibernating"
            else:
                return "Lost"
        
        df['Segment'] = df.apply(assign_segment, axis=1)
        
        # Log segment distribution
        segment_counts = df['Segment'].value_counts()
        logger.info(f"Segment distribution:\n{segment_counts}")
        
        return df
    
    def perform_clustering(self, rfm_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Perform K-means clustering analysis (if sklearn available)
        
        Args:
            rfm_df: DataFrame with RFM metrics
            n_clusters: Number of clusters to create
            
        Returns:
            DataFrame with cluster assignments
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available - using monetary quintiles for clustering")
            rfm_df['Cluster'] = pd.qcut(rfm_df['monetary_sar'], n_clusters, 
                                       labels=[f'Cluster_{i}' for i in range(n_clusters)],
                                       duplicates='drop')
            return rfm_df
        
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        # Prepare features for clustering
        features = ['recency_days', 'frequency', 'monetary_sar']
        X = rfm_df[features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        rfm_df['Cluster'] = [f'Cluster_{label}' for label in cluster_labels]
        
        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            logger.info(f"Clustering silhouette score: {silhouette_avg:.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
        
        return rfm_df
    
    def generate_summary_report(self, rfm_df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive summary report with Saudi Riyal formatting
        
        Args:
            rfm_df: DataFrame with customer segments
            
        Returns:
            Dictionary containing summary statistics and insights
        """
        logger.info("Generating summary report...")
        
        # Overall statistics
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_customers': len(rfm_df),
            'total_revenue_sar': rfm_df['monetary_sar'].sum(),
            'avg_customer_value_sar': rfm_df['monetary_sar'].mean(),
            'total_transactions': rfm_df['frequency'].sum(),
            'avg_recency_days': rfm_df['recency_days'].mean(),
            'avg_frequency': rfm_df['frequency'].mean()
        }
        
        # Business segments analysis
        segment_analysis = rfm_df.groupby('Segment').agg({
            'customer_id': 'count',
            'recency_days': 'mean',
            'frequency': 'mean', 
            'monetary_sar': ['mean', 'sum'],
            'avg_order_value_sar': 'mean'
        }).round(2)
        
        # Flatten column names
        segment_analysis.columns = [
            'customer_count', 'avg_recency_days', 'avg_frequency',
            'avg_monetary_sar', 'total_revenue_sar', 'avg_order_value_sar'
        ]
        
        # Calculate percentages
        segment_analysis['percentage'] = (
            segment_analysis['customer_count'] / summary['total_customers'] * 100
        ).round(2)
        
        # Format currency columns
        currency_columns = ['avg_monetary_sar', 'total_revenue_sar', 'avg_order_value_sar']
        for col in currency_columns:
            segment_analysis[f'{col}_formatted'] = segment_analysis[col].apply(
                self.currency_formatter.format_sar
            )
        
        summary['segment_analysis'] = segment_analysis.to_dict('index')
        
        # Format overall currency values
        summary['total_revenue_sar_formatted'] = self.currency_formatter.format_sar(
            summary['total_revenue_sar']
        )
        summary['avg_customer_value_sar_formatted'] = self.currency_formatter.format_sar(
            summary['avg_customer_value_sar']
        )
        
        return summary
    
    def get_segment_recommendations(self) -> Dict[str, List[str]]:
        """
        Get actionable business recommendations for each customer segment
        
        Returns:
            Dictionary of segment recommendations
        """
        return {
            "Champions": [
                "Implement VIP customer program with exclusive benefits",
                "Request testimonials and case studies for marketing",
                "Offer early access to new products and services",
                "Create referral programs with attractive incentives",
                "Provide dedicated account management"
            ],
            "Loyal Customers": [
                "Maintain regular engagement through personalized communication",
                "Offer loyalty program rewards and tier upgrades",
                "Cross-sell complementary products and services",
                "Prevent churn with proactive customer success initiatives",
                "Gather feedback for product improvement"
            ],
            "Potential Loyalists": [
                "Encourage higher purchase frequency with targeted promotions",
                "Provide product education and usage recommendations", 
                "Offer membership programs with increasing benefits",
                "Send personalized product suggestions based on purchase history",
                "Create engagement campaigns to build relationship"
            ],
            "New Customers": [
                "Design comprehensive welcome series and onboarding",
                "Introduce product range gradually with educational content",
                "Offer new customer incentives for repeat purchases",
                "Collect early feedback to optimize customer experience",
                "Assign customer success representatives"
            ],
            "Promising": [
                "Focus on increasing purchase frequency with targeted campaigns",
                "Offer bundle deals and volume discounts",
                "Share customer success stories and use cases",
                "Provide exceptional customer service and support",
                "Create urgency with limited-time offers"
            ],
            "Need Attention": [
                "Send personalized re-engagement offers immediately",
                "Investigate reasons for declining purchase activity",
                "Offer customer service support and issue resolution",
                "Create win-back campaigns with attractive incentives",
                "Provide product recommendations based on past purchases"
            ],
            "About to Sleep": [
                "Launch immediate re-engagement campaigns",
                "Offer special discounts and limited-time promotions",
                "Survey customers to understand satisfaction issues",
                "Provide personalized product recommendations",
                "Increase communication frequency temporarily"
            ],
            "At Risk": [
                "Execute urgent retention campaigns with deep discounts",
                "Provide free shipping and additional value-adds",
                "Arrange personal outreach from account managers",
                "Address any service issues or complaints immediately",
                "Offer service recovery and relationship rebuilding"
            ],
            "Cannot Lose Them": [
                "Priority escalation to senior customer success team",
                "Provide exclusive offers they cannot refuse",
                "Arrange personal contact from executive leadership",
                "Investigate and resolve any underlying issues",
                "Offer service credits or compensation if appropriate"
            ],
            "Hibernating": [
                "Design strong win-back campaigns with compelling offers",
                "Showcase new product innovations and improvements",
                "Provide limited-time reactivation incentives",
                "Survey for feedback and implement improvements",
                "Consider different communication channels"
            ],
            "Lost": [
                "Execute final win-back attempt with best available offers",
                "Collect detailed feedback for future customer experience improvements",
                "Remove from active marketing lists to reduce costs",
                "Focus marketing resources on more promising segments",
                "Maintain minimal touchpoints for potential future reactivation"
            ]
        }
    
    def run_complete_analysis(self, df: pd.DataFrame, 
                            column_mapping: Dict[str, str] = None,
                            n_clusters: int = 5) -> Dict:
        """
        Run complete customer segmentation analysis
        
        Args:
            df: Input customer data DataFrame
            column_mapping: Column mapping dictionary
            n_clusters: Number of clusters for K-means analysis
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info("Starting complete customer segmentation analysis...")
            
            # Validate input data
            is_valid, errors = self.validate_data(df)
            if not is_valid:
                raise ValueError(f"Data validation failed: {'; '.join(errors)}")
            
            # Prepare data
            prepared_data = self.prepare_data(df, column_mapping)
            
            # Calculate RFM metrics
            rfm_data = self.calculate_rfm_metrics(prepared_data)
            
            # Assign RFM scores
            rfm_data = self.assign_rfm_scores(rfm_data)
            
            # Create business segments
            rfm_data = self.create_business_segments(rfm_data)
            
            # Perform clustering analysis
            rfm_data = self.perform_clustering(rfm_data, n_clusters)
            
            # Generate summary report
            summary = self.generate_summary_report(rfm_data)
            
            # Get recommendations
            recommendations = self.get_segment_recommendations()
            
            # Compile final results
            results = {
                'status': 'success',
                'customer_data': rfm_data,
                'summary': summary,
                'recommendations': recommendations,
                'analysis_metadata': {
                    'total_input_records': len(df),
                    'processed_records': len(prepared_data),
                    'final_customers': len(rfm_data),
                    'data_quality_score': len(rfm_data) / len(df) * 100,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'system_version': '2.0'
                }
            }
            
            # Store results for later use
            self.analysis_results = results
            
            logger.info("✅ Customer segmentation analysis completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in customer segmentation analysis: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def print_analysis_report(self, results: Dict = None):
        """
        Print formatted analysis report to console
        
        Args:
            results: Analysis results dictionary (uses stored results if None)
        """
        if results is None:
            results = self.analysis_results
        
        if not results or results.get('status') != 'success':
            print("❌ No successful analysis results available")
            return
        
        summary = results['summary']
        metadata = results['analysis_metadata']
        
        print("\n" + "="*80)
        print("🏢 ACC SALES INTELLIGENCE SYSTEM")
        print("📊 CUSTOMER SEGMENTATION ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📈 ANALYSIS OVERVIEW")
        print(f"Analysis Date: {summary['analysis_date']}")
        print(f"Total Customers Analyzed: {summary['total_customers']:,}")
        print(f"Total Revenue: {summary['total_revenue_sar_formatted']}")
        print(f"Average Customer Value: {summary['avg_customer_value_sar_formatted']}")
        print(f"Total Transactions: {summary['total_transactions']:,}")
        print(f"Data Quality Score: {metadata['data_quality_score']:.1f}%")
        
        print(f"\n🎯 CUSTOMER SEGMENTS")
        print("-" * 80)
        
        for segment, data in summary['segment_analysis'].items():
            print(f"\n{segment.upper()}")
            print(f"  • Customers: {data['customer_count']:,} ({data['percentage']:.1f}%)")
            print(f"  • Avg Days Since Last Purchase: {data['avg_recency_days']:.0f}")
            print(f"  • Avg Orders per Customer: {data['avg_frequency']:.1f}")
            print(f"  • Avg Customer Value: {data['avg_monetary_sar_formatted']}")
            print(f"  • Total Segment Revenue: {data['total_revenue_sar_formatted']}")
            print(f"  • Avg Order Value: {data['avg_order_value_sar_formatted']}")
        
        print(f"\n💡 STRATEGIC RECOMMENDATIONS")
        print("-" * 80)
        
        recommendations = results['recommendations']
        for segment, recs in recommendations.items():
            if segment in summary['segment_analysis']:
                print(f"\n{segment.upper()}:")
                for i, rec in enumerate(recs[:3], 1):  # Show top 3 recommendations
                    print(f"  {i}. {rec}")
        
        print(f"\n📊 SYSTEM PERFORMANCE")
        print(f"Input Records: {metadata['total_input_records']:,}")
        print(f"Processed Records: {metadata['processed_records']:,}")
        print(f"Final Customer Profiles: {metadata['final_customers']:,}")
        
        print("\n" + "="*80)
        print("✅ Analysis Complete - Ready for Implementation!")
        print("="*80)
    
    def export_results(self, results: Dict = None, output_dir: str = "customer_segmentation_output") -> Dict[str, str]:
        """
        Export analysis results to files
        
        Args:
            results: Analysis results dictionary
            output_dir: Output directory path
            
        Returns:
            Dictionary of exported file paths
        """
        if results is None:
            results = self.analysis_results
        
        if not results or results.get('status') != 'success':
            logger.error("No successful results to export")
            return {}
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        exported_files = {}
        
        try:
            # Export customer segments CSV
            customer_data = results['customer_data']
            csv_path = os.path.join(output_dir, 'customer_segments.csv')
            customer_data.to_csv(csv_path, index=False)
            exported_files['customer_segments'] = csv_path
            
            # Export summary report JSON
            summary_path = os.path.join(output_dir, 'analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(results['summary'], f, indent=2, default=str)
            exported_files['summary'] = summary_path
            
            # Export recommendations JSON
            rec_path = os.path.join(output_dir, 'segment_recommendations.json')
            with open(rec_path, 'w') as f:
                json.dump(results['recommendations'], f, indent=2)
            exported_files['recommendations'] = rec_path
            
            logger.info(f"Results exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
        
        return exported_files

# Convenience functions for integration with Streamlit and other systems

def create_customer_segmentation(df: pd.DataFrame, 
                               column_mapping: Dict[str, str] = None,
                               n_clusters: int = 5) -> Optional[Dict]:
    """
    Main function to create customer segmentation analysis
    
    Args:
        df: Customer data DataFrame
        column_mapping: Column mapping dictionary
        n_clusters: Number of clusters
        
    Returns:
        Analysis results or None if failed
    """
    try:
        segmentation = CustomerSegmentation()
        results = segmentation.run_complete_analysis(df, column_mapping, n_clusters)
        
        if results.get('status') == 'success':
            return results
        else:
            logger.error(f"Segmentation failed: {results.get('error_message', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"Error in create_customer_segmentation: {e}")
        return None

def run_analysis_safe(df: pd.DataFrame, **kwargs) -> Optional[Dict]:
    """
    Safe wrapper for customer segmentation analysis
    
    Args:
        df: Customer data DataFrame
        **kwargs: Additional arguments
        
    Returns:
        Analysis results or None if failed
    """
    try:
        return create_customer_segmentation(df, **kwargs)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def load_excel_data(file_path: str, sheet_name: str = None) -> Optional[pd.DataFrame]:
    """
    Load customer data from Excel file
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name (optional)
        
    Returns:
        DataFrame or None if failed
    """
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading Excel file {file_path}: {e}")
        return None

def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available"""
    deps = {
        'pandas': False,
        'numpy': False,
        'pytz': PYTZ_AVAILABLE,
        'sklearn': SKLEARN_AVAILABLE
    }
    
    try:
        import pandas
        deps['pandas'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        deps['numpy'] = True
    except ImportError:
        pass
    
    return deps

def main():
    """Main function for standalone execution"""
    print("🏢 ACC Sales Intelligence System - Customer Segmentation")
    print("=" * 60)
    
    # Check dependencies
    deps = check_dependencies()
    missing = [k for k, v in deps.items() if not v and k in ['pandas', 'numpy']]
    
    if missing:
        print(f"❌ Missing required dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return
    
    # Create sample data for demonstration
    print("📊 Generating sample data for demonstration...")
    
    try:
        segmentation = CustomerSegmentation()
        
        # Generate sample customer data
        np.random.seed(42)
        n_customers = 1000
        
        sample_data = []
        for i in range(n_customers):
            customer_id = f"CUST_{i+1:04d}"
            
            # Random purchase dates
            days_ago = np.random.randint(1, 365)
            purchase_date = datetime.now() - timedelta(days=days_ago)
            
            # Random amounts
            amount = np.random.exponential(500) + 50
            
            sample_data.append({
                'customer_id': customer_id,
                'purchase_date': purchase_date,
                'amount_sar': amount
            })
        
        # Create multiple transactions per customer
        all_data = []
        for _ in range(3):  # Average 3 transactions per customer
            for record in sample_data:
                if np.random.random() > 0.3:  # 70% chance of additional transaction
                    new_record = record.copy()
                    # Vary the date and amount slightly
                    days_variation = np.random.randint(-30, 30)
                    new_record['purchase_date'] += timedelta(days=days_variation)
                    new_record['amount_sar'] *= np.random.uniform(0.5, 1.5)
                    all_data.append(new_record)
        
        sample_df = pd.DataFrame(all_data)
        
        print(f"✅ Sample data created: {len(sample_df)} transactions for {sample_df['customer_id'].nunique()} customers")
        
        # Run analysis
        print("🚀 Running customer segmentation analysis...")
        results = segmentation.run_complete_analysis(sample_df)
        
        if results and results.get('status') == 'success':
            # Print report
            segmentation.print_analysis_report(results)
            
            # Export results
            exported = segmentation.export_results(results)
            if exported:
                print(f"\n📁 Results exported to: {list(exported.values())}")
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
