"""
Stock Market Impact Analysis on Art Auction Prices
Integrated with MongoDB data pipeline and S3 storage

This script:
1. Fetches bid data from MongoDB
2. Runs stock market impact analysis
3. Runs hedonic pricing model
4. Uploads results to S3
5. Generates download links
"""

import pandas as pd
import numpy as np
import os
import logging
import boto3
from pymongo import MongoClient
from dotenv import load_dotenv
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log'),
        logging.StreamHandler()
    ]
)

# MongoDB configuration
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME')
SAFFRON_BID_COLLECTION_NAME = os.getenv('SAFFRON_BID_COLLECTION_NAME')

# S3 configuration
S3_BUCKET = 'scraped-art-data'
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Output directory
OUTPUT_DIR = './analysis_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def upload_to_s3(file_path, bucket_name, object_name=None):
    """Upload a file to S3 and return the download URL"""
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        logging.info(f"✓ Uploaded {file_path} to S3: {url}")
        return url
    except Exception as e:
        logging.error(f"Failed to upload {file_path} to S3: {str(e)}")
        return None


def fetch_bid_data_from_mongodb():
    """Fetch bid data from MongoDB collection"""
    try:
        logging.info("Connecting to MongoDB...")
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[SAFFRON_BID_COLLECTION_NAME]
        
        logging.info("Fetching bid data...")
        cursor = collection.find({})
        df = pd.DataFrame(list(cursor))
        
        logging.info(f"Fetched {len(df)} records from MongoDB")
        
        # Save raw data locally for analysis
        raw_data_path = os.path.join(OUTPUT_DIR, 'bid_data_complete_all_periods.csv')
        df.to_csv(raw_data_path, index=False)
        logging.info(f"Saved raw data to {raw_data_path}")
        
        client.close()
        return df, raw_data_path
        
    except Exception as e:
        logging.error(f"Error fetching data from MongoDB: {str(e)}")
        return None, None


def run_stock_impact_analysis(df):
    """Run stock market impact analysis on art auction prices"""
    logging.info("="*90)
    logging.info("RUNNING: Stock Market Impact Analysis")
    logging.info("="*90)
    
    results = {}
    
    try:
        # Clean data
        df_clean = df[
            (df['bid_usd'].notna()) & 
            (df['bid_usd'] > 0) &
            (df['sp500_pct_change_7day'].notna())
        ].copy()
        
        logging.info(f"Analysis dataset: {len(df_clean):,} rows")
        
        # Create log of bid amount
        df_clean['log_bid_usd'] = np.log(df_clean['bid_usd'])
        
        # ANALYSIS 1: Correlations
        market_vars = [
            'sp500_pct_change_7day',
            'sp500_pct_change_14day',
            'sp500_pct_change_21day',
            'sp500_pct_change_30day',
            'sp500_large_cap_pct_change_2day',
            'sp400_mid_cap_pct_change_2day',
            'sp600_small_cap_pct_change_2day',
            'vanguard_bond_pct_change_2day',
        ]
        
        correlations = []
        for var in market_vars:
            if var in df_clean.columns and df_clean[var].notna().sum() > 100:
                valid_data = df_clean[['bid_usd', var]].dropna()
                if len(valid_data) > 100:
                    corr = valid_data.corr().iloc[0, 1]
                    p_value = stats.pearsonr(valid_data['bid_usd'], valid_data[var])[1]
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    correlations.append({
                        'Variable': var,
                        'Correlation': corr,
                        'P_value': p_value,
                        'Significance': sig
                    })
        
        correlations_df = pd.DataFrame(correlations)
        corr_file = os.path.join(OUTPUT_DIR, 'stock_impact_correlations.csv')
        correlations_df.to_csv(corr_file, index=False)
        logging.info(f"✓ Saved correlations to {corr_file}")
        
        # ANALYSIS 2: Regression Models
        model1 = smf.ols('log_bid_usd ~ sp500_pct_change_7day', data=df_clean).fit()
        
        model2 = smf.ols('''log_bid_usd ~ sp500_pct_change_7day + 
                                            sp500_pct_change_14day + 
                                            sp500_pct_change_21day + 
                                            sp500_pct_change_30day''', data=df_clean).fit()
        
        # ANALYSIS 3: Market Conditions
        df_clean['market_condition'] = pd.cut(
            df_clean['sp500_pct_change_7day'],
            bins=[-100, -2, 0, 2, 100],
            labels=['Strong Decline', 'Slight Decline', 'Slight Increase', 'Strong Increase']
        )
        
        market_analysis = df_clean.groupby('market_condition', observed=True).agg({
            'bid_usd': ['count', 'mean', 'median', 'std'],
            'sp500_pct_change_7day': 'mean'
        }).round(2)
        
        market_file = os.path.join(OUTPUT_DIR, 'market_condition_analysis.csv')
        market_analysis.to_csv(market_file)
        logging.info(f"✓ Saved market condition analysis to {market_file}")
        
        # ANALYSIS 4: Summary Statistics
        overall_corr = df_clean[['bid_usd', 'sp500_pct_change_7day']].corr().iloc[0, 1]
        positive_market = df_clean[df_clean['sp500_pct_change_7day'] > 0]['bid_usd'].mean()
        negative_market = df_clean[df_clean['sp500_pct_change_7day'] <= 0]['bid_usd'].mean()
        
        summary = {
            'Overall_Correlation': overall_corr,
            'Model1_R_Squared': model1.rsquared,
            'Model1_P_Value': model1.pvalues['sp500_pct_change_7day'],
            'Model2_R_Squared': model2.rsquared,
            'Avg_Bid_Market_UP': positive_market,
            'Avg_Bid_Market_DOWN': negative_market,
            'Bid_Difference': positive_market - negative_market,
            'Total_Observations': len(df_clean)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_file = os.path.join(OUTPUT_DIR, 'stock_impact_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"✓ Saved summary statistics to {summary_file}")
        
        # Save detailed regression results
        regression_file = os.path.join(OUTPUT_DIR, 'stock_impact_regression_results.txt')
        with open(regression_file, 'w') as f:
            f.write("="*90 + "\n")
            f.write("STOCK MARKET IMPACT ON ART PRICES - REGRESSION RESULTS\n")
            f.write("="*90 + "\n\n")
            f.write("MODEL 1: Basic S&P 500 7-Day Effect\n")
            f.write("-"*90 + "\n")
            f.write(model1.summary().as_text())
            f.write("\n\n")
            f.write("MODEL 2: Multiple Time Periods\n")
            f.write("-"*90 + "\n")
            f.write(model2.summary().as_text())
        
        logging.info(f"✓ Saved regression results to {regression_file}")
        
        results['correlations_file'] = corr_file
        results['market_file'] = market_file
        results['summary_file'] = summary_file
        results['regression_file'] = regression_file
        
        return results
        
    except Exception as e:
        logging.error(f"Error in stock impact analysis: {str(e)}")
        return None


def run_hedonic_pricing_analysis(df):
    """Run hedonic pricing model analysis"""
    logging.info("="*90)
    logging.info("RUNNING: Hedonic Pricing Model")
    logging.info("="*90)
    
    results = {}
    
    try:
        # Prepare data
        df['log_bid'] = np.log(df['bid_usd'].replace(0, np.nan))
        df = df[np.isfinite(df['log_bid'])].copy()
        
        # Convert estimates
        df['lo_est'] = pd.to_numeric(df['lo_est'], errors='coerce')
        df['hi_est'] = pd.to_numeric(df['hi_est'], errors='coerce')
        df['log_lo_est'] = np.log(df['lo_est'].replace(0, np.nan))
        df['log_hi_est'] = np.log(df['hi_est'].replace(0, np.nan))
        df['estimate_midpoint'] = (df['lo_est'] + df['hi_est']) / 2
        df['log_estimate_midpoint'] = np.log(df['estimate_midpoint'].replace(0, np.nan))
        
        # Time variables
        df['year'] = pd.to_datetime(df['iso_date']).dt.year
        df['quarter'] = pd.to_datetime(df['iso_date']).dt.quarter
        
        logging.info(f"Hedonic analysis dataset: {len(df):,} rows")
        
        # MODEL 1: Market Conditions Only
        model1_data = df[[
            'log_bid', 
            'sp500_large_cap_pct_change_2day',
            'sp400_mid_cap_pct_change_2day',
            'sp600_small_cap_pct_change_2day',
            'vanguard_bond_pct_change_2day'
        ]].dropna()
        
        formula1 = '''log_bid ~ sp500_large_cap_pct_change_2day + 
                                sp400_mid_cap_pct_change_2day + 
                                sp600_small_cap_pct_change_2day + 
                                vanguard_bond_pct_change_2day'''
        
        model1 = smf.ols(formula1, data=model1_data).fit()
        logging.info(f"Model 1 R-squared: {model1.rsquared:.6f}")
        
        # MODEL 2: With Artist Fixed Effects
        artist_counts = df['artist_name'].value_counts()
        top_artists = artist_counts[artist_counts >= 50].index.tolist()
        logging.info(f"Including {len(top_artists)} artists with ≥50 observations")
        
        df_model2 = df[df['artist_name'].isin(top_artists)].copy()
        model2_data = df_model2[[
            'log_bid', 'artist_name',
            'sp500_large_cap_pct_change_2day',
            'sp400_mid_cap_pct_change_2day',
            'sp600_small_cap_pct_change_2day',
            'vanguard_bond_pct_change_2day'
        ]].dropna()
        
        formula2 = '''log_bid ~ sp500_large_cap_pct_change_2day + 
                                sp400_mid_cap_pct_change_2day + 
                                sp600_small_cap_pct_change_2day + 
                                vanguard_bond_pct_change_2day +
                                C(artist_name)'''
        
        model2 = smf.ols(formula2, data=model2_data).fit()
        logging.info(f"Model 2 R-squared: {model2.rsquared:.6f}")
        
        # MODEL 3: Multiple Time Periods
        model3_data = df_model2[[
            'log_bid', 'artist_name', 'year',
            'sp500_pct_change_7day',
            'sp500_pct_change_14day',
            'sp500_pct_change_21day',
            'sp500_pct_change_30day',
            'vanguard_bond_pct_change_2day'
        ]].dropna()
        
        formula3 = '''log_bid ~ sp500_pct_change_7day + 
                                sp500_pct_change_14day + 
                                sp500_pct_change_21day + 
                                sp500_pct_change_30day +
                                vanguard_bond_pct_change_2day +
                                C(artist_name) +
                                C(year)'''
        
        model3 = smf.ols(formula3, data=model3_data).fit()
        logging.info(f"Model 3 R-squared: {model3.rsquared:.6f}")
        
        # MODEL 4: With Estimate Controls
        model4_data = df_model2[[
            'log_bid', 'artist_name', 'log_estimate_midpoint',
            'sp500_large_cap_pct_change_2day',
            'sp400_mid_cap_pct_change_2day',
            'sp600_small_cap_pct_change_2day',
            'vanguard_bond_pct_change_2day'
        ]].dropna()
        
        formula4 = '''log_bid ~ sp500_large_cap_pct_change_2day + 
                                sp400_mid_cap_pct_change_2day + 
                                sp600_small_cap_pct_change_2day + 
                                vanguard_bond_pct_change_2day +
                                log_estimate_midpoint +
                                C(artist_name)'''
        
        model4 = smf.ols(formula4, data=model4_data).fit()
        logging.info(f"Model 4 R-squared: {model4.rsquared:.6f}")
        
        # Model Comparison
        comparison = pd.DataFrame({
            'Model': ['1. Market Only', '2. + Artist FE', '3. + Time Periods', '4. + Estimates'],
            'N': [len(model1_data), len(model2_data), len(model3_data), len(model4_data)],
            'R_Squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared],
            'Adj_R_Squared': [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj, model4.rsquared_adj],
            'F_Stat': [model1.fvalue, model2.fvalue, model3.fvalue, model4.fvalue],
            'AIC': [model1.aic, model2.aic, model3.aic, model4.aic],
            'BIC': [model1.bic, model2.bic, model3.bic, model4.bic]
        })
        
        comparison_file = os.path.join(OUTPUT_DIR, 'hedonic_model_comparison.csv')
        comparison.to_csv(comparison_file, index=False)
        logging.info(f"✓ Saved model comparison to {comparison_file}")
        
        # Hedonic Prices (Implicit Prices)
        market_vars = ['sp500_large_cap_pct_change_2day', 'sp400_mid_cap_pct_change_2day',
                       'sp600_small_cap_pct_change_2day', 'vanguard_bond_pct_change_2day']
        
        hedonic_prices = pd.DataFrame({
            'Variable': market_vars,
            'Coefficient': [model2.params[v] for v in market_vars],
            'Std_Error': [model2.bse[v] for v in market_vars],
            'P_Value': [model2.pvalues[v] for v in market_vars],
            'CI_Lower': [model2.conf_int().loc[v, 0] for v in market_vars],
            'CI_Upper': [model2.conf_int().loc[v, 1] for v in market_vars]
        })
        
        hedonic_file = os.path.join(OUTPUT_DIR, 'hedonic_implicit_prices.csv')
        hedonic_prices.to_csv(hedonic_file, index=False)
        logging.info(f"✓ Saved implicit prices to {hedonic_file}")
        
        # Artist Effects
        artist_effects = []
        for param in model2.params.index:
            if 'C(artist_name)' in param:
                artist = param.replace('C(artist_name)[T.', '').replace(']', '')
                effect = model2.params[param]
                premium_pct = (np.exp(effect) - 1) * 100
                artist_effects.append({
                    'Artist': artist,
                    'Effect': effect,
                    'Premium_Pct': premium_pct
                })
        
        artist_df = pd.DataFrame(artist_effects).sort_values('Effect', ascending=False)
        artist_file = os.path.join(OUTPUT_DIR, 'hedonic_artist_effects.csv')
        artist_df.to_csv(artist_file, index=False)
        logging.info(f"✓ Saved artist effects to {artist_file}")
        
        # Save detailed results
        detailed_file = os.path.join(OUTPUT_DIR, 'hedonic_model_detailed_results.txt')
        with open(detailed_file, 'w') as f:
            f.write("="*90 + "\n")
            f.write("HEDONIC PRICING MODEL - DETAILED RESULTS\n")
            f.write("="*90 + "\n\n")
            
            for i, model in enumerate([model1, model2, model3, model4], 1):
                f.write(f"\nMODEL {i}\n")
                f.write("-"*90 + "\n")
                f.write(model.summary().as_text())
                f.write("\n\n")
        
        logging.info(f"✓ Saved detailed results to {detailed_file}")
        
        results['comparison_file'] = comparison_file
        results['hedonic_file'] = hedonic_file
        results['artist_file'] = artist_file
        results['detailed_file'] = detailed_file
        
        return results
        
    except Exception as e:
        logging.error(f"Error in hedonic pricing analysis: {str(e)}")
        return None


def main():
    """Main execution function"""
    try:
        logging.info("="*90)
        logging.info("STARTING STOCK MARKET ANALYSIS PIPELINE")
        logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("="*90)
        
        # Step 1: Fetch data from MongoDB
        df, raw_data_path = fetch_bid_data_from_mongodb()
        if df is None:
            logging.error("Failed to fetch data from MongoDB. Aborting.")
            return
        
        # Step 2: Run Stock Impact Analysis
        stock_results = run_stock_impact_analysis(df)
        if stock_results is None:
            logging.error("Stock impact analysis failed.")
        
        # Step 3: Run Hedonic Pricing Analysis
        hedonic_results = run_hedonic_pricing_analysis(df)
        if hedonic_results is None:
            logging.error("Hedonic pricing analysis failed.")
        
        # Step 4: Upload all results to S3
        logging.info("="*90)
        logging.info("UPLOADING RESULTS TO S3")
        logging.info("="*90)
        
        s3_links = {}
        
        # Upload raw data
        if raw_data_path:
            url = upload_to_s3(raw_data_path, S3_BUCKET, 'analysis/bid_data_complete_all_periods.csv')
            if url:
                s3_links['raw_data'] = url
        
        # Upload stock impact analysis results
        if stock_results:
            for key, file_path in stock_results.items():
                filename = os.path.basename(file_path)
                url = upload_to_s3(file_path, S3_BUCKET, f'analysis/{filename}')
                if url:
                    s3_links[key] = url
        
        # Upload hedonic pricing results
        if hedonic_results:
            for key, file_path in hedonic_results.items():
                filename = os.path.basename(file_path)
                url = upload_to_s3(file_path, S3_BUCKET, f'analysis/{filename}')
                if url:
                    s3_links[key] = url
        
        # Step 5: Create summary with download links
        logging.info("="*90)
        logging.info("GENERATING DOWNLOAD LINKS")
        logging.info("="*90)
        
        links_summary = pd.DataFrame([
            {'File_Type': 'Raw Bid Data', 'Download_URL': s3_links.get('raw_data', 'N/A')},
            {'File_Type': 'Stock Impact Correlations', 'Download_URL': s3_links.get('correlations_file', 'N/A')},
            {'File_Type': 'Market Condition Analysis', 'Download_URL': s3_links.get('market_file', 'N/A')},
            {'File_Type': 'Stock Impact Summary', 'Download_URL': s3_links.get('summary_file', 'N/A')},
            {'File_Type': 'Regression Results (TXT)', 'Download_URL': s3_links.get('regression_file', 'N/A')},
            {'File_Type': 'Hedonic Model Comparison', 'Download_URL': s3_links.get('comparison_file', 'N/A')},
            {'File_Type': 'Hedonic Implicit Prices', 'Download_URL': s3_links.get('hedonic_file', 'N/A')},
            {'File_Type': 'Artist Effects', 'Download_URL': s3_links.get('artist_file', 'N/A')},
            {'File_Type': 'Hedonic Detailed Results (TXT)', 'Download_URL': s3_links.get('detailed_file', 'N/A')},
        ])
        
        links_file = os.path.join(OUTPUT_DIR, 'download_links.csv')
        links_summary.to_csv(links_file, index=False)
        logging.info(f"✓ Saved download links to {links_file}")
        
        # Upload links file to S3
        links_url = upload_to_s3(links_file, S3_BUCKET, 'analysis/download_links.csv')
        
        # Print summary
        logging.info("="*90)
        logging.info("✅ ANALYSIS COMPLETE")
        logging.info("="*90)
        logging.info("\nDOWNLOAD LINKS:")
        logging.info("-"*90)
        for _, row in links_summary.iterrows():
            logging.info(f"{row['File_Type']:40s}: {row['Download_URL']}")
        
        if links_url:
            logging.info(f"\n📋 All links available at: {links_url}")
        
        logging.info("\n" + "="*90)
        logging.info("Pipeline execution completed successfully!")
        logging.info("="*90)
        
        # Write links to file for email integration
        with open('links.txt', 'w') as f:
            f.write("Stock Market Analysis Results - Download Links\n")
            f.write("="*90 + "\n\n")
            for _, row in links_summary.iterrows():
                f.write(f"{row['File_Type']}\n{row['Download_URL']}\n\n")
        
        logging.info("✓ Links written to links.txt for email distribution")
        
    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
