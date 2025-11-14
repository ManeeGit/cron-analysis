"""
Analysis: Does Stock Market Performance Affect Art Auction Prices?
This script tests the relationship between stock market variables and bid amounts
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("IMPACT ANALYSIS: Stock Market Effect on Art Auction Prices")
print("="*90)

# Load the complete dataset
print("\nLoading data...")
df = pd.read_csv('bid_data_complete_all_periods.csv', low_memory=False)

# Clean data: remove missing values for key variables
print(f"Original dataset: {len(df):,} rows")

# Filter for rows with valid bid amounts and market data
df_clean = df[
    (df['bid_usd'].notna()) & 
    (df['bid_usd'] > 0) &
    (df['sp500_pct_change_7day'].notna())
].copy()

print(f"After cleaning: {len(df_clean):,} rows with valid data")
print(f"Date range: {df_clean['iso_date'].min()} to {df_clean['iso_date'].max()}")

# Create log of bid amount for better regression properties
df_clean['log_bid_usd'] = np.log(df_clean['bid_usd'])

print("\n" + "="*90)
print("ANALYSIS 1: Simple Correlation Analysis")
print("="*90)

# Correlations between market variables and bid amounts
market_vars = [
    'sp500_pct_change_7day',
    'sp500_pct_change_14day',
    'sp500_pct_change_21day',
    'sp500_pct_change_30day',
    'sp500_large_cap_pct_change_2day',
    'sp400_mid_cap_pct_change_2day',
    'sp600_small_cap_pct_change_2day',
    'bloomberg_agg_bond_pct_change_2day',
    'trading_volume_minus_1'
]

print("\nCorrelation with Bid Amount (USD):")
print("-" * 70)
correlations = []
for var in market_vars:
    if var in df_clean.columns and df_clean[var].notna().sum() > 100:
        # Use only rows where both variables are not null
        valid_data = df_clean[['bid_usd', var]].dropna()
        if len(valid_data) > 100:
            corr = valid_data.corr().iloc[0, 1]
            p_value = stats.pearsonr(valid_data['bid_usd'], valid_data[var])[1]
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            correlations.append({
                'Variable': var,
                'Correlation': corr,
                'P-value': p_value,
                'Sig': sig
            })
            print(f"  {var:45s}: {corr:>7.4f} {sig:>4s} (p={p_value:.4f})")

print("\n" + "="*90)
print("ANALYSIS 2: Regression Analysis - Effect on Bid Amounts")
print("="*90)

# Model 1: Basic S&P 500 effect (weekly)
print("\nMODEL 1: Basic S&P 500 Weekly Return Effect")
print("-" * 70)
model1 = smf.ols('log_bid_usd ~ sp500_pct_change_7day', data=df_clean).fit()
print(model1.summary().tables[1])
print(f"\nR-squared: {model1.rsquared:.4f}")
print(f"Interpretation: A 1% increase in S&P 500 is associated with a "
      f"{model1.params['sp500_pct_change_7day']:.4f} change in log(bid)")

# Model 2: Multiple time periods
print("\n\nMODEL 2: Multiple Time Period Effects")
print("-" * 70)
model2 = smf.ols('''log_bid_usd ~ sp500_pct_change_7day + 
                                    sp500_pct_change_14day + 
                                    sp500_pct_change_21day + 
                                    sp500_pct_change_30day''', data=df_clean).fit()
print(model2.summary().tables[1])
print(f"\nR-squared: {model2.rsquared:.4f}")

# Model 3: All market cap indices
print("\n\nMODEL 3: Market Cap Indices Effect")
print("-" * 70)
df_clean_bonds = df_clean[df_clean['bloomberg_agg_bond_pct_change_2day'].notna()]
if len(df_clean_bonds) > 1000:
    model3 = smf.ols('''log_bid_usd ~ sp500_large_cap_pct_change_2day + 
                                        sp400_mid_cap_pct_change_2day + 
                                        sp600_small_cap_pct_change_2day +
                                        bloomberg_agg_bond_pct_change_2day''', 
                     data=df_clean_bonds).fit()
    print(model3.summary().tables[1])
    print(f"\nR-squared: {model3.rsquared:.4f}")
else:
    print("Not enough data with bond information for this model")

print("\n" + "="*90)
print("ANALYSIS 3: Market Conditions vs Average Bid Amounts")
print("="*90)

# Group by market performance categories
df_clean['market_condition'] = pd.cut(
    df_clean['sp500_pct_change_7day'],
    bins=[-100, -2, 0, 2, 100],
    labels=['Strong Decline (< -2%)', 'Slight Decline (0 to -2%)', 
            'Slight Increase (0 to 2%)', 'Strong Increase (> 2%)']
)

market_analysis = df_clean.groupby('market_condition', observed=True).agg({
    'bid_usd': ['count', 'mean', 'median', 'std'],
    'sp500_pct_change_7day': 'mean'
}).round(2)

print("\nBid Statistics by Market Condition:")
print("-" * 70)
print(market_analysis)

print("\n" + "="*90)
print("ANALYSIS 4: Time Period Analysis")
print("="*90)

# Compare different holding periods
print("\nAverage Bid Amount by S&P 500 Performance Period:")
print("-" * 70)

for period in ['7day', '14day', '21day', '30day']:
    col = f'sp500_pct_change_{period}'
    if col in df_clean.columns:
        # Divide into positive and negative market moves
        positive_market = df_clean[df_clean[col] > 0]['bid_usd'].mean()
        negative_market = df_clean[df_clean[col] <= 0]['bid_usd'].mean()
        
        print(f"\n{period.upper()} Period:")
        print(f"  Avg bid when S&P UP:   ${positive_market:>12,.2f}")
        print(f"  Avg bid when S&P DOWN: ${negative_market:>12,.2f}")
        print(f"  Difference:            ${positive_market - negative_market:>12,.2f}")
        
        # T-test
        positive_bids = df_clean[df_clean[col] > 0]['bid_usd']
        negative_bids = df_clean[df_clean[col] <= 0]['bid_usd']
        t_stat, p_value = stats.ttest_ind(positive_bids, negative_bids)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"  T-test p-value: {p_value:.4f} {sig}")

print("\n" + "="*90)
print("ANALYSIS 5: High-Value vs Low-Value Lots")
print("="*90)

# Split by bid amount quartiles
df_clean['bid_quartile'] = pd.qcut(df_clean['bid_usd'], q=4, 
                                     labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])

print("\nCorrelation with S&P 500 by Bid Size:")
print("-" * 70)
for quartile in ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']:
    subset = df_clean[df_clean['bid_quartile'] == quartile]
    if len(subset) > 10:
        corr = subset[['bid_usd', 'sp500_pct_change_7day']].corr().iloc[0, 1]
        print(f"  {quartile}: {corr:>7.4f}")

print("\n" + "="*90)
print("SUMMARY OF FINDINGS")
print("="*90)

# Calculate key statistics for summary
overall_corr = df_clean[['bid_usd', 'sp500_pct_change_7day']].corr().iloc[0, 1]
significant_vars = sum(1 for c in correlations if c['P-value'] < 0.05)

print(f"""
KEY FINDINGS:

1. CORRELATION STRENGTH:
   - Overall correlation (S&P 500 7-day vs Bid Amount): {overall_corr:.4f}
   - Significant variables (p < 0.05): {significant_vars} out of {len(correlations)}
   
2. STATISTICAL SIGNIFICANCE:
   - Model 1 R-squared: {model1.rsquared:.4f}
   - Model 2 R-squared: {model2.rsquared:.4f}
   
3. PRACTICAL SIGNIFICANCE:
   - Bids when market UP:   ${df_clean[df_clean['sp500_pct_change_7day'] > 0]['bid_usd'].mean():,.2f}
   - Bids when market DOWN: ${df_clean[df_clean['sp500_pct_change_7day'] <= 0]['bid_usd'].mean():,.2f}
   - Difference: ${df_clean[df_clean['sp500_pct_change_7day'] > 0]['bid_usd'].mean() - df_clean[df_clean['sp500_pct_change_7day'] <= 0]['bid_usd'].mean():,.2f}

4. CONCLUSION:
   {"Stock market performance DOES appear to affect art auction prices." if abs(overall_corr) > 0.02 
    else "Stock market performance shows WEAK effect on art auction prices."}
   
   The relationship is {"statistically significant" if model1.pvalues['sp500_pct_change_7day'] < 0.05 
    else "not statistically significant"} at the 5% level.
""")

print("="*90)

# Save detailed results
results_summary = pd.DataFrame(correlations)
results_summary.to_csv('stock_impact_analysis_results.csv', index=False)
print("\n✓ Detailed results saved to stock_impact_analysis_results.csv")

print("\n✓ Analysis complete!")
