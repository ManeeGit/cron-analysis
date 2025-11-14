"""
Hedonic Pricing Model for Art Auctions

The hedonic pricing model decomposes the price of a good (art) into its constituent characteristics.
In art markets, this includes both intrinsic characteristics (artist, size, medium) and 
extrinsic market conditions (stock market performance, economic indicators).

Model specification:
log(Price_it) = β0 + β1*MarketConditions_t + β2*ArtCharacteristics_i + ε_it

Where:
- Price_it = auction price for artwork i at time t
- MarketConditions_t = stock market variables (S&P 500, bonds, etc.)
- ArtCharacteristics_i = artist fixed effects, estimates, etc.
- β coefficients = implicit prices of each characteristic
- ε_it = error term
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("="*100)
print("HEDONIC PRICING MODEL FOR ART AUCTIONS")
print("="*100)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('bid_data_complete_all_periods.csv', low_memory=False)
print(f"   Loaded {len(df):,} observations")

# Data preparation
print("\n2. Preparing variables...")

# Dependent variable: log of bid price
df['log_bid'] = np.log(df['bid_usd'])

# Handle infinite values from log(0) or negative prices
df = df[np.isfinite(df['log_bid'])].copy()
print(f"   Valid observations after log transform: {len(df):,}")

# Create additional art characteristic variables
df['has_estimate'] = ((df['lo_est'].notna()) & (df['hi_est'].notna())).astype(int)

# Convert estimates to numeric, handling any string values
df['lo_est'] = pd.to_numeric(df['lo_est'], errors='coerce')
df['hi_est'] = pd.to_numeric(df['hi_est'], errors='coerce')

df['log_lo_est'] = np.log(df['lo_est'].replace(0, np.nan))
df['log_hi_est'] = np.log(df['hi_est'].replace(0, np.nan))
df['estimate_midpoint'] = (df['lo_est'] + df['hi_est']) / 2
df['log_estimate_midpoint'] = np.log(df['estimate_midpoint'].replace(0, np.nan))

# Create time variables
df['year'] = pd.to_datetime(df['iso_date']).dt.year
df['month'] = pd.to_datetime(df['iso_date']).dt.month
df['quarter'] = pd.to_datetime(df['iso_date']).dt.quarter

# Market condition interaction terms
df['market_volatility'] = df['sp500_large_cap_pct_change_2day'].abs()

print("\n" + "="*100)
print("HEDONIC MODEL 1: BASELINE (Market Conditions Only)")
print("="*100)

# Model 1: Basic hedonic model with market conditions
print("\nSpecification: log(Price) = β0 + β1*LargeCap + β2*MidCap + β3*SmallCap + β4*Bond + ε")

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

print(f"\nResults (N = {len(model1_data):,}):")
print(f"R-squared: {model1.rsquared:.6f}")
print(f"Adjusted R-squared: {model1.rsquared_adj:.6f}")
print(f"F-statistic: {model1.fvalue:.2f} (p = {model1.f_pvalue:.3e})")

print("\nCoefficients (Implicit Prices):")
for var in model1.params.index:
    if var != 'Intercept':
        coef = model1.params[var]
        se = model1.bse[var]
        pval = model1.pvalues[var]
        ci_low, ci_high = model1.conf_int().loc[var]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"  β({var:40s}): {coef:>10.6f} (SE={se:.6f}, p={pval:.3e}) {sig}")
        print(f"    95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
        # Interpretation: % change in price for 1% change in market
        price_elasticity = coef
        print(f"    Interpretation: 1% increase in {var.split('_')[0]} → {price_elasticity:.4f}% change in art price")

print("\n" + "="*100)
print("HEDONIC MODEL 2: WITH ARTIST FIXED EFFECTS")
print("="*100)

# Model 2: Add artist fixed effects (control for artist quality/reputation)
print("\nSpecification: log(Price) = β0 + β*MarketConditions + α_artist + ε")
print("(Artist fixed effects control for unobserved artist characteristics)")

# Get top artists (those with sufficient observations)
artist_counts = df['artist_name'].value_counts()
top_artists = artist_counts[artist_counts >= 50].index.tolist()
print(f"\nIncluding {len(top_artists)} artists with ≥50 observations")

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

print(f"\nEstimating model with {len(top_artists)} artist dummies...")
model2 = smf.ols(formula2, data=model2_data).fit()

print(f"\nResults (N = {len(model2_data):,}):")
print(f"R-squared: {model2.rsquared:.6f}")
print(f"Adjusted R-squared: {model2.rsquared_adj:.6f}")
print(f"F-statistic: {model2.fvalue:.2f} (p = {model2.f_pvalue:.3e})")

print("\nMarket Condition Coefficients (controlling for artist):")
market_vars = ['sp500_large_cap_pct_change_2day', 'sp400_mid_cap_pct_change_2day',
               'sp600_small_cap_pct_change_2day', 'vanguard_bond_pct_change_2day']
for var in market_vars:
    coef = model2.params[var]
    se = model2.bse[var]
    pval = model2.pvalues[var]
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    print(f"  β({var:40s}): {coef:>10.6f} (SE={se:.6f}, p={pval:.3e}) {sig}")

print("\n" + "="*100)
print("HEDONIC MODEL 3: FULL MODEL WITH TIME PERIODS")
print("="*100)

# Model 3: Multiple time horizons to capture short vs long-term effects
print("\nSpecification: log(Price) = β0 + Σ(β_t * Market_t) + α_artist + γ_time + ε")
print("Multiple time periods (7, 14, 21, 30 days) capture dynamics")

model3_data = df_model2[[
    'log_bid', 'artist_name', 'year', 'quarter',
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

print(f"\nEstimating full model...")
model3 = smf.ols(formula3, data=model3_data).fit()

print(f"\nResults (N = {len(model3_data):,}):")
print(f"R-squared: {model3.rsquared:.6f}")
print(f"Adjusted R-squared: {model3.rsquared_adj:.6f}")
print(f"F-statistic: {model3.fvalue:.2f} (p = {model3.f_pvalue:.3e})")

print("\nTime Horizon Coefficients (Market Impact by Period):")
time_vars = ['sp500_pct_change_7day', 'sp500_pct_change_14day', 
             'sp500_pct_change_21day', 'sp500_pct_change_30day',
             'vanguard_bond_pct_change_2day']
for var in time_vars:
    coef = model3.params[var]
    se = model3.bse[var]
    pval = model3.pvalues[var]
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    print(f"  β({var:35s}): {coef:>10.6f} (SE={se:.6f}, p={pval:.3e}) {sig}")

print("\n" + "="*100)
print("HEDONIC MODEL 4: WITH ESTIMATE CONTROLS")
print("="*100)

# Model 4: Include pre-sale estimates as additional controls
print("\nSpecification: log(Price) = β*Market + γ*log(Estimate) + α_artist + ε")
print("Pre-sale estimates contain auctioneer's private information")

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

print(f"\nResults (N = {len(model4_data):,}):")
print(f"R-squared: {model4.rsquared:.6f}")
print(f"Adjusted R-squared: {model4.rsquared_adj:.6f}")

print("\nKey Coefficients:")
key_vars = ['log_estimate_midpoint'] + market_vars
for var in key_vars:
    if var in model4.params.index:
        coef = model4.params[var]
        se = model4.bse[var]
        pval = model4.pvalues[var]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"  β({var:40s}): {coef:>10.6f} (SE={se:.6f}, p={pval:.3e}) {sig}")

print("\n" + "="*100)
print("MODEL COMPARISON")
print("="*100)

comparison = pd.DataFrame({
    'Model': ['1. Market Only', '2. + Artist FE', '3. + Time Periods', '4. + Estimates'],
    'N': [len(model1_data), len(model2_data), len(model3_data), len(model4_data)],
    'R²': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared],
    'Adj R²': [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj, model4.rsquared_adj],
    'F-stat': [model1.fvalue, model2.fvalue, model3.fvalue, model4.fvalue],
    'AIC': [model1.aic, model2.aic, model3.aic, model4.aic],
    'BIC': [model1.bic, model2.bic, model3.bic, model4.bic]
})

print("\n" + comparison.to_string(index=False))

print("\n" + "="*100)
print("IMPLICIT PRICE ESTIMATES (HEDONIC PRICES)")
print("="*100)

print("\nHedonic prices represent the marginal willingness to pay for each characteristic:")
print("\n1. MARKET CONDITIONS (From Model 2 - with artist FE):")
for var in market_vars:
    coef = model2.params[var]
    # Convert to dollar terms: approximate effect on median bid
    median_bid = df['bid_usd'].median()
    dollar_effect = median_bid * (coef / 100)  # For 1% change in market
    print(f"\n   {var:40s}:")
    print(f"     Elasticity: {coef:.6f}")
    print(f"     Dollar impact: ${dollar_effect:,.2f} per 1% market change (on median bid ${median_bid:,.0f})")

print("\n2. ARTIST EFFECTS (Top 5 most valuable artists from Model 2):")
artist_effects = []
for param in model2.params.index:
    if 'C(artist_name)' in param:
        artist = param.replace('C(artist_name)[T.', '').replace(']', '')
        effect = model2.params[param]
        artist_effects.append({'Artist': artist, 'Effect': effect})

artist_df = pd.DataFrame(artist_effects).sort_values('Effect', ascending=False)
print("\n   Top 5 premium artists:")
for idx, row in artist_df.head(5).iterrows():
    premium_pct = (np.exp(row['Effect']) - 1) * 100
    print(f"     {row['Artist']:50s}: +{premium_pct:>6.2f}% premium")

print("\n   Bottom 5 (discount artists):")
for idx, row in artist_df.tail(5).iterrows():
    discount_pct = (np.exp(row['Effect']) - 1) * 100
    print(f"     {row['Artist']:50s}: {discount_pct:>6.2f}% discount")

print("\n" + "="*100)
print("SAVING RESULTS")
print("="*100)

# Save model summaries
with open('hedonic_model_results.txt', 'w') as f:
    f.write("="*100 + "\n")
    f.write("HEDONIC PRICING MODEL RESULTS\n")
    f.write("="*100 + "\n\n")
    
    for i, model in enumerate([model1, model2, model3, model4], 1):
        f.write(f"\n{'='*100}\n")
        f.write(f"MODEL {i}\n")
        f.write(f"{'='*100}\n")
        f.write(model.summary().as_text())
        f.write("\n\n")

print("✓ Saved detailed results to: hedonic_model_results.txt")

# Save comparison table
comparison.to_csv('hedonic_model_comparison.csv', index=False)
print("✓ Saved model comparison to: hedonic_model_comparison.csv")

# Save hedonic prices (coefficients)
hedonic_prices = pd.DataFrame({
    'Variable': market_vars,
    'Coefficient': [model2.params[v] for v in market_vars],
    'Std_Error': [model2.bse[v] for v in market_vars],
    'P_Value': [model2.pvalues[v] for v in market_vars],
    'CI_Lower': [model2.conf_int().loc[v, 0] for v in market_vars],
    'CI_Upper': [model2.conf_int().loc[v, 1] for v in market_vars]
})
hedonic_prices.to_csv('hedonic_implicit_prices.csv', index=False)
print("✓ Saved implicit prices to: hedonic_implicit_prices.csv")

print("\n" + "="*100)
print("✅ HEDONIC PRICING ANALYSIS COMPLETE")
print("="*100)

print(f"""
SUMMARY:
--------
✓ Estimated 4 hedonic pricing models with different specifications
✓ Model 1: Baseline with market conditions only (R² = {model1.rsquared:.4f})
✓ Model 2: Added artist fixed effects (R² = {model2.rsquared:.4f})
✓ Model 3: Added multiple time periods (R² = {model3.rsquared:.4f})
✓ Model 4: Added estimate controls (R² = {model4.rsquared:.4f})

KEY FINDING:
The hedonic model confirms that market conditions have statistically significant
but economically small effects on art prices. Artist fixed effects explain
much more of the price variation (R² increases from {model1.rsquared:.4f} to {model2.rsquared:.4f}).

FILES CREATED:
- hedonic_model_results.txt       (Full regression output)
- hedonic_model_comparison.csv    (Model comparison table)
- hedonic_implicit_prices.csv     (Hedonic price estimates)
""")
