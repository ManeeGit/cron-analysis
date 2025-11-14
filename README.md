# Saffron Art Data Pipeline - Stock Market Analysis Integration

## Overview
This repository contains automated cron jobs for scraping, processing, and analyzing art auction data from Saffron Art, with integrated stock market impact analysis.

## 🆕 New Addition: Stock Market Analysis

### What's New?
A comprehensive stock market analysis module has been added to the pipeline:
- **File**: `StockAnalysis.py`
- **Purpose**: Analyzes the relationship between stock market performance and art auction prices
- **Runs**: Automatically after data is loaded into MongoDB

### Analysis Components

#### 1. Stock Market Impact Analysis
Examines correlations and causal relationships between:
- S&P 500 performance (large, mid, small cap indices)
- Bond market performance
- Art auction bid amounts

**Outputs:**
- `stock_impact_correlations.csv` - Correlation coefficients with significance tests
- `market_condition_analysis.csv` - Bid statistics grouped by market conditions
- `stock_impact_summary.csv` - Overall summary statistics
- `stock_impact_regression_results.txt` - Detailed regression output

#### 2. Hedonic Pricing Model
Decomposes art prices into constituent characteristics:
- Market conditions (stocks, bonds)
- Artist fixed effects
- Pre-sale estimates
- Time period effects

**Outputs:**
- `hedonic_model_comparison.csv` - Comparison of 4 model specifications
- `hedonic_implicit_prices.csv` - Marginal prices for each characteristic
- `hedonic_artist_effects.csv` - Artist premium/discount effects
- `hedonic_model_detailed_results.txt` - Full regression tables

### S3 Storage & Download Links
All analysis outputs are automatically:
1. Saved locally to `./analysis_outputs/`
2. Uploaded to S3 bucket: `scraped-art-data`
3. Listed in `download_links.csv` with public URLs

### Integration with Existing Pipeline

The analysis runs as **Step 5** in the pipeline sequence:

```
1. cron_saffron.py       → Scrape auction data → S3
2. cron_mongo_upload.py  → Process images → MongoDB + Pinecone
3. cron_bid_scraper.py   → Scrape bid history → MongoDB
4. cron_regenerate.py    → Generate matches → MongoDB
5. StockAnalysis.py      → Run analysis → S3 ⭐ NEW
6. cron_file_saver.py    → Export CSVs → S3
7. cron_emailer.py       → Email subscribers
```

## Running the Pipeline

### Option 1: Run Complete Pipeline
```bash
python cron_master.py
```
This executes all jobs in sequence with automatic error handling.

### Option 2: Run Analysis Only
```bash
python StockAnalysis.py
```
This runs only the stock market analysis (requires data already in MongoDB).

### Option 3: Run Individual Jobs
```bash
python cron_saffron.py       # Scraping
python cron_mongo_upload.py  # Upload
python StockAnalysis.py      # Analysis
# ... etc
```

## Requirements

### New Dependencies Added
- `statsmodels==0.14.1` - For regression analysis

All dependencies are in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

## Environment Variables
Required in `.env` file:
```bash
# MongoDB
MONGO_URI=mongodb://...
DB_NAME=your_database
COLLECTION_NAME=art_collection
SAFFRON_BID_COLLECTION_NAME=saffron_bid_data

# AWS S3
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Email (for notifications)
EMAIL_USER=your_email
EMAIL_PASSWORD=your_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Pinecone
PINECONE_API_KEY=your_key
PINECONE_INDEX=your_index
```

## Output Files

### Analysis Outputs (saved to S3)
| File | Description |
|------|-------------|
| `bid_data_complete_all_periods.csv` | Raw bid data with stock market variables |
| `stock_impact_correlations.csv` | Correlation analysis results |
| `market_condition_analysis.csv` | Bid statistics by market condition |
| `stock_impact_summary.csv` | Summary statistics |
| `stock_impact_regression_results.txt` | Full regression output |
| `hedonic_model_comparison.csv` | Model comparison table |
| `hedonic_implicit_prices.csv` | Hedonic prices (implicit values) |
| `hedonic_artist_effects.csv` | Artist premium/discount effects |
| `hedonic_model_detailed_results.txt` | Detailed regression tables |
| `download_links.csv` | All S3 download URLs |

### Regular Pipeline Outputs
| File | Description |
|------|-------------|
| `similarities.csv` | Art collection with similarity matches |
| `bid_data.csv` | Raw bid history data |
| `transformed_bid_data.csv` | Bid data with corrected timestamps |

## Logging
Each component logs to its own file:
- `cron_master.log` - Master orchestrator logs
- `stock_analysis.log` - Analysis execution logs
- `saffron_scraper.log` - Scraping logs
- `mongo_upload.log` - Upload logs
- `bid_data.log` - Bid scraper logs
- `regenerate_matches.log` - Match regeneration logs
- `email_distribution.log` - Email logs

## Scheduling with Cron

To run automatically, add to crontab:
```bash
# Run every day at 2 AM
0 2 * * * cd /path/to/saffron-cron-jobs && /usr/bin/python3 cron_master.py

# Or run analysis only at 6 AM
0 6 * * * cd /path/to/saffron-cron-jobs && /usr/bin/python3 StockAnalysis.py
```

## Analysis Methodology

### Stock Impact Analysis
Tests the hypothesis: "Does stock market performance affect art auction prices?"

**Methods:**
1. Correlation analysis with significance tests
2. OLS regression (log-linear specification)
3. Market condition grouping (strong/slight increase/decline)
4. Time period analysis (7, 14, 21, 30-day windows)

### Hedonic Pricing Model
Estimates implicit prices for art characteristics using regression:

```
log(Price) = β₀ + β₁*MarketConditions + β₂*ArtistFE + β₃*Estimates + ε
```

**Four Model Specifications:**
1. Market conditions only (baseline)
2. + Artist fixed effects
3. + Multiple time periods
4. + Pre-sale estimate controls

## Key Findings
The analysis provides:
- Statistical significance of market effects (p-values, R²)
- Economic significance (dollar impact estimates)
- Artist-specific premiums/discounts
- Time-varying effects of market conditions

## Troubleshooting

### Analysis Fails
- Check MongoDB connection
- Ensure bid data exists in `saffron_bid_data` collection
- Verify stock market columns exist in data

### S3 Upload Fails
- Verify AWS credentials in `.env`
- Check S3 bucket permissions
- Ensure bucket name is correct

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## File Structure
```
saffron-cron-jobs/
├── cron_master.py              # Master orchestrator ⭐ NEW
├── StockAnalysis.py            # Stock market analysis ⭐ NEW
├── cron_saffron.py             # Main scraper
├── cron_mongo_upload.py        # Image processing
├── cron_bid_scraper.py         # Bid history scraper
├── cron_regenerate.py          # Match regenerator
├── cron_file_saver.py          # CSV exporter
├── cron_emailer.py             # Email notifier
├── requirements.txt            # Dependencies (updated)
├── .env                        # Environment variables
├── README.md                   # This file ⭐ NEW
├── hedonic_pricing_model.py    # Original analysis script
├── analyze_stock_impact_on_art.py  # Original analysis script
└── analysis_outputs/           # Analysis output directory ⭐ NEW
    ├── *.csv
    └── *.txt
```

## Support
For issues or questions, check the log files first:
- `cron_master.log` - Overall pipeline status
- `stock_analysis.log` - Analysis-specific issues

## License
[Your License Here]

## Contributors
[Your Team/Name Here]

---
**Last Updated**: November 2025
**Version**: 2.0 (with Stock Market Analysis)
