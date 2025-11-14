# Quick Start Guide - Stock Market Analysis Integration

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup
```bash
python test_setup.py
```
This will check:
- ✅ All required packages installed
- ✅ Environment variables configured
- ✅ MongoDB connection working
- ✅ AWS S3 access working

### Step 3: Run the Pipeline
```bash
python cron_master.py
```

---

## 📊 What You'll Get

After running the pipeline, you'll receive analysis files uploaded to S3:

### Stock Market Impact Analysis
1. **Correlations** - How stock market changes correlate with art prices
2. **Market Conditions** - Bid statistics when markets are up/down
3. **Regression Results** - Statistical models showing relationships
4. **Summary Statistics** - Key findings in one file

### Hedonic Pricing Analysis
1. **Model Comparison** - 4 different model specifications
2. **Implicit Prices** - Marginal value of market conditions
3. **Artist Effects** - Which artists command premiums
4. **Detailed Results** - Full regression tables

### Download Links
All files are available via S3 URLs in:
- `download_links.csv` - Master list of all download URLs
- `links.txt` - Formatted for email distribution

---

## 🔄 Pipeline Sequence

```
START
  ↓
1️⃣ Scrape new auctions (cron_saffron.py)
  ↓
2️⃣ Process images → MongoDB (cron_mongo_upload.py)
  ↓
3️⃣ Scrape bid history (cron_bid_scraper.py)
  ↓
4️⃣ Generate matches (cron_regenerate.py)
  ↓
5️⃣ Run stock analysis → S3 ⭐ NEW (StockAnalysis.py)
  ↓
6️⃣ Export data → S3 (cron_file_saver.py)
  ↓
7️⃣ Email subscribers (cron_emailer.py)
  ↓
END
```

---

## 🎯 Run Options

### Option A: Complete Pipeline
```bash
python cron_master.py
```
Runs all 7 steps sequentially with error handling.

### Option B: Analysis Only
```bash
python StockAnalysis.py
```
Skips scraping, runs analysis on existing MongoDB data.

### Option C: Individual Components
```bash
python cron_saffron.py       # Just scraping
python StockAnalysis.py      # Just analysis
python cron_emailer.py       # Just emails
```

---

## 📁 Output Files Location

### Local (during processing)
```
./analysis_outputs/
├── bid_data_complete_all_periods.csv
├── stock_impact_correlations.csv
├── market_condition_analysis.csv
├── stock_impact_summary.csv
├── stock_impact_regression_results.txt
├── hedonic_model_comparison.csv
├── hedonic_implicit_prices.csv
├── hedonic_artist_effects.csv
├── hedonic_model_detailed_results.txt
└── download_links.csv
```

### S3 (permanent storage)
```
s3://scraped-art-data/analysis/
├── bid_data_complete_all_periods.csv
├── stock_impact_correlations.csv
├── market_condition_analysis.csv
├── stock_impact_summary.csv
├── stock_impact_regression_results.txt
├── hedonic_model_comparison.csv
├── hedonic_implicit_prices.csv
├── hedonic_artist_effects.csv
├── hedonic_model_detailed_results.txt
└── download_links.csv
```

---

## 🔍 Understanding the Analysis

### Stock Impact Analysis
**Question**: Does the stock market affect art prices?

**Method**: 
- Correlation analysis
- Regression models
- Market condition grouping

**Example Finding**:
```
When S&P 500 is UP (past 7 days):   Avg bid = $25,430
When S&P 500 is DOWN (past 7 days): Avg bid = $24,890
Difference: $540 (2.2%)
```

### Hedonic Pricing Model
**Question**: What characteristics make art more valuable?

**Method**:
- Log-linear regression
- Artist fixed effects
- Market condition variables

**Example Finding**:
```
Artist Premium:
- Top artist:    +47% premium
- Average artist: baseline
- Bottom artist: -23% discount

Market Effect:
- 1% S&P 500 increase → 0.12% art price increase
```

---

## 🐛 Troubleshooting

### "Import Error: No module named 'statsmodels'"
```bash
pip install statsmodels
# or
pip install -r requirements.txt
```

### "MongoDB connection failed"
Check your `.env` file:
```bash
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
DB_NAME=art_database
SAFFRON_BID_COLLECTION_NAME=saffron_bid_data
```

### "S3 upload failed"
Check AWS credentials in `.env`:
```bash
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
```

### "No data in analysis"
Make sure you've run the scraping steps first:
```bash
python cron_saffron.py
python cron_mongo_upload.py
python cron_bid_scraper.py
```

---

## 📧 Email Integration

The `links.txt` file is automatically generated with all download links.
It's used by `cron_emailer.py` to send results to subscribers.

To customize the email:
1. Edit `cron_emailer.py`
2. Modify the email body template
3. Update recipient list in MongoDB

---

## ⏰ Scheduling (Optional)

To run automatically every day at 2 AM:

```bash
# Edit crontab
crontab -e

# Add this line
0 2 * * * cd /path/to/saffron-cron-jobs && /usr/bin/python3 cron_master.py
```

---

## 📊 Sample Results

After running, check the logs:
```bash
tail -f cron_master.log      # Overall pipeline status
tail -f stock_analysis.log   # Analysis details
```

Expected completion times:
- Scraping: 30-60 minutes
- Image processing: 15-30 minutes
- Bid scraping: 20-40 minutes
- Match generation: 10-20 minutes
- **Stock analysis: 5-10 minutes** ⭐ NEW
- File export: 5-10 minutes
- Email: 1-2 minutes

**Total: ~2-3 hours**

---

## ✅ Success Indicators

Look for these in the logs:

```
✓ Fetched 15,234 records from MongoDB
✓ Saved correlations to stock_impact_correlations.csv
✓ Model 1 R-squared: 0.0234
✓ Uploaded to S3: https://scraped-art-data.s3.amazonaws.com/...
✓ Links written to links.txt for email distribution
✅ ANALYSIS COMPLETE
```

---

## 🆘 Need Help?

1. Run the test script: `python test_setup.py`
2. Check log files in the project directory
3. Review the full README.md for details
4. Verify .env configuration

---

## 🎉 You're All Set!

The pipeline will now:
- ✅ Scrape new auction data automatically
- ✅ Process and analyze the data
- ✅ Run stock market impact analysis
- ✅ Upload results to S3
- ✅ Email download links to subscribers

Run `python cron_master.py` to start! 🚀
