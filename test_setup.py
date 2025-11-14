"""
Test Script for Stock Analysis Pipeline
Verifies configuration and dependencies before running the full pipeline
"""

import sys
import os

print("="*90)
print("STOCK ANALYSIS PIPELINE - CONFIGURATION TEST")
print("="*90)

# Test 1: Check Python version
print("\n1. Python Version:")
print(f"   {sys.version}")
if sys.version_info < (3, 8):
    print("   ⚠️  Warning: Python 3.8+ recommended")
else:
    print("   ✓ Compatible")

# Test 2: Check required packages
print("\n2. Checking Required Packages:")
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'pymongo': 'pymongo',
    'boto3': 'boto3',
    'python-dotenv': 'dotenv',
    'statsmodels': 'statsmodels',
    'scipy': 'scipy',
    'requests': 'requests',
    'beautifulsoup4': 'bs4',
    'selenium': 'selenium',
    'pinecone': 'pinecone',
    'tensorflow': 'tensorflow',
    'pillow': 'PIL'
}

missing_packages = []
for package, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ⚠️  Missing packages: {', '.join(missing_packages)}")
    print("   Install with: pip install -r requirements.txt")
else:
    print("\n   ✓ All packages installed")

# Test 3: Check environment file
print("\n3. Environment Configuration:")
if os.path.exists('.env'):
    print("   ✓ .env file exists")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    env_vars = [
        'MONGO_URI',
        'DB_NAME',
        'SAFFRON_BID_COLLECTION_NAME',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY'
    ]
    
    missing_vars = []
    for var in env_vars:
        if os.getenv(var):
            print(f"   ✓ {var} is set")
        else:
            print(f"   ✗ {var} - NOT SET")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n   ⚠️  Missing variables: {', '.join(missing_vars)}")
    else:
        print("\n   ✓ All required environment variables set")
else:
    print("   ✗ .env file not found")
    print("   Create a .env file with required configuration")

# Test 4: Check MongoDB connection
print("\n4. MongoDB Connection:")
try:
    from pymongo import MongoClient
    from dotenv import load_dotenv
    load_dotenv()
    
    MONGO_URI = os.getenv('MONGO_URI')
    if MONGO_URI:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        print("   ✓ MongoDB connection successful")
        
        # Check collections
        db_name = os.getenv('DB_NAME')
        if db_name:
            db = client[db_name]
            collections = db.list_collection_names()
            print(f"   ✓ Database '{db_name}' accessible")
            print(f"   Collections found: {len(collections)}")
            
            bid_collection = os.getenv('SAFFRON_BID_COLLECTION_NAME')
            if bid_collection in collections:
                count = db[bid_collection].count_documents({})
                print(f"   ✓ Bid collection '{bid_collection}' exists ({count:,} documents)")
            else:
                print(f"   ⚠️  Bid collection '{bid_collection}' not found")
        
        client.close()
    else:
        print("   ✗ MONGO_URI not set")
except Exception as e:
    print(f"   ✗ MongoDB connection failed: {str(e)}")

# Test 5: Check AWS S3 connection
print("\n5. AWS S3 Connection:")
try:
    import boto3
    from dotenv import load_dotenv
    load_dotenv()
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    # Try to list buckets
    buckets = s3_client.list_buckets()
    print(f"   ✓ AWS S3 connection successful")
    print(f"   Buckets accessible: {len(buckets['Buckets'])}")
    
    # Check if target bucket exists
    bucket_names = [b['Name'] for b in buckets['Buckets']]
    if 'scraped-art-data' in bucket_names:
        print("   ✓ Target bucket 'scraped-art-data' exists")
    else:
        print("   ⚠️  Target bucket 'scraped-art-data' not found")
except Exception as e:
    print(f"   ✗ AWS S3 connection failed: {str(e)}")

# Test 6: Check required scripts exist
print("\n6. Pipeline Scripts:")
scripts = [
    'cron_master.py',
    'StockAnalysis.py',
    'cron_saffron.py',
    'cron_mongo_upload.py',
    'cron_bid_scraper.py',
    'cron_regenerate.py',
    'cron_file_saver.py',
    'cron_emailer.py'
]

for script in scripts:
    if os.path.exists(script):
        print(f"   ✓ {script}")
    else:
        print(f"   ✗ {script} - NOT FOUND")

# Test 7: Check output directory
print("\n7. Output Directory:")
if not os.path.exists('./analysis_outputs'):
    os.makedirs('./analysis_outputs')
    print("   ✓ Created ./analysis_outputs/")
else:
    print("   ✓ ./analysis_outputs/ exists")

# Test 8: Test statsmodels
print("\n8. Testing Statistical Packages:")
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats
    print("   ✓ statsmodels and scipy working")
except Exception as e:
    print(f"   ✗ Error with statistical packages: {str(e)}")

# Summary
print("\n" + "="*90)
print("SUMMARY")
print("="*90)

if not missing_packages and os.path.exists('.env'):
    print("✅ System appears ready for pipeline execution")
    print("\nTo run the pipeline:")
    print("  python cron_master.py          # Run complete pipeline")
    print("  python StockAnalysis.py        # Run analysis only")
else:
    print("⚠️  Configuration issues detected. Please resolve before running.")
    if missing_packages:
        print("\n  Install missing packages:")
        print("  pip install -r requirements.txt")
    if not os.path.exists('.env'):
        print("\n  Create .env file with required configuration")

print("="*90)
