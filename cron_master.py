"""
Master Cron Job Orchestrator
Runs all cron jobs in the correct sequence with proper error handling

Execution Order:
1. cron_saffron.py       - Scrape new auction data → S3
2. cron_mongo_upload.py  - Process images & embeddings → MongoDB
3. cron_bid_scraper.py   - Scrape bid history → MongoDB
4. cron_regenerate.py    - Generate similarity matches → MongoDB
5. StockAnalysis.py      - Run stock market analysis → S3 (NEW)
6. cron_file_saver.py    - Export data to CSV → S3
7. cron_emailer.py       - Send email notifications

Each step logs its execution and only proceeds if previous steps succeed.
"""

import os
import sys
import logging
import subprocess
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cron_master.log'),
        logging.StreamHandler()
    ]
)

# Define the execution order
CRON_JOBS = [
    {
        'name': 'Saffron Scraper',
        'script': 'cron_saffron.py',
        'description': 'Scrape new auction data from Saffron Art',
        'critical': True  # If this fails, stop pipeline
    },
    {
        'name': 'MongoDB Upload',
        'script': 'cron_mongo_upload.py',
        'description': 'Process images and upload to MongoDB with embeddings',
        'critical': True
    },
    {
        'name': 'Bid History Scraper',
        'script': 'cron_bid_scraper.py',
        'description': 'Scrape detailed bid history for lots',
        'critical': False  # Continue even if bid scraping fails
    },
    {
        'name': 'Match Regenerator',
        'script': 'cron_regenerate.py',
        'description': 'Regenerate similarity matches using Pinecone',
        'critical': False
    },
    {
        'name': 'Stock Market Analysis',
        'script': 'StockAnalysis.py',
        'description': 'Run stock market impact and hedonic pricing analysis',
        'critical': False  # NEW: Analysis job
    },
    {
        'name': 'File Saver',
        'script': 'cron_file_saver.py',
        'description': 'Export data to CSV and upload to S3',
        'critical': False
    },
    {
        'name': 'Email Notifier',
        'script': 'cron_emailer.py',
        'description': 'Send email notifications to subscribers',
        'critical': False
    }
]


def run_python_script(script_name):
    """
    Execute a Python script and return success status
    
    Args:
        script_name: Name of the Python script to run
        
    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    try:
        logging.info(f"Executing: {script_name}")
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        # Log output
        if result.stdout:
            logging.info(f"Output from {script_name}:\n{result.stdout}")
        
        if result.stderr:
            logging.warning(f"Errors from {script_name}:\n{result.stderr}")
        
        # Check return code
        if result.returncode == 0:
            logging.info(f"✓ {script_name} completed successfully")
            return True, result.stdout, result.stderr
        else:
            logging.error(f"✗ {script_name} failed with return code {result.returncode}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        logging.error(f"✗ {script_name} timed out after 2 hours")
        return False, "", "Script execution timed out"
    except FileNotFoundError:
        logging.error(f"✗ {script_name} not found")
        return False, "", "Script file not found"
    except Exception as e:
        logging.error(f"✗ Error executing {script_name}: {str(e)}")
        logging.error(traceback.format_exc())
        return False, "", str(e)


def main():
    """Main orchestrator function"""
    start_time = datetime.now()
    
    logging.info("="*100)
    logging.info("SAFFRON ART DATA PIPELINE - MASTER CRON JOB")
    logging.info("="*100)
    logging.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total Jobs: {len(CRON_JOBS)}")
    logging.info("="*100)
    
    # Track execution results
    results = []
    failed_critical = False
    
    # Execute each job in sequence
    for i, job in enumerate(CRON_JOBS, 1):
        logging.info("")
        logging.info(f"[{i}/{len(CRON_JOBS)}] Starting: {job['name']}")
        logging.info(f"Description: {job['description']}")
        logging.info(f"Script: {job['script']}")
        logging.info(f"Critical: {job['critical']}")
        logging.info("-"*100)
        
        job_start = datetime.now()
        
        # Execute the job
        success, stdout, stderr = run_python_script(job['script'])
        
        job_end = datetime.now()
        duration = (job_end - job_start).total_seconds()
        
        # Record result
        result = {
            'job_number': i,
            'name': job['name'],
            'script': job['script'],
            'success': success,
            'duration_seconds': duration,
            'critical': job['critical']
        }
        results.append(result)
        
        logging.info(f"Duration: {duration:.2f} seconds")
        
        # Check if we should continue
        if not success and job['critical']:
            logging.error(f"CRITICAL JOB FAILED: {job['name']}")
            logging.error("Stopping pipeline execution due to critical failure")
            failed_critical = True
            break
        elif not success:
            logging.warning(f"Non-critical job failed: {job['name']}. Continuing...")
        
        logging.info("-"*100)
    
    # Final summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    logging.info("")
    logging.info("="*100)
    logging.info("PIPELINE EXECUTION SUMMARY")
    logging.info("="*100)
    logging.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total Duration: {total_duration/60:.2f} minutes ({total_duration:.0f} seconds)")
    logging.info("")
    
    # Job-by-job summary
    successful_jobs = sum(1 for r in results if r['success'])
    failed_jobs = len(results) - successful_jobs
    
    logging.info(f"Jobs Executed: {len(results)}/{len(CRON_JOBS)}")
    logging.info(f"Successful: {successful_jobs}")
    logging.info(f"Failed: {failed_jobs}")
    logging.info("")
    
    logging.info("Job Details:")
    logging.info("-"*100)
    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        critical = "(CRITICAL)" if result['critical'] else ""
        logging.info(f"{result['job_number']:2d}. {result['name']:30s} {status:15s} {critical:12s} ({result['duration_seconds']:>6.1f}s)")
    
    # Overall status
    logging.info("")
    logging.info("="*100)
    if failed_critical:
        logging.error("⚠️  PIPELINE FAILED - Critical job failure")
        logging.info("="*100)
        sys.exit(1)
    elif failed_jobs > 0:
        logging.warning("⚠️  PIPELINE COMPLETED WITH WARNINGS - Some non-critical jobs failed")
        logging.info("="*100)
        sys.exit(0)
    else:
        logging.info("✅ PIPELINE COMPLETED SUCCESSFULLY - All jobs executed without errors")
        logging.info("="*100)
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"\n\nUnexpected error in master orchestrator: {str(e)}")
        logging.critical(traceback.format_exc())
        sys.exit(1)
