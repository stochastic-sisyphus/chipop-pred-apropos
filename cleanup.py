# cleanup.py
import os
from datetime import datetime, timedelta

def cleanup_old_runs():
    """Keep only last 3 runs and runs less than 30 days old"""
    runs = sorted(os.listdir('output'))
    keep_runs = runs[-3:]  # Keep last 3
    
    for run in runs:
        if run not in keep_runs:
            run_date = datetime.strptime(run, '%Y-%m-%d')
            if datetime.now() - run_date > timedelta(days=30):
                print(f"Archiving old run: {run}")
                os.system(f"zip -r archives/{run}.zip output/{run}")
                os.system(f"rm -rf output/{run}")

if __name__ == "__main__":
    cleanup_old_runs()
