# Automatic Data Refresh System

## Overview

The Chicago Housing Pipeline now includes an intelligent automatic data refresh system that keeps your data fresh without manual intervention. This system monitors data staleness, schedules regular updates, and provides background refreshing capabilities.

## Features

### 🔄 **Intelligent Staleness Detection**
- Monitors cache file ages and determines when data becomes stale
- Different staleness thresholds for different data sources:
  - **Census data**: 1 week (updated infrequently)
  - **FRED economic data**: 1.5 days (updated daily)
  - **Chicago city data**: 18 hours (can update frequently)

### ⏰ **Scheduled Refresh**
- **FRED data**: Daily at 6:00 AM (catches fresh economic indicators)
- **Chicago data**: Daily at 6:00 PM (after business hours updates)
- **Census data**: Weekly on Sunday at 2:00 AM (minimal changes)
- **Staleness checks**: Every 30 minutes (prevents stale data usage)

### 🔍 **Background Processing**
- Runs in separate daemon thread to avoid blocking pipeline execution
- Graceful error handling and automatic retry logic
- Comprehensive logging of all refresh activities

### 📊 **Status Monitoring**
- Real-time status reporting
- Last refresh time tracking
- Data freshness indicators

## Installation

The automatic refresh system requires the `schedule` library:

```bash
pip install schedule==1.2.2
```

This has been added to `requirements.txt` automatically.

## Usage

### Command Line Interface

#### Start Auto-Refresh Daemon
```bash
python main.py --start-auto-refresh
```

#### Stop Auto-Refresh Daemon  
```bash
python main.py --stop-auto-refresh
```

#### Check Status
```bash
python main.py --refresh-status
```

#### Manual Refresh
```bash
# Refresh all datasets
python main.py --refresh-now

# Refresh specific datasets
python main.py --refresh-now census fred
python main.py --refresh-now chicago
```

### Programmatic Usage

```python
from src.data_collection.auto_refresher import AutoRefresher

# Initialize
refresher = AutoRefresher('data/cache')

# Start background refresh
refresher.start()

# Check if data is stale
if refresher.is_data_stale('fred'):
    print("FRED data needs refresh")

# Manual refresh
success = refresher.refresh_data_source('census')

# Get status
status = refresher.get_status()
print(f"Running: {status['is_running']}")

# Stop when done
refresher.stop()
```

## Configuration

### Customizing Refresh Intervals

```python
# Change refresh intervals (in hours)
refresher.configure_refresh_intervals({
    'census': 24 * 14,  # Every 2 weeks
    'fred': 12,         # Every 12 hours  
    'chicago': 6        # Every 6 hours
})
```

### Customizing Staleness Thresholds

```python
# Change staleness thresholds (in hours)
refresher.configure_staleness_thresholds({
    'census': 24 * 10,  # 10 days
    'fred': 48,         # 2 days
    'chicago': 24       # 1 day
})
```

## How It Works

### Staleness Detection Algorithm

1. **File Age Check**: Examines modification time of cache files
2. **Pattern Matching**: Uses glob patterns to find relevant cache files (`*census*`, `*fred*`, `*chicago*`)
3. **Threshold Comparison**: Compares file age against configured thresholds
4. **Intelligent Defaults**: Assumes stale if no cache files exist

### Refresh Process

1. **Cache Clearing**: Removes stale cache files for the target dataset
2. **Fresh Collection**: Calls appropriate collector to gather new data
3. **Validation**: Ensures data was successfully collected
4. **Timestamp Update**: Records successful refresh time
5. **Error Handling**: Logs failures and continues with other datasets

### Background Worker Thread

- **Non-blocking**: Runs in daemon thread, won't prevent program exit
- **Scheduled Jobs**: Uses `schedule` library for time-based triggers
- **Continuous Monitoring**: Checks staleness every 5 minutes
- **Graceful Shutdown**: Stops cleanly when requested

## Status Information

The `--refresh-status` command provides comprehensive information:

```
=== Auto-Refresh Status ===
Status: RUNNING
Available collectors: census, fred, chicago

Refresh intervals (hours):
  census: 168
  fred: 24
  chicago: 12

Data staleness:
  census: FRESH
  fred: STALE
  chicago: FRESH

Last refresh times:
  census: 2025-06-08T10:30:00.123456
  fred: 2025-06-07T15:45:00.789012
  chicago: 2025-06-08T18:00:00.456789
```

## Integration with Pipeline

The automatic refresh system integrates seamlessly with the existing pipeline:

1. **On Pipeline Start**: Auto-refresher checks for stale data
2. **Before Data Collection**: Fresh data is used if available
3. **Background Updates**: Data refreshes while pipeline runs
4. **Cache Management**: Works with existing cache system

## Error Handling

### Common Issues and Solutions

#### **No API Keys Available**
```
Error initializing collectors: No API keys found
```
**Solution**: Set API keys in `.env` file:
```bash
export CENSUS_API_KEY='your_key'
export FRED_API_KEY='your_key'  
export CHICAGO_DATA_TOKEN='your_token'
```

#### **Network Connectivity Issues**
```
Error refreshing census: Connection timeout
```
**Solution**: System will retry automatically. Check network connectivity.

#### **Rate Limiting**
```
API rate limit exceeded for FRED
```
**Solution**: System will wait and retry. Consider increasing refresh intervals.

#### **Disk Space Issues**
```
Error saving cache: No space left on device
```
**Solution**: Clean old cache files using `--clear-cache`.

## Monitoring and Logs

### Log Files
- **General logs**: `pipeline.log`
- **Auto-refresh specific**: Look for `AutoRefresher` in logs

### Key Log Messages
```
INFO - AutoRefresher - Starting automatic data refresh system
INFO - AutoRefresher - fred data is stale (age: 2.5 days, threshold: 1.5 days)
INFO - AutoRefresher - Successfully refreshed chicago in 45.32 seconds
WARNING - AutoRefresher - No collector available for unknown_dataset
ERROR - AutoRefresher - Error refreshing census: API timeout
```

## Best Practices

### 1. **Monitor Logs Regularly**
Check `pipeline.log` for refresh activities and errors.

### 2. **Set Appropriate Intervals**
- More frequent for time-sensitive data (Chicago permits)
- Less frequent for stable data (Census demographics)

### 3. **Consider API Limits**
- Don't set intervals too aggressively
- Monitor API usage to avoid rate limiting

### 4. **Disk Space Management**
- Regular cache cleanup with `--clear-cache`
- Monitor cache size with `--cache-info`

### 5. **Start Auto-Refresh on System Boot**
Add to crontab or systemd for production deployments:
```bash
@reboot cd /path/to/chicago-pipeline && python main.py --start-auto-refresh
```

## Troubleshooting

### System Won't Start
1. Check API keys are set correctly
2. Ensure `schedule` library is installed
3. Verify cache directory permissions

### Data Not Refreshing
1. Check if auto-refresh is running: `--refresh-status`
2. Look for error messages in logs
3. Try manual refresh: `--refresh-now`
4. Verify network connectivity

### High Memory/CPU Usage
1. Increase refresh intervals
2. Check for stuck refresh processes
3. Monitor cache size and clear if needed

## Advanced Usage

### Custom Scheduling
For production deployments, you can customize the scheduling:

```python
import schedule

# Clear default schedules
schedule.clear()

# Add custom schedules
schedule.every().hour.do(lambda: refresher.refresh_data_source('chicago'))
schedule.every(4).hours.do(lambda: refresher.refresh_data_source('fred'))
```

### Integration with Monitoring Systems
```python
# Export metrics for monitoring
def export_refresh_metrics():
    status = refresher.get_status()
    return {
        'refresh_running': status['is_running'],
        'stale_datasets': sum(status['data_staleness'].values()),
        'last_refresh_age': calculate_age(status['last_refresh_times'])
    }
```

This automatic refresh system ensures your Chicago Housing Pipeline always has the freshest data available while minimizing manual intervention and API calls. 