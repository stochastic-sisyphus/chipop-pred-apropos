#!/usr/bin/env python3
"""
API Key Setup Helper for Chicago Housing Pipeline

This script helps users set up API keys required for production data collection.
"""

import os
import sys
from pathlib import Path

def main():
    """Main setup function."""
    print("=" * 60)
    print("Chicago Housing Pipeline - API Key Setup")
    print("=" * 60)
    print()
    
    print("This script will help you set up API keys for production data collection.")
    print()
    
    # Check current status
    print("Current API Key Status:")
    print("-" * 30)
    
    api_keys = {
        'CENSUS_API_KEY': {
            'description': 'Census Bureau API (demographic data)',
            'signup_url': 'https://api.census.gov/data/key_signup.html',
            'required': True
        },
        'FRED_API_KEY': {
            'description': 'Federal Reserve Economic Data API',
            'signup_url': 'https://fred.stlouisfed.org/docs/api/api_key.html',
            'required': True
        },
        'CHICAGO_DATA_TOKEN': {
            'description': 'Chicago Data Portal API Token',
            'signup_url': 'https://data.cityofchicago.org/profile/app_tokens',
            'required': True
        },
        'BEA_API_KEY': {
            'description': 'Bureau of Economic Analysis API',
            'signup_url': 'https://apps.bea.gov/API/signup/',
            'required': False
        }
    }
    
    missing_keys = []
    
    for key_name, info in api_keys.items():
        current_value = os.environ.get(key_name, '')
        status = "✓ SET" if current_value and not current_value.startswith('your_') else "✗ NOT SET"
        required_text = "(Required)" if info['required'] else "(Optional)"
        
        print(f"{key_name:<20}: {status} {required_text}")
        print(f"  {info['description']}")
        
        if status == "✗ NOT SET" and info['required']:
            missing_keys.append((key_name, info))
        print()
    
    if not missing_keys:
        print("✓ All required API keys are set!")
        print()
        print("You can now run the pipeline with production data:")
        print("  python main.py")
        return
    
    print("=" * 60)
    print("Required API Keys Missing")
    print("=" * 60)
    print()
    
    for key_name, info in missing_keys:
        print(f"Missing: {key_name}")
        print(f"Description: {info['description']}")
        print(f"Sign up at: {info['signup_url']}")
        print()
    
    print("=" * 60)
    print("Setup Instructions")
    print("=" * 60)
    print()
    
    print("1. Sign up for API keys at the URLs above")
    print()
    print("2. Set environment variables:")
    print()
    
    # Generate shell commands
    for key_name, info in missing_keys:
        print(f"   export {key_name}='your_actual_api_key_here'")
    
    print()
    print("3. Add to your shell profile (optional, for persistence):")
    print()
    
    shell_profile = "~/.bashrc"  # Default to bash
    if 'zsh' in os.environ.get('SHELL', ''):
        shell_profile = "~/.zshrc"
    
    print(f"   echo 'export CENSUS_API_KEY=\"your_actual_census_key\"' >> {shell_profile}")
    print(f"   echo 'export FRED_API_KEY=\"your_actual_fred_key\"' >> {shell_profile}")
    print(f"   echo 'export CHICAGO_DATA_TOKEN=\"your_actual_chicago_token\"' >> {shell_profile}")
    print(f"   source {shell_profile}")
    
    print()
    print("4. Test your setup:")
    print("   python main.py --check-api-keys")
    print()
    print("5. Run the pipeline:")
    print("   python main.py                    # Uses production data")
    print("   python main.py --use-sample-data  # Uses sample data")
    print()
    
    print("=" * 60)
    print("Alternative: Use Sample Data")
    print("=" * 60)
    print()
    print("If you don't want to set up API keys right now, you can run")
    print("the pipeline with sample data:")
    print()
    print("   python main.py --use-sample-data")
    print()
    print("This will use the provided sample data files for testing.")

if __name__ == "__main__":
    main() 