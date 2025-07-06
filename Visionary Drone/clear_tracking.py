#!/usr/bin/env python3
"""
Clear Tracking Data Script
This script clears all persistent tracking data to ensure a fresh start.
"""

import os
import shutil
import glob

def clear_tracking_data():
    """Clear all tracking data and temporary files"""
    print("🧹 Clearing all tracking data...")
    
    # Clear Python cache files
    cache_dirs = glob.glob("**/__pycache__", recursive=True)
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            print(f"✅ Cleared cache: {cache_dir}")
        except Exception as e:
            print(f"❌ Failed to clear cache {cache_dir}: {e}")
    
    # Clear any temporary files
    temp_files = glob.glob("**/*.tmp", recursive=True)
    temp_files.extend(glob.glob("**/*.log", recursive=True))
    
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"✅ Removed temp file: {temp_file}")
        except Exception as e:
            print(f"❌ Failed to remove {temp_file}: {e}")
    
    print("✅ All tracking data cleared!")
    print("💡 You can now run the main script for a fresh start.")

if __name__ == "__main__":
    clear_tracking_data() 