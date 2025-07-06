#!/usr/bin/env python3
"""
Test Keypoints Script
This script tests the keypoint handling to ensure it works correctly.
"""

import numpy as np

def test_keypoint_validation():
    """Test the keypoint validation logic"""
    print("🧪 Testing keypoint validation...")
    
    # Test case 1: Valid keypoints
    valid_keypoints = np.array([[100, 200], [150, 250], [200, 300], [250, 350], [300, 400], 
                               [350, 450], [400, 500], [450, 550], [500, 600], [550, 650], 
                               [600, 700], [650, 750], [700, 800], [750, 850], [800, 900], 
                               [850, 950], [900, 1000]])
    
    try:
        result = np.any(valid_keypoints != 0)
        print(f"✅ Valid keypoints test: {result}")
    except Exception as e:
        print(f"❌ Valid keypoints test failed: {e}")
    
    # Test case 2: All zeros
    zero_keypoints = np.zeros((17, 2))
    
    try:
        result = np.any(zero_keypoints != 0)
        print(f"✅ Zero keypoints test: {result}")
    except Exception as e:
        print(f"❌ Zero keypoints test failed: {e}")
    
    # Test case 3: None keypoints
    try:
        result = None is not None and len(None) >= 11 and np.any(None != 0)
        print(f"✅ None keypoints test: {result}")
    except Exception as e:
        print(f"❌ None keypoints test failed: {e}")
    
    # Test case 4: NaN keypoints
    nan_keypoints = np.full((17, 2), np.nan)
    
    try:
        result = nan_keypoints is not None and len(nan_keypoints) >= 11 and np.any(nan_keypoints != 0)
        print(f"✅ NaN keypoints test: {result}")
    except Exception as e:
        print(f"❌ NaN keypoints test failed: {e}")
    
    print("✅ Keypoint validation tests completed!")

if __name__ == "__main__":
    test_keypoint_validation() 