"""
Quick test script to verify A3C training works correctly.
Runs for 50k steps to check for errors and verify logging.
"""

import os
import sys

# Temporarily modify MAX_GLOBAL_STEPS for testing
sys.path.insert(0, os.path.dirname(__file__))

# Import and modify the training script's constants
from a3c_mario6 import *

# Override for quick test
MAX_GLOBAL_STEPS = 50_000
SAVE_PATH = "./a3c_mario_test.pth"

if __name__ == "__main__":
    print("=" * 60)
    print("Running quick training test (50k steps)...")
    print("=" * 60)
    main()

