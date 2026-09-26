import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.config import OUTPUT_DIR, TEST_DIR, CHECKPOINT_DIR
print(f"OUTPUT_DIR={OUTPUT_DIR}")
print(f"TEST_DIR={TEST_DIR}")
print(f"CHECKPOINT_DIR={CHECKPOINT_DIR}")
print(f"TEST_DIR exists={os.path.isdir(TEST_DIR)}")
print(f"test_source1 exists={os.path.exists(os.path.join(TEST_DIR, 'test_source1.tsv'))}")
