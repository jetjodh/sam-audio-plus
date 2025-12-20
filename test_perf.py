
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

try:
    import scripts.separate
except ImportError:
    sys.path.append(os.getcwd())
    import scripts.separate

from sam_audio import SAMAudio

print("--- Starting Load Performance Test ---")
t_start = time.perf_counter()
try:
    # Use 'small' model - cache should be hit
    model = SAMAudio.from_pretrained("facebook/sam-audio-small")
    print(f"Total Model Load Time: {time.perf_counter() - t_start:.2f}s")
except Exception as e:
    print(f"Model load failed: {e}")

print("--- Test Complete ---")
