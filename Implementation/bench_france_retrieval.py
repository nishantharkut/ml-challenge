"""Measure local France retrieval capacity without invoking a matcher.

This is an operational probe for assigning the France partition to a lower-RAM
machine.  It does not write candidates or submission files.
"""
import argparse
import importlib
import os
import time

import psutil

from src.blocking import TargetSearchIndex


def main(query_count: int) -> None:
    runner = importlib.import_module("50_run_country")
    process = psutil.Process(os.getpid())
    s1 = runner.load_test_records_by_country("test_source1.tsv", "FRANCE")
    s2 = runner.load_test_records_by_country("test_source2.tsv", "FRANCE")
    print(f"Loaded France S1={len(s1):,}, S2={len(s2):,}; RSS={process.memory_info().rss / 2**30:.2f} GiB", flush=True)
    index = TargetSearchIndex(s2, "S2")
    print(f"Built S2 index; RSS={process.memory_info().rss / 2**30:.2f} GiB", flush=True)
    start = time.perf_counter()
    results = index.retrieve_for_s1_chunk(s1[:query_count])
    elapsed = time.perf_counter() - start
    pair_count = sum(len(row) for row in results.values())
    print(
        f"Retrieved {len(results):,} queries / {pair_count:,} pairs in {elapsed:.1f}s; "
        f"RSS={process.memory_info().rss / 2**30:.2f} GiB",
        flush=True,
    )
    index.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=5_000)
    args = parser.parse_args()
    main(args.queries)
