import sys
print(f"Python: {sys.version}")
for pkg in ['sklearn', 'pandas', 'numpy', 'lightgbm', 'scipy', 'rapidfuzz', 'polars', 'sparse_dot_topn', 'anyascii', 'unidecode']:
    try:
        __import__(pkg)
        print(f"  {pkg}: OK")
    except ImportError:
        print(f"  {pkg}: MISSING")
