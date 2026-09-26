import sys, os, time
from rapidfuzz import fuzz, distance
from joblib import Parallel, delayed

s1 = {"name_norm": "tata consultancy services limited", "name_core": "tata consultancy services", "name_suffix": "limited", "name_tokens": ["tata", "consultancy", "services", "limited"], "addr_norm": "nariman point mumbai maharashtra 400021", "addr_empty": False, "addr_numbers": {400021}, "addr_postal": "400021"}
s2 = {"name_norm": "tata consultancy service ltd", "name_core": "tata consultancy service", "name_suffix": "limited", "name_tokens": ["tata", "consultancy", "service", "limited"], "addr_norm": "nariman point mumbai 400021", "addr_empty": False, "addr_numbers": {400021}, "addr_postal": "400021"}

from src.features import extract_pair_features
pairs = [(s1, s2, {}) for _ in range(5000)]

t0 = time.time()
res1 = [extract_pair_features(a, b, m) for a, b, m in pairs]
t1 = time.time() - t0
rate1 = int(5000 / t1)
print(f"Single-thread 5k: {t1:.2f}s ({rate1} pairs/sec)")

t0 = time.time()
res2 = Parallel(n_jobs=8, prefer="threads")(delayed(extract_pair_features)(a, b, m) for a, b, m in pairs)
t2 = time.time() - t0
rate2 = int(5000 / t2)
print(f"Threads (8) 5k: {t2:.2f}s ({rate2} pairs/sec)")

t0 = time.time()
res3 = Parallel(n_jobs=8, backend="loky", batch_size=500)(delayed(extract_pair_features)(a, b, m) for a, b, m in pairs)
t3 = time.time() - t0
rate3 = int(5000 / t3)
print(f"Loky (8) 5k: {t3:.2f}s ({rate3} pairs/sec)")
