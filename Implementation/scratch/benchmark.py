import time, numpy as np, scipy.sparse as sp, sparse_dot_topn

N = 200000; M = 262144; Q = 1000
print(f"Creating test matrices N={N}, M={M}, Q={Q}...")
target = sp.random(N, M, density=0.0003, format="csr", dtype=np.float32)
query = sp.random(Q, M, density=0.0003, format="csr", dtype=np.float32)

t0 = time.time()
res = sparse_dot_topn.sp_matmul_topn(query, target.T, top_n=36, threshold=0.12, sort=True, n_threads=4)
print(f"Time for {Q} queries against {N} targets (4 threads): {time.time()-t0:.2f}s")

t0 = time.time()
target_csr = target.T.tocsr()
print(f"Pre-transposed to CSR: {time.time()-t0:.2f}s")

t0 = time.time()
res2 = sparse_dot_topn.sp_matmul_topn(query, target_csr, top_n=36, threshold=0.12, sort=True, n_threads=20)
print(f"Time for {Q} queries against {N} targets (20 threads, pre-transposed): {time.time()-t0:.2f}s")
