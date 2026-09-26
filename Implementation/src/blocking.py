"""Scalable, source-separated multi-channel candidate generation.

The blocker deliberately keeps retrieval independent from pair classification.  Each
candidate records the real channel ranks and scores that caused its inclusion, so the
same frozen candidate universe can be audited, trained, and scored.
"""
from __future__ import annotations

import math
import os
import time
import unicodedata
from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

from .config import (
    EXACT_ADDR_BUCKET_LIMIT,
    EXACT_NAME_BUCKET_LIMIT,
    MAX_CANDIDATES_PER_SOURCE,
    RARE_TOKEN_IDF_PERCENTILE,
    TFIDF_NAME_NGRAMS,
    TFIDF_NAME_THRESHOLD,
    TOTAL_CANDIDATE_BUDGET,
)


def build_inverted_index(records, key_func, max_bucket=None):
    """Build a deterministic inverted index without truncating input order."""
    index = defaultdict(list)
    for record in records:
        key = key_func(record)
        if key:
            index[key].append(record["entity_id"])
    return index


def build_rare_token_index(records, percentile=RARE_TOKEN_IDF_PERCENTILE):
    """Return an index of distinctive name tokens and their smoothed IDF."""
    document_frequency = defaultdict(int)
    n_documents = len(records)
    for record in records:
        for token in set(record.get("name_tokens", ())):
            if len(token) >= 4:
                document_frequency[token] += 1
    if not document_frequency:
        return {}, {}
    idf = {
        token: math.log((n_documents + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }
    threshold = float(np.percentile(list(idf.values()), percentile))
    index = defaultdict(list)
    for record in records:
        for token in set(record.get("name_tokens", ())):
            frequency = document_frequency.get(token, 0)
            if 2 <= frequency <= 100 and idf.get(token, 0.0) >= threshold:
                index[token].append(record["entity_id"])
    return index, idf


def _address_anchor_tokens(record: dict) -> tuple[str, ...]:
    """Return bounded lexical address anchors, excluding number-only terms."""
    tokens = (record.get("addr_norm") or "").split()
    anchors = tokens[:2] + tokens[-2:]
    return tuple(
        dict.fromkeys(token for token in anchors if len(token) >= 4 and not token.isdigit())
    )


def build_rare_address_anchor_index(records, max_document_frequency=200):
    """Build a compact, low-DF head/tail-address retrieval index."""
    document_frequency = defaultdict(int)
    for record in records:
        for token in _address_anchor_tokens(record):
            document_frequency[token] += 1
    allowed = {
        token for token, frequency in document_frequency.items()
        if 2 <= frequency <= max_document_frequency
    }
    index = defaultdict(list)
    for record in records:
        for token in _address_anchor_tokens(record):
            if token in allowed:
                index[token].append(record["entity_id"])
    return index


def _fold_latin(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _name_view(record: dict) -> str:
    """Combine distinct searchable name views while preserving token boundaries."""
    views = []
    for key in ("name_core", "name_norm", "name_folded", "name_translit"):
        value = (record.get(key) or "").strip()
        if value and value not in views:
            views.append(value)
    base = record.get("name_norm", "")
    folded = _fold_latin(base)
    if folded and folded not in views:
        views.append(folded)
    return " ".join(views)


def _address_view(record: dict) -> str:
    base = (record.get("addr_folded") or record.get("addr_norm") or "").strip()
    return _fold_latin(base)


def _combined_view(record: dict) -> str:
    """One bounded sparse view with name and head/tail address anchors."""
    core = (record.get("name_core") or record.get("name_norm") or "").strip()
    name = (record.get("name_norm") or "").strip()
    translit = (record.get("name_translit") or "").strip()
    addr = (record.get("addr_norm") or "").strip()
    views = [core]
    if name and name != core:
        views.append(name)
    if translit and translit != core:
        views.append(translit)
    if addr:
        address_tokens = addr.split()
        views.append(" ".join(address_tokens[:4]))
        if len(address_tokens) > 4:
            views.append(" ".join(address_tokens[-4:]))
    return " ".join(v for v in views if v)


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _secondary_agreement(query: dict, target: dict) -> float:
    name_score = _jaccard(query.get("name_tokens", ()), target.get("name_tokens", ()))
    query_address = (query.get("addr_folded") or query.get("addr_norm") or "").split()
    target_address = (target.get("addr_folded") or target.get("addr_norm") or "").split()
    address_score = _jaccard(query_address, target_address)
    q_numbers = set(query.get("addr_numbers", ()))
    t_numbers = set(target.get("addr_numbers", ()))
    number_score = _jaccard(q_numbers, t_numbers)
    postal_score = float(
        bool(query.get("addr_postal"))
        and query.get("addr_postal") == target.get("addr_postal")
    )
    return 1.5 * name_score + 2.0 * address_score + 2.5 * number_score + 2.0 * postal_score


class _HashedTfidfChannel:
    """Bounded-vocabulary sparse TF-IDF channel suitable for multi-million galleries."""

    def __init__(self, texts, ngram_range, n_features):
        self.enabled = any(bool(text) for text in texts)
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            n_features=n_features,
            alternate_sign=False,
            lowercase=False,
            norm=None,
            dtype=np.float32,
        )
        self.transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
        if self.enabled:
            counts = self.vectorizer.transform(texts)
            self.transformer.fit(counts)
            self.matrix = self.transformer.transform(counts, copy=False).tocsr()
        else:
            self.matrix = csr_matrix((len(texts), n_features), dtype=np.float32)

    def transform(self, texts):
        if not self.enabled:
            return csr_matrix((len(texts), self.matrix.shape[1]), dtype=np.float32)
        return self.transformer.transform(self.vectorizer.transform(texts)).tocsr()


def _base_meta(source: str) -> dict:
    return {
        "source": source,
        "channels": set(),
        "channel_ranks": {},
        "exact_name": 0,
        "exact_addr": 0,
        "rare_token": 0,
        "address_anchor": 0,
        "tfidf_name_score": 0.0,
        "tfidf_addr_score": 0.0,
        "best_rank": 999_999,
        "number_overlap": 0.0,
    }


def _add_channel(meta: dict, channel: str, rank: int, score_field=None, score=0.0):
    meta["channels"].add(channel)
    previous = meta["channel_ranks"].get(channel)
    meta["channel_ranks"][channel] = rank if previous is None else min(previous, rank)
    meta["best_rank"] = min(meta["best_rank"], rank)
    if score_field:
        meta[score_field] = max(float(meta.get(score_field, 0.0)), float(score))


def _candidate_score(meta: dict) -> float:
    return (
        4.0 * meta["exact_name"]
        + 3.0 * meta["exact_addr"]
        + 1.5 * meta["rare_token"]
        + 1.8 * meta.get("address_anchor", 0)
        + 3.0 * meta["tfidf_name_score"]
        + 2.0 * meta["tfidf_addr_score"]
        + 1.5 * meta.get("number_overlap", 0.0)
        + 0.15 * len(meta["channels"])
        + 0.05 / max(meta["best_rank"], 1)
    )


class TargetSearchIndex:
    """Searchable index for one target source inside one country partition."""

    def __init__(self, records, source_label="S2", k_per_source=MAX_CANDIDATES_PER_SOURCE):
        self.source_label = source_label
        self.k_per_source = k_per_source
        records_list = list(records)
        # Target indexes remain resident for millions of rows.  These fields are
        # not used by blocking or pair features, so releasing their references
        # before building postings materially lowers peak RSS without changing
        # retrieval semantics.
        unused_target_fields = (
            "name_raw", "addr_raw", "country", "name_compact",
            "name_numeric_tokens", "addr_compact", "addr_numeric_tokens",
            "addr_tokens", "is_indic",
        )
        for record in records_list:
            for field in unused_target_fields:
                record.pop(field, None)
        self.records = None  # Free redundant list to keep RAM minimal
        self.rec_map = {record["entity_id"]: record for record in records_list}
        self.target_ids = [record["entity_id"] for record in records_list]

        self.idx_name = build_inverted_index(records_list, lambda r: r.get("name_norm", ""))
        self.idx_folded = build_inverted_index(records_list, lambda r: _fold_latin(r.get("name_norm", "")))
        self.idx_core = build_inverted_index(records_list, lambda r: r.get("name_core", ""))
        self.idx_addr = build_inverted_index(
            records_list,
            lambda r: "" if r.get("addr_empty", True) else r.get("addr_norm", ""),
        )
        self.idx_rare, self.rare_idf = build_rare_token_index(records_list)
        self.idx_addr_anchor = build_rare_address_anchor_index(records_list)
        self.idx_tok_num = defaultdict(list)
        self.idx_tok_post = defaultdict(list)
        self.idx_name_num = defaultdict(list)
        self.idx_name_post = defaultdict(list)
        for record in records_list:
            target_id = record["entity_id"]
            tokens = list(record.get("name_tokens", ()))
            exact_name = record.get("name_norm", "")
            for number in record.get("addr_numbers", ()):
                if exact_name:
                    self.idx_name_num[(exact_name, number)].append(target_id)
            if exact_name and record.get("addr_postal"):
                self.idx_name_post[(exact_name, record["addr_postal"])].append(target_id)
            for token in tokens[:2]:
                for number in record.get("addr_numbers", ()):
                    self.idx_tok_num[(token, number)].append(target_id)
                if record.get("addr_postal"):
                    self.idx_tok_post[(token, record["addr_postal"])].append(target_id)

        combined_texts = [_combined_view(record) for record in records_list]
        del records_list
        self.combined_channel = _HashedTfidfChannel(combined_texts, TFIDF_NAME_NGRAMS, 2**18)
        del combined_texts
        self.sparse_channel_count = 1
        self.overflow_counts = defaultdict(int)

    def _sparse_retrieve(self, query_matrix, target_matrix, top_n, threshold):
        if not query_matrix.nnz or not target_matrix.nnz or not self.target_ids:
            return {}
        top_n = min(max(1, top_n), len(self.target_ids))
        threads = min(20, os.cpu_count() or 4)
        similarities = sp_matmul_topn(
            query_matrix,
            target_matrix.T,
            top_n=top_n,
            threshold=threshold,
            sort=True,
            n_threads=threads,
        ).tocsr()
        rows = {}
        for row in range(similarities.shape[0]):
            start, end = similarities.indptr[row], similarities.indptr[row + 1]
            pairs = list(zip(similarities.indices[start:end], similarities.data[start:end]))
            pairs.sort(key=lambda pair: (-float(pair[1]), self.target_ids[pair[0]]))
            rows[row] = pairs
        return rows

    def _rank_bucket(self, query, target_ids, limit):
        unique_ids = dict.fromkeys(target_ids)
        ranked = sorted(
            unique_ids,
            key=lambda target_id: (
                -_secondary_agreement(query, self.rec_map[target_id]),
                target_id,
            ),
        )
        return ranked[:limit]

    def retrieve_for_s1_chunk(self, s1_chunk, k=None):
        k = int(k or self.k_per_source)
        candidate_rows = defaultdict(dict)
        query_ids = [record["entity_id"] for record in s1_chunk]

        retrieval_width = max(k * 3, 12)
        combined_rows = self._sparse_retrieve(
            self.combined_channel.transform([_combined_view(record) for record in s1_chunk]),
            self.combined_channel.matrix,
            retrieval_width,
            TFIDF_NAME_THRESHOLD,
        )

        for row_index, pairs in combined_rows.items():
            query_id = query_ids[row_index]
            for rank, (column, score) in enumerate(pairs, start=1):
                target_id = self.target_ids[column]
                meta = candidate_rows[query_id].setdefault(target_id, _base_meta(self.source_label))
                _add_channel(meta, "char_name", rank, "tfidf_name_score", score)

        bucket_width = max(k * 4, 20)
        for query in s1_chunk:
            query_id = query["entity_id"]
            query_candidates = candidate_rows[query_id]

            exact_sources = (
                ("exact_name", self.idx_name.get(query.get("name_norm", ""), ())),
                ("folded_name", self.idx_folded.get(_fold_latin(query.get("name_norm", "")), ())),
                ("core_name", self.idx_core.get(query.get("name_core", ""), ())),
            )
            for channel, bucket in exact_sources:
                if not bucket:
                    continue
                limit = min(EXACT_NAME_BUCKET_LIMIT, bucket_width)
                if channel == "exact_name" and len(bucket) > limit:
                    narrowed = []
                    for number in query.get("addr_numbers", ()):
                        narrowed.extend(self.idx_name_num.get((query.get("name_norm", ""), number), ()))
                    if query.get("addr_postal"):
                        narrowed.extend(
                            self.idx_name_post.get(
                                (query.get("name_norm", ""), query["addr_postal"]), ()
                            )
                        )
                    if narrowed:
                        bucket = narrowed
                    elif len(bucket) > EXACT_NAME_BUCKET_LIMIT:
                        self.overflow_counts["ambiguous_exact_name"] += 1
                        bucket = ()
                for rank, target_id in enumerate(self._rank_bucket(query, bucket, limit), start=1):
                    meta = query_candidates.setdefault(target_id, _base_meta(self.source_label))
                    _add_channel(meta, channel, rank)
                    if channel == "exact_name":
                        meta["exact_name"] = 1

            if not query.get("addr_empty", True):
                address_bucket = self.idx_addr.get(query.get("addr_norm", ""), ())
                limit = min(EXACT_ADDR_BUCKET_LIMIT, bucket_width)
                for rank, target_id in enumerate(self._rank_bucket(query, address_bucket, limit), start=1):
                    meta = query_candidates.setdefault(target_id, _base_meta(self.source_label))
                    _add_channel(meta, "exact_addr", rank)
                    meta["exact_addr"] = 1

            rare_targets = []
            for token in set(query.get("name_tokens", ())):
                rare_targets.extend(self.idx_rare.get(token, ()))
            for rank, target_id in enumerate(self._rank_bucket(query, rare_targets, bucket_width), start=1):
                meta = query_candidates.setdefault(target_id, _base_meta(self.source_label))
                _add_channel(meta, "rare_token", rank)
                meta["rare_token"] = 1

            address_anchor_targets = []
            for token in _address_anchor_tokens(query):
                address_anchor_targets.extend(self.idx_addr_anchor.get(token, ()))
            for rank, target_id in enumerate(
                self._rank_bucket(query, address_anchor_targets, bucket_width), start=1
            ):
                meta = query_candidates.setdefault(target_id, _base_meta(self.source_label))
                _add_channel(meta, "address_anchor", rank)
                meta["address_anchor"] = 1

            compound_targets = []
            for token in list(query.get("name_tokens", ()))[:2]:
                for number in query.get("addr_numbers", ()):
                    compound_targets.extend(self.idx_tok_num.get((token, number), ()))
                if query.get("addr_postal"):
                    compound_targets.extend(self.idx_tok_post.get((token, query["addr_postal"]), ()))
            for rank, target_id in enumerate(self._rank_bucket(query, compound_targets, bucket_width), start=1):
                meta = query_candidates.setdefault(target_id, _base_meta(self.source_label))
                _add_channel(meta, "compound", rank)

            for target_id, meta in query_candidates.items():
                q_numbers = set(query.get("addr_numbers", ()))
                t_numbers = set(self.rec_map[target_id].get("addr_numbers", ()))
                meta["number_overlap"] = _jaccard(q_numbers, t_numbers)
                query_address_tokens = _address_view(query).split()
                target_address_tokens = _address_view(self.rec_map[target_id]).split()
                meta["tfidf_addr_score"] = _jaccard(
                    query_address_tokens, target_address_tokens
                )

            ranked = sorted(
                query_candidates.items(),
                key=lambda item: (-_candidate_score(item[1]), item[0]),
            )
            candidate_rows[query_id] = dict(ranked[:k])

        return {query_id: dict(candidate_rows.get(query_id, {})) for query_id in query_ids}

    def close(self):
        """Release gallery-sized structures before another source index is built."""
        for name in (
            "combined_channel",
            "rec_map",
            "records",
            "target_ids",
            "idx_name",
            "idx_folded",
            "idx_core",
            "idx_addr",
            "idx_rare",
            "idx_addr_anchor",
            "idx_tok_num",
            "idx_tok_post",
            "idx_name_num",
            "idx_name_post",
        ):
            if hasattr(self, name):
                setattr(self, name, None)


class SourceSeparatedBlocker:
    """Run independent S1→S2 and S1→S3 retrieval with balanced final quotas."""

    def __init__(self, k_per_source=MAX_CANDIDATES_PER_SOURCE, total_k=TOTAL_CANDIDATE_BUDGET):
        self.k_per_source = int(k_per_source)
        self.total_k = int(total_k)

    @staticmethod
    def _rank(items):
        return sorted(items, key=lambda item: (-_candidate_score(item[1]), item[0]))

    def _balanced_merge(self, source2: dict, source3: dict):
        ranked2, ranked3 = self._rank(source2.items()), self._rank(source3.items())
        quota2 = (self.total_k + 1) // 2
        quota3 = self.total_k // 2
        selected = ranked2[:quota2] + ranked3[:quota3]
        remaining = ranked2[quota2:] + ranked3[quota3:]
        if len(selected) < self.total_k:
            selected.extend(self._rank(remaining)[: self.total_k - len(selected)])
        return dict(self._rank(selected)[: self.total_k])

    def generate_candidates_for_chunk(self, s1_chunk, s2_index, s3_index):
        candidates2 = s2_index.retrieve_for_s1_chunk(s1_chunk, k=self.k_per_source)
        candidates3 = s3_index.retrieve_for_s1_chunk(s1_chunk, k=self.k_per_source)
        structured, simple = {}, {}
        for record in s1_chunk:
            query_id = record["entity_id"]
            merged = self._balanced_merge(
                candidates2.get(query_id, {}),
                candidates3.get(query_id, {}),
            )
            structured[query_id] = merged
            simple[query_id] = set(merged)
        return structured, simple

    def generate_candidates(self, s1_records, s2_records, s3_records):
        import gc

        def retrieve_source(records, source_label):
            index = TargetSearchIndex(records, source_label, self.k_per_source)
            result = {}
            n_chunks = (len(s1_records) + 9999) // 10000
            try:
                for chunk_idx, start in enumerate(range(0, len(s1_records), 10_000), 1):
                    chunk = s1_records[start : start + 10_000]
                    t_c = time.time()
                    res = index.retrieve_for_s1_chunk(chunk, self.k_per_source)
                    result.update(res)
                    print(f"      [{source_label}] Chunk {chunk_idx}/{n_chunks} ({len(chunk):,} queries) in {time.time()-t_c:.1f}s", flush=True)
            finally:
                index.close()
                del index
                gc.collect()
            return result

        candidates2 = retrieve_source(s2_records, "S2")
        candidates3 = retrieve_source(s3_records, "S3")
        structured, simple = {}, {}
        for query in s1_records:
            query_id = query["entity_id"]
            merged = self._balanced_merge(
                candidates2.get(query_id, {}), candidates3.get(query_id, {})
            )
            structured[query_id] = merged
            simple[query_id] = set(merged)
        return structured, simple
