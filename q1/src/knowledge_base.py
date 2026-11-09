"""Local Knowledge-Base retrieval module using deterministic BM25 lexical ranking.

Designed for the SaafHawa agent to query official heatwave and air quality advisories
on the local machine without external network dependencies.
"""

from dataclasses import dataclass
import json
import math
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Document:
    doc_id: str
    title: str
    authority: str
    source_url: str
    filename: str
    content: str
    token_count: int = 0


@dataclass
class SearchHit:
    doc_id: str
    title: str
    score: float
    snippet: str


@dataclass
class RetrievalResult:
    query_id: str
    query_text: str
    hits: List[SearchHit]
    indexed_doc_count: int
    elapsed_ms: float


class BM25Retriever:
    """Deterministic BM25 document ranking implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, Document] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.doc_freq: Dict[str, int] = {}
        self.num_docs: int = 0
        self.indexing_elapsed_ms: float = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, deterministic multilingual tokenizer for English and Roman Urdu."""
        normalized = text.lower()
        tokens = re.findall(r"\b[a-z0-9_\u0600-\u06FF]{2,}\b", normalized)
        return tokens

    def load_corpus(self, kb_dir: str, metadata_path: str) -> float:
        """Load corpus documents, construct the inverted index, and record indexing duration."""
        start_ns = time.perf_counter_ns()

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Knowledge-base metadata not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        total_length = 0
        for item in metadata_list:
            doc_id = item["document_id"]
            doc_file = os.path.join(kb_dir, item["filename"])
            if not os.path.exists(doc_file):
                continue
            with open(doc_file, "r", encoding="utf-8") as df:
                content = df.read()

            tokens = self.tokenize(content)
            doc_len = len(tokens)
            doc = Document(
                doc_id=doc_id,
                title=item["title"],
                authority=item["authority"],
                source_url=item["source_url"],
                filename=item["filename"],
                content=content,
                token_count=doc_len,
            )
            self.documents[doc_id] = doc
            self.doc_lengths[doc_id] = doc_len
            total_length += doc_len

            # Build term frequency
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term, count in tf_map.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = count
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.num_docs = len(self.documents)
        self.avg_doc_len = (total_length / self.num_docs) if self.num_docs > 0 else 0.0

        end_ns = time.perf_counter_ns()
        self.indexing_elapsed_ms = (end_ns - start_ns) / 1_000_000.0
        return self.indexing_elapsed_ms

    def search(self, query: str, top_k: int = 3) -> Tuple[List[SearchHit], float]:
        """Execute BM25 search and return top hits along with elapsed query lookup time in ms."""
        start_ns = time.perf_counter_ns()
        query_tokens = self.tokenize(query)

        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.documents}

        for token in query_tokens:
            if token not in self.doc_freq:
                continue
            df = self.doc_freq[token]
            # Standard Lucene/Robertson IDF formulation
            idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

            for doc_id, tf in self.inverted_index[token].items():
                doc_len = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * ((tf * (self.k1 + 1.0)) / denom)
                scores[doc_id] += term_score

        # Sort documents by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [item for item in ranked if item[1] > 0.0][:top_k]

        hits: List[SearchHit] = []
        for doc_id, score in top_candidates:
            doc = self.documents[doc_id]
            # Extract first 150 characters as snippet
            snippet = doc.content[:150].replace("\n", " ").strip() + "..."
            hits.append(SearchHit(doc_id=doc_id, title=doc.title, score=round(score, 4), snippet=snippet))

        end_ns = time.perf_counter_ns()
        elapsed_ms = (end_ns - start_ns) / 1_000_000.0
        return hits, elapsed_ms


class KnowledgeBaseService:
    """Manages knowledge base queries and provides timed lookup."""

    def __init__(self, kb_dir: str, metadata_path: str):
        self.retriever = BM25Retriever()
        self.indexing_time_ms = self.retriever.load_corpus(kb_dir, metadata_path)

    def query(self, query_id: str, query_text: str, top_k: int = 3) -> RetrievalResult:
        hits, elapsed_ms = self.retriever.search(query_text, top_k=top_k)
        return RetrievalResult(
            query_id=query_id,
            query_text=query_text,
            hits=hits,
            indexed_doc_count=self.retriever.num_docs,
            elapsed_ms=elapsed_ms,
        )

