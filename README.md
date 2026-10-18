# Learned Ranker with Query-Term Interactions vs. Keyword TF-IDF on 20 Newsgroups

Keyword search ranks documents by how well their terms match the query, but multi-term queries often depend on how query terms interact within a document, not only on their individual weights. The paper "A discovery system for narrative query graphs: entity-interaction-aware document retrieval" (Kroll et al., 2024) argues that modelling entity interactions improves document retrieval over keyword-initial ranking. This experiment is a small, public-data stand-in for that idea.

**Question:** Can a learned LightGBM ranker that adds query-term interaction features beat a keyword TF-IDF cosine ranker at ranking documents for multi-term queries on public text?

**Dataset:** 20 Newsgroups, loaded with scikit-learn's `fetch_20newsgroups` (headers, footers and quotes removed). Queries are 2-3 distinctive terms from one newsgroup, and a document is relevant if it belongs to that newsgroup. This is not the paper's own dataset (`same_as_paper: false`).

**Baseline:** a keyword ranker that scores each candidate document by TF-IDF cosine similarity to the query terms.

**Improved model:** a LightGBM lambdarank ranker over the same candidate documents. It uses the TF-IDF score plus pairwise term co-occurrence, term-weight and SVD latent-similarity features, which approximate entity interactions.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json`. Charts are saved to `figures/`.

## Results (latest run)

On 20 Newsgroups multi-term queries (100 test queries), nDCG@10 was 0.735 for a standard TF-IDF keyword ranker and 0.799 for my LightGBM lambdarank with query-term interaction features on the same held-out candidates, a gain of 0.063 from one split.

| Metric | Value |
|---|---|
| baseline_score | 0.7355 |
| improved_score | 0.7987 |
| improvement | 0.06322 |
| baseline_score_std | 0 |
| improved_score_std | 0.003271 |
| baseline_map | 0.1605 |
| improved_map | 0.1786 |
| baseline_recall_at_100 | 0.1763 |
| improved_recall_at_100 | 0.1907 |
| best_num_leaves | 31 |
| cv_ndcg_best | 0.849 |
| n_test_queries | 100 |

Run on 2026-10-18. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Baseline is deterministic, so its std is 0. Candidates are the top-200 TF-IDF documents per query, so Recall@100 is bounded by that pool. Queries are built from train-split statistics and evaluated on held-out documents. Queries are shared between train and test (different documents). Relevance is topical, which favours lexical matching. sklearn's downloader does not expose a timeout, so retries wrap it. Results depend on this one split and seed set.

## Reference

This project is motivated by the following paper:

> H Kroll, J Pirklbauer, JC Kalo, M Kunz (2024). *A discovery system for narrative query graphs: entity-interaction-aware document retrieval: H. Kroll et al.*. International Journal on. https://link.springer.com/article/10.1007/s00799-023-00356-3

Paper author: Prof. Wolf-Tilo Balke, Technische Universität Braunschweig (balke@ifis.cs.tu-bs.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
