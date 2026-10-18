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

## Reference

This project is motivated by the following paper:

> H Kroll, J Pirklbauer, JC Kalo, M Kunz (2024). *A discovery system for narrative query graphs: entity-interaction-aware document retrieval: H. Kroll et al.*. International Journal on. https://link.springer.com/article/10.1007/s00799-023-00356-3

Paper author: Prof. Wolf-Tilo Balke, Technische Universität Braunschweig (balke@ifis.cs.tu-bs.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
