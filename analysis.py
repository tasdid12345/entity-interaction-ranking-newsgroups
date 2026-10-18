"""Keyword TF-IDF cosine ranker (baseline) vs LightGBM lambdarank with query-term
interaction and SVD latent features (improved), on 20 Newsgroups multi-term queries."""
import json, sys, time
from itertools import combinations

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # headless charts
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import normalize

SEED = 42
SEEDS = [42, 43, 44]          # LightGBM is randomized -> run with 3 seeds
N_QUERIES_PER_GROUP = 5       # 20 groups x 5 = 100 queries
TOP_TERMS = 10                # distinctive-term pool per newsgroup
K_CAND = 200                  # candidates per query, identical for both models
TOPN, RECALL_K = 10, 100
METRIC = "nDCG@10"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    """Fetch 20 Newsgroups (headers/footers/quotes removed) with retry + backoff."""
    for attempt in range(5):
        try:
            d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                                   shuffle=True, random_state=SEED)
            return d.data, d.target, d.target_names
        except Exception as e:  # network / HTTP errors: back off and retry
            wait = 5 * 2 ** attempt
            log(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    sys.exit("could not download 20 Newsgroups")


def ndcg(rels, n_rel):
    """nDCG@10 for a binary ranked relevance list; ideal uses all relevant pool docs."""
    disc = 1 / np.log2(np.arange(2, TOPN + 2))
    top = rels[:TOPN]
    dcg = (top * disc[:len(top)]).sum()
    return dcg / disc[:min(TOPN, n_rel)].sum()


def avg_prec(rels, n_rel):
    prec = np.cumsum(rels) / np.arange(1, len(rels) + 1)
    return (prec * rels).sum() / n_rel


def evaluate(sets, scores):
    """Mean nDCG@10 (main), MAP and Recall@100 (secondary) over queries."""
    nd, ap, rc = [], [], []
    for s, sc in zip(sets, scores):
        r = s["y"][np.argsort(-sc, kind="stable")]
        nd.append(ndcg(r, s["n_rel"]))
        ap.append(avg_prec(r, s["n_rel"]))
        rc.append(r[:RECALL_K].sum() / s["n_rel"])
    return {"ndcg": float(np.mean(nd)), "map": float(np.mean(ap)), "recall": float(np.mean(rc))}


def build_sets(Qn, Zq, queries, D, Zd, y_pool):
    """Per query: top-K pool docs by TF-IDF cosine, feature matrix and binary labels."""
    S = (csr_matrix(Qn) @ D.T).toarray()          # query x pool cosine (rows L2-normalised)
    nnz = np.diff(D.indptr)                        # distinct terms per doc (length proxy)
    sets = []
    for i, (tids, g) in enumerate(queries):
        top = np.argsort(-S[i])[:K_CAND]
        idx = top[S[i, top] > 0]
        if len(idx) == 0:
            continue
        W = D[idx][:, tids].toarray()              # doc TF-IDF weights of query terms
        pres = W > 0
        pairs = list(combinations(range(len(tids)), 2))
        co = np.stack([pres[:, a] & pres[:, b] for a, b in pairs], 1)        # pair co-occurrence
        cw = np.stack([np.minimum(W[:, a], W[:, b]) for a, b in pairs], 1)  # weighted pair overlap
        lat = Zd[idx] @ Zq[i]                      # SVD latent cosine
        X = np.column_stack([S[i, idx], pres.sum(1), pres.mean(1), W.min(1), W.max(1),
                             W.sum(1), co.sum(1), co.mean(1), cw.sum(1), nnz[idx], lat])
        sets.append(dict(X=X, y=(y_pool[idx] == g).astype(int), base=S[i, idx],
                         n_rel=int((y_pool == g).sum())))
    return sets


def fit_ranker(sets, leaves, seed):
    X = np.vstack([s["X"] for s in sets])
    y = np.concatenate([s["y"] for s in sets])
    grp = [len(s["y"]) for s in sets]
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05,
                       num_leaves=leaves, min_child_samples=10, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, random_state=seed, verbose=-1)
    return m.fit(X, y, group=grp)


def main():
    docs, target, names = load_data()
    log(f"loaded {len(docs)} documents, {len(names)} newsgroups")
    # 70/30 document split, stratified by newsgroup, fixed seed
    tr, te = train_test_split(np.arange(len(docs)), test_size=0.3, stratify=target, random_state=SEED)
    vec = TfidfVectorizer(stop_words="english", min_df=3, max_features=30000, sublinear_tf=True)
    Xtr = vec.fit_transform([docs[i] for i in tr])      # vocabulary/IDF from TRAIN docs only
    Xte = vec.transform([docs[i] for i in te])
    ytr, yte = target[tr], target[te]
    log(f"vocab={Xtr.shape[1]}, train={len(tr)}, test={len(te)}")

    # Queries: 2-3 terms drawn from each group's top distinctive TRAIN terms
    overall = np.asarray(Xtr.mean(axis=0)).ravel()
    G = np.vstack([np.asarray(Xtr[ytr == g].mean(axis=0)).ravel() for g in range(len(names))])
    rng = np.random.default_rng(SEED)
    queries = []
    for g in range(len(names)):
        frac = (ytr == g).mean()
        out_mean = (overall - frac * G[g]) / (1 - frac)
        top = np.argsort(-(G[g] - out_mean))[:TOP_TERMS]
        for _ in range(N_QUERIES_PER_GROUP):
            m = rng.choice([2, 3])
            queries.append((sorted(int(t) for t in rng.choice(top, m, replace=False)), g))
    log(f"built {len(queries)} queries")

    # Query vectors (IDF-weighted, normalised) and SVD latent space fit on train docs
    Q = np.zeros((len(queries), Xtr.shape[1]))
    for i, (tids, _) in enumerate(queries):
        Q[i, tids] = vec.idf_[tids]
    Qn = normalize(Q)
    svd = TruncatedSVD(n_components=100, random_state=SEED).fit(Xtr)
    Zq = normalize(svd.transform(Qn))
    Ztr, Zte = normalize(svd.transform(Xtr)), normalize(svd.transform(Xte))
    train_sets = build_sets(Qn, Zq, queries, Xtr, Ztr, ytr)
    test_sets = build_sets(Qn, Zq, queries, Xte, Zte, yte)
    log(f"candidate sets: train={len(train_sets)}, test={len(test_sets)}")
    if len(test_sets) < 10:
        sys.exit("too few usable test queries; cannot produce meaningful metrics")

    # Baseline: keyword TF-IDF cosine ranking (deterministic, no training)
    base = evaluate(test_sets, [s["base"] for s in test_sets])
    log(f"baseline test {METRIC}={base['ndcg']:.4f}")

    # Tiny hyper-parameter search: 3-fold CV over TRAIN queries only
    best_leaves, best_cv = None, -1.0
    for leaves in [15, 31]:
        cv = []
        for a, b in KFold(3, shuffle=True, random_state=SEED).split(train_sets):
            model = fit_ranker([train_sets[i] for i in a], leaves, SEED)
            ev = [train_sets[i] for i in b]
            cv.append(evaluate(ev, [model.predict(s["X"]) for s in ev])["ndcg"])
        log(f"CV num_leaves={leaves}: {METRIC}={np.mean(cv):.4f}")
        if np.mean(cv) > best_cv:
            best_leaves, best_cv = leaves, float(np.mean(cv))

    # Improved: LightGBM lambdarank, refit on all train queries, one test evaluation per seed
    runs = []
    for seed in SEEDS:
        model = fit_ranker(train_sets, best_leaves, seed)
        runs.append(evaluate(test_sets, [model.predict(s["X"]) for s in test_sets]))
        log(f"seed {seed}: improved test {METRIC}={runs[-1]['ndcg']:.4f}")
    imp = {k: float(np.mean([r[k] for r in runs])) for k in ["ndcg", "map", "recall"]}
    imp_std = float(np.std([r["ndcg"] for r in runs]))

    results = {
        "question": "Can a learned LightGBM ranker that adds query-term interaction features beat a "
                    "keyword TF-IDF cosine ranker at ranking documents for multi-term queries on public text?",
        "method": "Queries of 2-3 distinctive terms per 20 Newsgroups group; relevance = same group; "
                  "70/30 stratified document split; both rankers re-rank the same top-200 TF-IDF candidates.",
        "dataset": {"name": "20 Newsgroups (sklearn fetch_20newsgroups, headers/footers/quotes removed)",
                    "source_url": "https://ndownloader.figshare.com/files/5975967",
                    "n_rows": int(len(docs)), "n_train": int(len(tr)), "n_test": int(len(te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": METRIC,
        "higher_is_better": True,
        "baseline": "Keyword ranker: TF-IDF cosine between query terms and each candidate document.",
        "improved": "LightGBM lambdarank over the same candidates, using the TF-IDF score plus term "
                    "co-occurrence, term-weight and SVD latent-similarity features.",
        "metrics": {
            "baseline_score": base["ndcg"],
            "improved_score": imp["ndcg"],
            "improvement": imp["ndcg"] - base["ndcg"],
            "baseline_score_std": 0.0,
            "improved_score_std": imp_std,
            "baseline_map": base["map"],
            "improved_map": imp["map"],
            "baseline_recall_at_100": base["recall"],
            "improved_recall_at_100": imp["recall"],
            "best_num_leaves": best_leaves,
            "cv_ndcg_best": best_cv,
            "n_test_queries": len(test_sets),
        },
        "notes": "Baseline is deterministic, so its std is 0. Candidates are the top-200 TF-IDF "
                 "documents per query, so Recall@100 is bounded by that pool. Queries are built from "
                 "train-split statistics and evaluated on held-out documents. Queries are shared "
                 "between train and test (different documents). Relevance is topical, which favours "
                 "lexical matching. sklearn's downloader does not expose a timeout, so retries wrap it. "
                 "Results depend on this one split and seed set.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log("wrote results.json")

    # Optional chart: baseline vs improved nDCG@10
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline (TF-IDF)", "improved (LightGBM)"], [base["ndcg"], imp["ndcg"]],
           yerr=[0.0, imp_std], color=["grey", "steelblue"], capsize=4)
    ax.set_ylabel(METRIC)
    ax.set_title("Test ranking quality")
    fig.tight_layout()
    fig.savefig("figures/ndcg_comparison.png", dpi=120)


if __name__ == "__main__":
    main()
