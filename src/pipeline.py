"""End-to-end пайплайн эксперимента с кэшированием стадий.

Файл отвечает за связывание всех стадий (загрузка -> предобработка ->
взвешивание -> разреживание -> граф -> сообщества -> сводки -> визуализация)
и за кэширование промежуточных артефактов. Кэш существует ради быстрой
обратной связи: тяжёлые стадии (лемматизация, матрица пересечений) считаются
один раз, а при изменении только параметров детекции (например, ``resolution``)
переиспользуются. Ключ кэша зависит от параметров, влияющих на стадию, поэтому
смена ручки автоматически инвалидирует нужный артефакт.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict

import igraph as ig
import numpy as np
from scipy import sparse

from src.community.detect import detect_communities, modularity
from src.community.summarize import summarize_communities
from src.config import ExperimentConfig
from src.data.load_corpus import Document, load_corpus
from src.graph.build_graph import build_igraph, sparsify
from src.graph.cooccurrence import build_doc_term_matrix, compute_weights
from src.graph.preprocess import preprocess_document
from src.viz.plot_graph import visualize_graph


def _hash(obj: object) -> str:
    """Короткий стабильный хеш от произвольного сериализуемого объекта."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def _cache_path(cache_dir: str, name: str, key: str, ext: str) -> str:
    """Формирует путь к файлу кэша."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}_{key}.{ext}")


def _load_documents(cfg: ExperimentConfig) -> list[Document]:
    """Загружает документы (кэш не нужен: чтение parquet быстрое)."""
    return load_corpus(cfg.data)


def _preprocess(
    documents: list[Document], cfg: ExperimentConfig
) -> list[set[str]]:
    """Предобрабатывает документы в n-граммы с кэшированием по параметрам."""
    key = _hash(
        {
            "preprocess": asdict(cfg.preprocess),
            "doc_ids": [d.doc_id for d in documents],
        }
    )
    path = _cache_path(cfg.cache_dir, "ngrams", key, "pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    doc_ngrams = [preprocess_document(d.text, cfg.preprocess) for d in documents]
    with open(path, "wb") as f:
        pickle.dump(doc_ngrams, f)
    return doc_ngrams


def _weights(
    doc_ngrams: list[set[str]], cfg: ExperimentConfig
) -> sparse.csr_matrix:
    """Считает матрицу весов рёбер с кэшированием по параметрам."""
    key = _hash(
        {
            "preprocess": asdict(cfg.preprocess),
            "weighting": asdict(cfg.weighting),
            "n_docs": len(doc_ngrams),
        }
    )
    path = _cache_path(cfg.cache_dir, "weights", key, "npz")
    if os.path.exists(path):
        return sparse.load_npz(path)

    matrix, _ = build_doc_term_matrix(
        doc_ngrams,
        max_vocab=cfg.preprocess.max_vocab,
        min_doc_freq=cfg.preprocess.min_doc_freq,
        max_doc_freq=cfg.preprocess.max_doc_freq,
    )
    weights = compute_weights(matrix, cfg.weighting)
    sparse.save_npz(path, weights)
    return weights


def _graph(
    weights: sparse.csr_matrix, documents: list[Document], cfg: ExperimentConfig
) -> ig.Graph:
    """Строит граф igraph с кэшированием по параметрам разреживания."""
    key = _hash(
        {
            "sparsify": asdict(cfg.sparsify),
            "weighting": asdict(cfg.weighting),
            "n_docs": len(documents),
        }
    )
    path = _cache_path(cfg.cache_dir, "graph", key, "graphml")
    if os.path.exists(path):
        return ig.Graph.Read_GraphML(path)

    sparse_weights = sparsify(weights, cfg.sparsify)
    graph = build_igraph(
        sparse_weights,
        doc_ids=[d.doc_id for d in documents],
        titles=[d.title for d in documents],
    )
    graph.write_graphml(path)
    return graph


def run(cfg: ExperimentConfig) -> dict:
    """Прогоняет полный пайплайн и возвращает сводку результатов."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    documents = _load_documents(cfg)
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    doc_ngrams = _preprocess(documents, cfg)
    timings["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    weights = _weights(doc_ngrams, cfg)
    timings["weights"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    graph = _graph(weights, documents, cfg)
    timings["graph"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    membership = detect_communities(graph, cfg.community)
    timings["detect"] = time.perf_counter() - t0
    mod = modularity(graph, membership)

    summaries = summarize_communities(membership, doc_ngrams, documents)

    t0 = time.perf_counter()
    html_path = visualize_graph(graph, membership, cfg.viz)
    timings["viz"] = time.perf_counter() - t0

    result = {
        "config": asdict(cfg),
        "n_documents": len(documents),
        "n_nodes": graph.vcount(),
        "n_edges": graph.ecount(),
        "n_communities": len({m for m in membership}),
        "modularity": mod,
        "timings": timings,
        "html_path": html_path,
        "communities": [
            {
                "community_id": s.community_id,
                "size": s.size,
                "top_ngrams": s.top_ngrams,
                "sample_titles": s.sample_titles,
            }
            for s in summaries
        ],
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    report_path = os.path.join(cfg.output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result["report_path"] = report_path
    return result
