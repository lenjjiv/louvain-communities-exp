"""Разреживание матрицы весов и построение графа igraph.

Файл отвечает за превращение плотной матрицы весов рёбер в разреженный граф,
пригодный для детекции сообществ. Разреживание критично: без него граф
документов почти полный, и алгоритмы сообществ на нём работают плохо и
медленно. Реализованы три способа:

* ``topk`` — оставить k сильнейших соседей на узел (стандарт, ограничивает
  степень узла);
* ``threshold`` — оставить рёбра выше порога по весу;
* ``both`` — сначала порог, затем top-k (максимум контроля).
"""

from __future__ import annotations

import logging

import igraph as ig
import numpy as np
from scipy import sparse
from tqdm import tqdm

from src.config import SparsifyConfig

logger = logging.getLogger(__name__)


def _topk_sparsify(weights: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Оставляет k сильнейших соседей для каждого узла.

    Ребро сохраняется, если оно входит в top-k хотя бы у одного из концов —
    так граф остаётся симметричным и не теряет связи, важные для одного узла.
    """
    weights = weights.tocsr()
    n = weights.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in tqdm(range(n), desc="top-k sparsify", unit="node"):
        start, end = weights.indptr[i], weights.indptr[i + 1]
        if start == end:
            continue
        row_data = weights.data[start:end]
        row_cols = weights.indices[start:end]
        if len(row_data) > k:
            top = np.argpartition(row_data, -k)[-k:]
            row_data = row_data[top]
            row_cols = row_cols[top]
        rows.extend([i] * len(row_cols))
        cols.extend(row_cols.tolist())
        data.extend(row_data.tolist())

    result = sparse.csr_matrix((data, (rows, cols)), shape=weights.shape)
    # Симметризация: ребро остаётся, если попало в top-k хотя бы у одного конца.
    result = result.maximum(result.T).tocsr()
    result.eliminate_zeros()
    return result


def sparsify(weights: sparse.csr_matrix, cfg: SparsifyConfig) -> sparse.csr_matrix:
    """Применяет выбранный способ разреживания к матрице весов."""
    result = weights.tocsr()
    if cfg.method in {"threshold", "both"}:
        result = result.copy()
        result.data[result.data < cfg.min_weight] = 0.0
        result.eliminate_zeros()
    if cfg.method in {"topk", "both"}:
        result = _topk_sparsify(result, cfg.top_k)
    if cfg.method not in {"topk", "threshold", "both"}:
        raise ValueError(f"Неизвестный способ разреживания: '{cfg.method}'")
    return result


def build_igraph(
    weights: sparse.csr_matrix,
    doc_ids: list[str],
    titles: list[str],
) -> ig.Graph:
    """Строит неориентированный взвешенный граф igraph из матрицы весов.

    Матрица симметрична, поэтому берётся только верхний треугольник, чтобы
    каждое ребро не дублировалось.
    """
    upper = sparse.triu(weights, k=1).tocoo()
    edges = list(zip(upper.row.tolist(), upper.col.tolist()))
    edge_weights = upper.data.tolist()

    graph = ig.Graph(n=len(doc_ids), edges=edges, directed=False)
    graph.es["weight"] = edge_weights
    graph.vs["doc_id"] = doc_ids
    graph.vs["title"] = titles
    return graph
