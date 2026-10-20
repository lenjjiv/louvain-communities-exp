"""Матрица пересечений n-грамм между документами и взвешивание рёбер.

Файл отвечает за построение разреженной матрицы «документ × n-грамма» и
вычисление веса ребра между двумя документами как меры их лексической
пересекаемости. Узлы графа — документы, рёбра — общие n-граммы.

Реализованы три способа взвешивания, потому что они по-разному подчёркивают
структуру и интересны для сравнения:

* ``raw`` — сырое число общих n-грамм (просто и наглядно, но доминируют
  длинные документы и частотные n-граммы);
* ``ppmi`` — positive PMI (штрафует повсеместные n-граммы, выявляет
  содержательные связи);
* ``condprob`` — условная вероятность P(общая n-грамма | документ),
  нормирует на длину документа; асимметрична, поэтому симметризуется максимумом.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from src.config import WeightingConfig


def build_doc_term_matrix(
    doc_ngrams: list[set[str]],
    max_vocab: int,
    min_doc_freq: int,
    max_doc_freq: float,
) -> tuple[sparse.csr_matrix, list[str]]:
    """Строит бинарную матрицу «документ × n-грамма» с фильтрацией словаря.

    Фильтрация по частоте документа важна: слишком редкие n-граммы дают шум,
    слишком частые (встречающиеся почти в каждом документе) не различают
    документы и только добавляют плотности графу.
    """
    vocab: dict[str, int] = {}
    for ngrams in doc_ngrams:
        for ng in ngrams:
            vocab.setdefault(ng, len(vocab))

    n_docs = len(doc_ngrams)
    rows: list[int] = []
    cols: list[int] = []
    for i, ngrams in enumerate(doc_ngrams):
        for ng in ngrams:
            rows.append(i)
            cols.append(vocab[ng])

    data = np.ones(len(rows), dtype=np.float32)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_docs, len(vocab)))

    doc_freq = np.asarray(matrix.sum(axis=0)).ravel()
    max_df = max_doc_freq * n_docs
    keep = (doc_freq >= min_doc_freq) & (doc_freq <= max_df)
    matrix = matrix[:, keep]

    # Ограничение словаря: оставляем самые частотные n-граммы, чтобы матрица
    # пересечений не разрасталась на больших подвыборках.
    if matrix.shape[1] > max_vocab:
        freq = np.asarray(matrix.sum(axis=0)).ravel()
        top = np.argsort(freq)[::-1][:max_vocab]
        matrix = matrix[:, np.sort(top)]

    matrix = matrix.tocsr()
    matrix.data[:] = 1.0
    return matrix, list(vocab.keys())


def _intersection_counts(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Считает число общих n-грамм для каждой пары документов.

    Для бинарной матрицы X произведение X @ X.T даёт в ячейке (i, j) число
    n-грамм, присутствующих одновременно в документах i и j.
    """
    cooc = (matrix @ matrix.T).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    return cooc


def _ppmi(cooc: sparse.csr_matrix) -> sparse.csr_matrix:
    """Positive PMI по матрице совместной встречаемости."""
    total = cooc.sum()
    if total == 0:
        return cooc
    row_sums = np.asarray(cooc.sum(axis=1)).ravel()
    # p_i * p_j в знаменателе; нулевые строки дают деление на ноль, поэтому
    # заменяем нули на единицу — такие ячейки всё равно отсеются по p_ij = 0.
    denom = np.outer(row_sums, row_sums) / (total * total)
    denom[denom == 0] = 1.0

    cooc = cooc.tocoo()
    p_ij = cooc.data / total
    pmi = np.log(p_ij / denom[cooc.row, cooc.col])
    pmi = np.maximum(pmi, 0.0)
    result = sparse.csr_matrix((pmi, (cooc.row, cooc.col)), shape=cooc.shape)
    result.eliminate_zeros()
    return result


def _condprob(cooc: sparse.csr_matrix, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Условная вероятность P(общая n-грамма | документ), симметризованная максимумом."""
    lengths = np.asarray(matrix.sum(axis=1)).ravel()
    lengths[lengths == 0] = 1.0

    cooc = cooc.tocoo()
    cond = cooc.data / lengths[cooc.row]
    forward = sparse.csr_matrix((cond, (cooc.row, cooc.col)), shape=cooc.shape)
    # Симметризация: ребро неориентированного графа берёт максимум из двух
    # направленных условных вероятностей.
    sym = forward.maximum(forward.T).tocsr()
    sym.eliminate_zeros()
    return sym


def compute_weights(
    matrix: sparse.csr_matrix,
    cfg: WeightingConfig,
) -> sparse.csr_matrix:
    """Вычисляет матрицу весов рёбер выбранным способом."""
    cooc = _intersection_counts(matrix)
    if cfg.method == "raw":
        return cooc
    if cfg.method == "ppmi":
        return _ppmi(cooc)
    if cfg.method == "condprob":
        return _condprob(cooc, matrix)
    raise ValueError(f"Неизвестный способ взвешивания: '{cfg.method}'")
