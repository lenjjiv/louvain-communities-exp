"""Детекция сообществ алгоритмами Leiden и Louvain.

Файл отвечает за запуск алгоритмов поиска сообществ на графе документов и
возврат разбиения узлов на сообщества. Оба алгоритма оптимизируют модулярность,
но Leiden дополнительно гарантирует связность сообществ и не создаёт «плохих»
мелких кластеров, поэтому их сравнение и есть цель эксперимента.

Параметр ``resolution`` управляет масштабом сообществ: больше — мельче и
больше сообществ, меньше — крупнее и меньше.
"""

from __future__ import annotations

import igraph as ig
import leidenalg

from src.config import CommunityConfig


def detect_communities(graph: ig.Graph, cfg: CommunityConfig) -> list[int]:
    """Возвращает список номеров сообществ для каждого узла графа."""
    if cfg.algorithm == "leiden":
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=cfg.resolution,
            n_iterations=cfg.n_iterations,
            seed=cfg.seed,
        )
    elif cfg.algorithm == "louvain":
        # Реализация Louvain в igraph не принимает n_iterations и seed:
        # алгоритм детерминированно останавливается по достижении локального
        # максимума модулярности, поэтому эти ручки к нему неприменимы.
        partition = graph.community_multilevel(
            weights="weight",
            resolution=cfg.resolution,
        )
    else:
        raise ValueError(f"Неизвестный алгоритм: '{cfg.algorithm}'")
    return list(partition.membership)


def modularity(graph: ig.Graph, membership: list[int]) -> float:
    """Считает модулярность разбиения (метрика качества сообществ)."""
    return graph.modularity(membership, weights="weight")
