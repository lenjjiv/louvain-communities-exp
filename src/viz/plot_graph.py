"""Интерактивная визуализация графа документов через PyVis.

Файл отвечает за построение интерактивного HTML-графа, в котором узлы —
документы, цвет — сообщество, а подпись — заголовок документа. PyVis тянет
только подвыборку узлов: на полном графе браузер не справится, поэтому число
узлов ограничивается конфигом, а для отображения берутся самые крупные
сообщества.
"""

from __future__ import annotations

import os

import igraph as ig
from pyvis.network import Network

from src.config import VizConfig

# Палитра для раскраски сообществ; при большем числе сообществ цвета
# повторяются по кругу.
PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
]


def visualize_graph(
    graph: ig.Graph,
    membership: list[int],
    cfg: VizConfig,
) -> str:
    """Строит интерактивный HTML-граф и сохраняет его по пути из конфига.

    Возвращает путь к сохранённому файлу.
    """
    # Отбираем узлы: сначала самые крупные сообщества, чтобы на картинке были
    # именно они, а не случайный шум.
    sizes: dict[int, int] = {}
    for comm in membership:
        sizes[comm] = sizes.get(comm, 0) + 1
    ranked_comms = sorted(sizes, key=lambda c: sizes[c], reverse=True)

    selected: list[int] = []
    for comm in ranked_comms:
        for idx, c in enumerate(membership):
            if c == comm:
                selected.append(idx)
            if len(selected) >= cfg.max_nodes:
                break
        if len(selected) >= cfg.max_nodes:
            break

    selected_set = set(selected)
    net = Network(height="900px", width="100%", notebook=False, directed=False)
    net.barnes_hut()

    for idx in selected:
        comm = membership[idx]
        title = graph.vs["title"][idx] or graph.vs["doc_id"][idx]
        net.add_node(
            idx,
            label=title[:40],
            title=f"[{comm}] {title}",
            color=PALETTE[comm % len(PALETTE)],
        )

    for edge in graph.es:
        src, dst = edge.tuple
        if src in selected_set and dst in selected_set:
            net.add_edge(src, dst, value=float(edge["weight"]))

    os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)
    net.write_html(cfg.output, notebook=False)
    return cfg.output
