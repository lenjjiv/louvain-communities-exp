"""Точка входа эксперимента по детекции сообществ документов RuBQRetrieval.

Запуск:
    python run_experiment.py --community.algorithm leiden --community.resolution 1.0
    python run_experiment.py --weighting.method ppmi --sparsify.top_k 15

Все параметры конфига переопределяются через CLI в формате ``--section.field``.
"""

from __future__ import annotations

import json

from src.config import parse_cli
from src.pipeline import run


def main() -> None:
    """Парсит конфиг, прогоняет пайплайн и печатает краткую сводку."""
    cfg = parse_cli()
    result = run(cfg)

    print(f"Документов: {result['n_documents']}")
    print(f"Узлов: {result['n_nodes']}, рёбер: {result['n_edges']}")
    print(f"Сообществ: {result['n_communities']}, модулярность: {result['modularity']:.4f}")
    print(f"Время по стадиям (с): {json.dumps(result['timings'], ensure_ascii=False)}")
    print(f"HTML-граф: {result['html_path']}")
    print(f"Отчёт (JSON): {result['report_path']}")
    print(f"Отчёт (текст): {result['text_report_path']}")
    print("\nТоп-5 сообществ (метка = топ-n-граммы, ниже — примеры документов):")
    for comm in result["communities"][:5]:
        top = ", ".join(ng for ng, _ in comm["top_ngrams"][:8])
        print(f"\n  Сообщество {comm['community_id']} (документов: {comm['size']})")
        print(f"    метка: {top}")
        for doc in comm["sample_documents"][:3]:
            print(f"    - {doc['title']}: {doc['text_snippet'][:120]}")


if __name__ == "__main__":
    main()
