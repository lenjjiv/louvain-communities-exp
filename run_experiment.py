"""Точка входа эксперимента по детекции сообществ документов RuBQRetrieval.

Запуск:
    python run_experiment.py --community.algorithm leiden --community.resolution 1.0
    python run_experiment.py --weighting.method ppmi --sparsify.top_k 15

Все параметры конфига переопределяются через CLI в формате ``--section.field``.
"""

from __future__ import annotations

import json
import logging

from src.config import parse_cli
from src.logging_setup import setup_logging
from src.pipeline import run

logger = logging.getLogger(__name__)


def main() -> None:
    """Парсит конфиг, прогоняет пайплайн и печатает краткую сводку."""
    setup_logging()
    cfg = parse_cli()
    result = run(cfg)

    logger.info("Документов: %d", result["n_documents"])
    logger.info("Узлов: %d, рёбер: %d", result["n_nodes"], result["n_edges"])
    logger.info("Сообществ: %d, модулярность: %.4f",
                result["n_communities"], result["modularity"])
    logger.info("Время по стадиям (с): %s",
                json.dumps(result["timings"], ensure_ascii=False))
    logger.info("HTML-граф: %s", result["html_path"])
    logger.info("Отчёт (JSON): %s", result["report_path"])
    logger.info("Отчёт (текст): %s", result["text_report_path"])
    logger.info("Топ-5 сообществ (метка = топ-n-граммы, ниже — примеры документов):")
    for comm in result["communities"][:5]:
        top = ", ".join(ng for ng, _ in comm["top_ngrams"][:8])
        logger.info("  Сообщество %d (документов: %d)", comm["community_id"], comm["size"])
        logger.info("    метка: %s", top)
        for doc in comm["sample_documents"][:3]:
            logger.info("    - %s: %s", doc["title"], doc["text_snippet"][:120])


if __name__ == "__main__":
    main()
