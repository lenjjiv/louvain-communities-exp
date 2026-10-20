"""Сводки по сообществам: топ-n-граммы и примеры документов.

Файл отвечает за «человекочитаемое» описание найденных сообществ, чтобы можно
было глазами понять, что за тексты в них попали. Для каждого сообщества
считаются самые характерные n-граммы (по частоте внутри сообщества) и
приводятся примеры заголовков документов.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.data.load_corpus import Document


@dataclass
class CommunitySummary:
    """Описание одного сообщества."""

    community_id: int
    size: int
    top_ngrams: list[tuple[str, int]]
    sample_titles: list[str]


def summarize_communities(
    membership: list[int],
    doc_ngrams: list[set[str]],
    documents: list[Document],
    top_ngrams: int = 15,
    sample_titles: int = 5,
) -> list[CommunitySummary]:
    """Строит сводки по всем сообществам, отсортированные по размеру."""
    groups: dict[int, list[int]] = {}
    for idx, comm in enumerate(membership):
        groups.setdefault(comm, []).append(idx)

    summaries: list[CommunitySummary] = []
    for comm, members in groups.items():
        counter: Counter[str] = Counter()
        for idx in members:
            counter.update(doc_ngrams[idx])
        # Характерность n-граммы — её частота внутри сообщества; берём самые
        # частые, они и формируют «тему» сообщества.
        top = counter.most_common(top_ngrams)
        titles = [documents[idx].title for idx in members[:sample_titles]]
        summaries.append(
            CommunitySummary(
                community_id=comm,
                size=len(members),
                top_ngrams=top,
                sample_titles=titles,
            )
        )

    summaries.sort(key=lambda s: s.size, reverse=True)
    return summaries
