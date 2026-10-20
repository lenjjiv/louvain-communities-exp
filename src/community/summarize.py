"""Сводки по сообществам: топ-n-граммы и примеры документов.

Файл отвечает за «человекочитаемое» описание найденных сообществ, чтобы можно
было глазами понять, что за тексты в них попали. Сообщество — это группа
документов; топ-n-граммы здесь лишь метка-подпись сообщества (самые частые
n-граммы внутри группы), а не сами сообщества. Для каждого сообщества
приводятся примеры документов с заголовком и фрагментом текста.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.data.load_corpus import Document


@dataclass
class SampleDocument:
    """Пример документа внутри сообщества."""

    doc_id: str
    title: str
    text_snippet: str


def document_label(document: Document, max_len: int = 60) -> str:
    """Возвращает подпись документа.

    В корпусе RuBQRetrieval поле ``title`` пустое у всех документов, поэтому
    подписью служат первые слова текста — иначе в отчётах и на графе не за что
    зацепиться глазами.
    """
    if document.title:
        return document.title
    return document.text[:max_len].replace("\n", " ").strip()


@dataclass
class CommunitySummary:
    """Описание одного сообщества (группы документов)."""

    community_id: int
    size: int
    top_ngrams: list[tuple[str, int]]
    sample_documents: list[SampleDocument]


def summarize_communities(
    membership: list[int],
    doc_ngrams: list[set[str]],
    documents: list[Document],
    top_ngrams: int = 15,
    sample_docs: int = 5,
    snippet_len: int = 300,
    min_size: int = 1,
) -> list[CommunitySummary]:
    """Строит сводки по сообществам, отсортированные по размеру.

    Сообщества размером меньше ``min_size`` пропускаются: микрогруппы и
    одиночные документы — это шум разреживания, а не тематические кластеры.
    """
    groups: dict[int, list[int]] = {}
    for idx, comm in enumerate(membership):
        groups.setdefault(comm, []).append(idx)

    summaries: list[CommunitySummary] = []
    for comm, members in groups.items():
        if len(members) < min_size:
            continue
        counter: Counter[str] = Counter()
        for idx in members:
            counter.update(doc_ngrams[idx])
        # Характерность n-граммы — её частота внутри сообщества; берём самые
        # частые, они и формируют «тему» сообщества.
        top = counter.most_common(top_ngrams)
        samples = [
            SampleDocument(
                doc_id=documents[idx].doc_id,
                title=document_label(documents[idx]),
                text_snippet=documents[idx].text[:snippet_len].replace("\n", " "),
            )
            for idx in members[:sample_docs]
        ]
        summaries.append(
            CommunitySummary(
                community_id=comm,
                size=len(members),
                top_ngrams=top,
                sample_documents=samples,
            )
        )

    summaries.sort(key=lambda s: s.size, reverse=True)
    return summaries
