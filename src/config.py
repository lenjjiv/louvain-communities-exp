"""Конфигурация эксперимента.

Файл отвечает за описание всех ручек эксперимента в виде вложенных dataclass'ов
и за переопределение любого поля через CLI. Конфиг существует, чтобы можно было
крутить параметры без правки кода: значения по умолчанию заданы здесь, а
конкретный прогон переопределяется аргументами вида ``--preprocess.mode stem``.

Вложенность отражает стадии пайплайна: data -> preprocess -> weighting ->
sparsify -> community -> viz. Путь к полю в CLI совпадает с путём в dataclass
(например, ``--community.resolution 1.5``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class DataConfig:
    """Загрузка корпуса и подвыборка."""

    corpus_path: str = "data/RuBQRetrieval/corpus/test-00000-of-00001.parquet"
    # Подвыборка нужна для быстрой отладки: полный корпус — 56k документов,
    # на нём матрица пересечений n-грамм слишком тяжёлая для итераций.
    sample_size: int = 10_000
    seed: int = 42


@dataclass
class PreprocessConfig:
    """Токенизация, нормализация и построение n-грамм."""

    # light — только нижний регистр и фильтрация; lemmatize — pymorphy3;
    # stem — snowball-стемминг (быстро, без словарей).
    mode: str = "lemmatize"
    ngram_min: int = 1
    ngram_max: int = 2
    # Отсечение слишком редких и слишком частых n-грамм по числу документов.
    min_doc_freq: int = 2
    max_doc_freq: float = 0.5
    max_vocab: int = 50000


@dataclass
class WeightingConfig:
    """Способ взвешивания рёбер между документами."""

    # raw — сырое число общих n-грамм; ppmi — positive PMI;
    # condprob — условная вероятность P(общая n-грамма | документ).
    method: str = "raw"


@dataclass
class SparsifyConfig:
    """Разреживание графа."""

    # topk — k сильнейших соседей на узел; threshold — порог по весу;
    # both — сначала порог, затем top-k.
    method: str = "topk"
    top_k: int = 10
    min_weight: float = 0.0


@dataclass
class CommunityConfig:
    """Параметры детекции сообществ."""

    algorithm: str = "leiden"  # leiden | louvain
    resolution: float = 1.0
    n_iterations: int = 2
    seed: int = 42
    # Сообщества меньше этого размера отбрасываются из отчётов: одиночные
    # документы и микрогруппы — это шум разреживания, а не тематические кластеры.
    min_community_size: int = 10


@dataclass
class ReportConfig:
    """Содержимое отчётов по сообществам (report.json и communities.txt)."""

    # Сколько самых частых n-грамм показывать как метку сообщества.
    top_ngrams: int = 50
    # Сколько примеров документов приводить на сообщество.
    sample_docs: int = 10
    # Длина фрагмента текста документа в примере (символов).
    snippet_len: int = 700


@dataclass
class VizConfig:
    """Визуализация графа через PyVis."""

    # PyVis тянет только подвыборку: на полном графе браузер умрёт.
    max_nodes: int = 500
    output: str = "outputs/graph.html"


@dataclass
class ExperimentConfig:
    """Корневой конфиг эксперимента."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    sparsify: SparsifyConfig = field(default_factory=SparsifyConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    cache_dir: str = "cache"
    output_dir: str = "outputs"


def _flatten(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """Разворачивает вложенный dataclass в плоский словарь ``путь -> значение``."""
    flat: dict[str, Any] = {}
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        path = f"{prefix}{f.name}"
        if is_dataclass(value):
            flat.update(_flatten(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def _coerce(current: Any, raw: str) -> Any:
    """Приводит строку из CLI к типу текущего значения поля."""
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "y"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def apply_overrides(cfg: ExperimentConfig, overrides: dict[str, str]) -> ExperimentConfig:
    """Применяет CLI-переопределения к конфигу, проверяя корректность путей."""
    valid = _flatten(cfg)
    for path, raw in overrides.items():
        if path not in valid:
            raise KeyError(
                f"Неизвестный параметр конфига: '{path}'. "
                f"Доступные: {', '.join(sorted(valid))}"
            )
        section, _, name = path.rpartition(".")
        target = cfg if not section else _resolve_section(cfg, section)
        setattr(target, name, _coerce(valid[path], raw))
    return cfg


def _resolve_section(cfg: ExperimentConfig, section: str) -> Any:
    """Возвращает вложенный dataclass по точечному пути секции."""
    target: Any = cfg
    for part in section.split("."):
        target = getattr(target, part)
    return target


def parse_cli(argv: list[str] | None = None) -> ExperimentConfig:
    """Собирает конфиг из значений по умолчанию и CLI-переопределений.

    Поддерживает форму ``--section.field value`` и ``--section.field=value``.
    Разбор ручной, потому что argparse с позиционным ``nargs="*"`` перехватывает
    значения неизвестных опций и мешает отделить ключи от значений.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)

    overrides: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Ожидался параметр вида --section.field, получено: '{token}'")
        key = token[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            overrides[key] = value
            i += 1
        else:
            if i + 1 >= len(tokens):
                raise ValueError(f"Для параметра '--{key}' не указано значение")
            overrides[key] = tokens[i + 1]
            i += 2

    return apply_overrides(ExperimentConfig(), overrides)
