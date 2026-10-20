"""Предобработка текста и построение n-грамм.

Файл отвечает за превращение сырого текста документа в набор n-грамм, по
которым затем считается пересечение документов. Поддерживаются три режима
нормализации, потому что они дают заметно разное качество сообществ:

* ``light`` — только нижний регистр и фильтрация, без словарей (быстро);
* ``lemmatize`` — лемматизация pymorphy3 (лучшее качество для русского,
  но требует словари и работает медленнее);
* ``stem`` — snowball-стемминг (компромисс: быстро, без словарей, но грубее).

Результат кэшируется на уровне пайплайна, поэтому стоимость лемматизации
платится один раз на подвыборку.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pymorphy3
from nltk.stem.snowball import SnowballStemmer

from src.config import PreprocessConfig

# Русские стоп-слова: частотные служебные слова, которые не несут тематики и
# только зашумляют пересечение документов.
STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
    по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг
    ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом
    себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам
    чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому
    этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
    куда зачем всех никогда можно при наконец два об другой хоть после над больше
    тот через эти нас про всего них какая много разве три эту моя впрочем хорошо
    свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно
    всю между это
    """.split()
)

_WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


@lru_cache(maxsize=1)
def _get_morph() -> pymorphy3.MorphAnalyzer:
    """Ленивая инициализация морфоанализатора (словари грузятся один раз)."""
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=1)
def _get_stemmer() -> SnowballStemmer:
    """Ленивая инициализация стеммера для русского языка."""
    return SnowballStemmer("russian")


def tokenize(text: str) -> list[str]:
    """Извлекает слова из текста, отбрасывая стоп-слова и короткие токены."""
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def normalize(tokens: list[str], mode: str) -> list[str]:
    """Приводит токены к нормальной форме согласно выбранному режиму."""
    if mode == "light":
        return tokens
    if mode == "lemmatize":
        morph = _get_morph()
        return [morph.parse(t)[0].normal_form for t in tokens]
    if mode == "stem":
        stemmer = _get_stemmer()
        return [stemmer.stem(t) for t in tokens]
    raise ValueError(f"Неизвестный режим предобработки: '{mode}'")


def build_ngrams(tokens: list[str], ngram_min: int, ngram_max: int) -> set[str]:
    """Строит множество n-грамм (n от ``ngram_min`` до ``ngram_max``).

    n-грамма представлена строкой из токенов через пробел — так её удобно
    хранить в словаре признаков и показывать в сводках.
    """
    ngrams: set[str] = set()
    for n in range(ngram_min, ngram_max + 1):
        if n == 1:
            ngrams.update(tokens)
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i : i + n]))
    return ngrams


def preprocess_document(text: str, cfg: PreprocessConfig) -> set[str]:
    """Полный конвейер: токенизация -> нормализация -> n-граммы."""
    tokens = tokenize(text)
    tokens = normalize(tokens, cfg.mode)
    return build_ngrams(tokens, cfg.ngram_min, cfg.ngram_max)
