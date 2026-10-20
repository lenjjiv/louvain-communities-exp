# Сообщества документов RuBQRetrieval (Leiden vs Louvain)

Эксперимент: строим граф, где **узлы — документы корпуса**, а **рёбра — мера
пересечения их n-грамм**, и ищем сообщества алгоритмами Leiden и Louvain
(igraph + leidenalg). Цель — наглядно потрогать алгоритмы руками: покрутить
ручки, посмотреть, какие тексты попадают в какие сообщества, и сравнить
поведение двух алгоритмов.

## Установка

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

## Запуск

```bash
.venv/bin/python run_experiment.py
```

Все параметры переопределяются через CLI в формате `--section.field value`:

```bash
.venv/bin/python run_experiment.py \
    --data.sample_size 2000 \
    --preprocess.mode lemmatize \
    --weighting.method ppmi \
    --sparsify.method topk --sparsify.top_k 10 \
    --community.algorithm leiden --community.resolution 1.0
```

## Ручки (параметры)

| Параметр | Значения | Смысл |
|----------|----------|-------|
| `data.sample_size` | int | размер подвыборки документов (для быстрой отладки) |
| `data.seed` | int | зерно подвыборки |
| `preprocess.mode` | `light` / `lemmatize` / `stem` | нормализация токенов |
| `preprocess.ngram_min/max` | int | диапазон n-грамм |
| `preprocess.min_doc_freq` | int | отсечь n-граммы реже, чем в N документах |
| `preprocess.max_doc_freq` | float | отсечь n-граммы чаще, чем в доле документов |
| `preprocess.max_vocab` | int | ограничение словаря n-грамм |
| `weighting.method` | `raw` / `ppmi` / `condprob` | способ взвешивания рёбер |
| `sparsify.method` | `topk` / `threshold` / `both` | способ разреживания графа |
| `sparsify.top_k` | int | число соседей на узел |
| `sparsify.min_weight` | float | порог по весу ребра |
| `community.algorithm` | `leiden` / `louvain` | алгоритм детекции |
| `community.resolution` | float | масштаб сообществ (больше — мельче) |
| `community.n_iterations` | int | итерации Leiden (Louvain игнорирует) |
| `community.seed` | int | зерно Leiden (Louvain детерминирован) |
| `community.min_community_size` | int | сообщества меньше этого размера не попадают в отчёты |
| `report.top_ngrams` | int | сколько топ-n-грамм показывать как метку сообщества |
| `report.sample_docs` | int | сколько примеров документов приводить на сообщество |
| `report.snippet_len` | int | длина фрагмента текста документа в примере (символов) |
| `viz.max_nodes` | int | число узлов на интерактивной картинке |

## Кэширование

Тяжёлые стадии кэшируются в `cache/` (n-граммы, матрица весов, граф). Ключ
кэша зависит от параметров стадии, поэтому смена `community.resolution`
переиспользует готовый граф мгновенно, а смена `preprocess.mode` пересчитает
n-граммы. Это даёт быструю обратную связь при переборе ручек.

## Логирование и прогресс

Статус стадий пишется в консоль и в `run.log` (время, уровень, модуль), а
тяжёлые циклы (предобработка, матрица «документ × n-грамма», top-k
разреживание) показывают прогресс-бар tqdm.

## Результаты

- `outputs/graph.html` — интерактивный граф (PyVis), цвет = сообщество.
- `outputs/report.json` — модулярность, число сообществ, топ-n-граммы и
  примеры документов по каждому сообществу.
- `outputs/communities.txt` — человекочитаемый отчёт: по каждому сообществу
  метка из топ-n-грамм и примеры документов с фрагментами текста.

## Структура

```
src/config.py              конфиг (dataclass) + CLI-оверрайды
src/logging_setup.py       настройка логирования (консоль + run.log)
src/data/load_corpus.py    загрузка корпуса с подвыборкой
src/graph/preprocess.py    токенизация, нормализация, n-граммы
src/graph/cooccurrence.py  матрица пересечений n-грамм + взвешивание
src/graph/build_graph.py   разреживание + построение igraph
src/community/detect.py    Leiden / Louvain
src/community/summarize.py сводки по сообществам
src/viz/plot_graph.py      интерактивная визуализация PyVis
src/pipeline.py            end-to-end пайплайн с кэшированием
run_experiment.py          точка входа
```
