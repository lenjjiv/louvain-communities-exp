"""Настройка логирования эксперимента.

Файл отвечает за единообразный вывод статуса пайплайна: сообщения идут и в
консоль, и в файл ``run.log``. Логирование существует потому, что стадии
(лемматизация, матрица пересечений, детекция сообществ) тяжёлые, и без статуса
непонятно, работает процесс или завис, а после прогона нужно видеть, сколько
времени ушло на каждую стадию.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO, log_file: str = "run.log") -> None:
    """Настраивает корневой логгер: консоль + файл, с временем и уровнем.

    Повторный вызов пересобирает хендлеры, чтобы не дублировать вывод при
    запуске из ноутбука или повторном вызове ``main``.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
