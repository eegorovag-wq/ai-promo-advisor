"""Запуск: python -m promo_advisor ..."""

import sys

from .cli import main


def _configure_output() -> None:
    """Печатает Unicode предсказуемо даже в Windows с кодировкой cp1251."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


_configure_output()
sys.exit(main())
