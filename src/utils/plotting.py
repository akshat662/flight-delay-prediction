"""Shared matplotlib styling helpers for EDA figures.

Keeps figure size, title/grid conventions, and the save-and-close routine in
one place so src/eda/plots.py stays focused on what each figure computes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config

logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (10, 6)
DPI = 150


def new_figure(figsize: tuple[float, float] = DEFAULT_FIGSIZE) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure/axes pair at the given size."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def style_axes(
    ax: plt.Axes,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = True,
) -> None:
    """Apply consistent title/label/grid styling to an axes."""
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)


def save_and_close(fig: plt.Figure, name: str, config: Config) -> Path:
    """Save a figure to reports/figures/<name>.png at 150 dpi and close it."""
    out_path = config.paths.figures_dir / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path
