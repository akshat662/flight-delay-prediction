"""Deterministic seeding across random, numpy, and (optionally) tensorflow/keras."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Seed PYTHONHASHSEED, random, numpy, and tensorflow/keras if installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        logger.debug("tensorflow not installed; skipping tf seeding")

    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        logger.debug("keras not installed; skipping keras seeding")

    logger.info("All seeds set to %d", seed)
