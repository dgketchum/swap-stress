"""Neural experiments -- tabular NN and masked autoencoder.

These constants were in ``map/learning/__init__.py``. The masked autoencoder is
their only consumer, so they moved here rather than into ``swapstress.model``,
which trains a quantile random forest and must not pull in torch. VG_PARAMS and
DROP_FEATURES were dropped in the move -- nothing referenced them.
"""

import torch

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
