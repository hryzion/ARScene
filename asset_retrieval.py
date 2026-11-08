"""Size-only asset retrieval shared by legacy decoding and scene JSON tools."""

import numpy as np


def get_modelid_by_size(q_size, category, model_metadata):
    """Return (model ID, asset size), or None if the category has no candidates.

    Sizes are full local XYZ side lengths. This is the implementation extracted
    from utils.get_modelid_by_size: same-category squared L2, with the original
    metadata iteration order and numpy.argsort selection (including ties).
    """
    if category not in model_metadata or not model_metadata[category]:
        return None
    model_ids = []
    mses_size = []

    for model_id, model_latent_and_size in model_metadata[category].items():
        model_size = np.array(model_latent_and_size['size'])
        model_ids.append(model_id)
        mses_size.append(np.sum((q_size - model_size)**2))

    ind = np.argsort(mses_size)
    return model_ids[ind[0]], model_metadata[category][model_ids[ind[0]]]['size']
