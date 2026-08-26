import os
import random

import numpy as np
import torch


def set_random_seed(seed=42, deterministic=False):
    if not isinstance(seed, int):
        raise TypeError('seed must be an integer')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    return seed


