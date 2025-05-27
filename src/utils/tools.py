import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 2025):
    """Функция фиксирования рандомизации для воспроизводимости

    Args:
        seed (int): Начальное значение генератора чисел
    """
    np.random.seed(seed)
    pd.core.common.random_state(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
