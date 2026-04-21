import numpy as np
from agencitylab.utils.math_utils import moving_average

def test_moving_average_length():
    x = np.arange(100)
    y = moving_average(x, window_size=5)
    assert len(y) == len(x)
