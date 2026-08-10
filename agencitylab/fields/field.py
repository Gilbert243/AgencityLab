from dataclasses import dataclass
import numpy as np

@dataclass
class AgencityField:
    x: np.ndarray
    t: np.ndarray
    beta: np.ndarray
