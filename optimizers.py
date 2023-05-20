from abc import ABC, abstractclassmethod
from typing import Tuple

import numpy as np

from schedulers import LRScheduler


class Optimizer(ABC):
    
    @abstractclassmethod
    def step(self):
        pass
    
    

class GradientDescentOptimizer(Optimizer):
    
    def __init__(self, lrscheduler: LRScheduler) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        #self.regularizer = regularizer
        
        
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        
        lr = self.lrscheduler.step()
        
        weights -= lr * dweights
        
        return weights
        