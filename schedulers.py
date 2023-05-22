from abc import ABC, abstractmethod

import numpy as np

class LRScheduler(ABC):
    
    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def get_stamp(self):
        pass
    
    
class LRFixedScheduler(ABC):
    def __init__(self, init_lr: float) -> None:
        super().__init__()
        self.init_lr: float = init_lr
        self.num_steps: int = 0
        self.lr = init_lr
        
        
    def step(self) -> float:
        self.num_steps += 1
        return self.lr
    
    
    def get_stamp(self) -> int:
        return self.num_steps
    
    
class InverseDecayScheduler(ABC):
    def __init__(self, init_lr: float) -> None:
        super().__init__()
        self.init_lr: float = init_lr
        self.num_steps: int = 0
        self.lr = init_lr
        
        
    def step(self) -> float:
        self.num_steps += 1
        return self.lr / self.num_steps
    
    
    def get_stamp(self) -> int:
        return self.num_steps
    
    
class CyclicScheduler(LRScheduler):
    def __init__(self, min_lr: float=1e-5, max_lr: float=1e-2, num_increase: int=50, num_decrease: int=50) -> None:
        super().__init__()
        self.num_steps: int = 0
        self.lr = min_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_increase  = num_increase
        self.num_decrease = num_decrease
        
    def step(self) -> float:
        self.num_steps += 1
    
        res_step = self.num_steps % (self.num_increase + self.num_decrease)

        if res_step <= self.num_increase:
            self.lr =  self.min_lr + (self.max_lr - self.min_lr) * res_step / self.num_increase
        else:
            self.lr = self.max_lr - (self.max_lr - self.min_lr) * (res_step - self.num_increase) / (self.num_decrease)
            
        return self.lr
    
    def get_stamp(self) -> int:
        return self.num_steps
    
    

class LinearLRDecay(LRScheduler):
    def __init__(self, init_lr: float, min_lr: float=1e-6, max_steps: int=10000) -> None:
        super().__init__()
        self.init_lr: float = init_lr
        self.min_lr: float = min_lr
        self.num_steps: int = 0
        self.lr = init_lr
        self.max_steps: int = max_steps
        
    
    def step(self) -> float:
        self.num_steps += 1
        
        if self.num_steps >= self.max_steps:
            self.lr = self.min_lr
        else:
            self.lr = self.init_lr - self.num_steps * (self.init_lr - self.min_lr) / self.max_steps
            
        return self.lr
    
    
    def get_stamp(self) -> int:
        return self.num_steps
    
    
class ExponentialDecayScheduler(LRScheduler):
    def __init__(self, init_lr: float, alpha: float=0.001) -> None:
        super().__init__()
        self.init_lr: float = init_lr
        self.alpha = alpha
        self.num_steps: int = 0
        self.lr = init_lr
        
        
    def step(self) -> float:
        self.num_steps += 1
        self.lr = self.init_lr * np.exp(-self.alpha*self.num_steps)
        return self.lr
    
    
    def get_stamp(self) -> int:
        return self.num_steps
    
    
class FactorDecayScheduler(LRScheduler):
    def __init__(self, init_lr: float, alpha: float=0.2, cycle: int=1000) -> None:
        super().__init__()
        self.init_lr: float = init_lr
        self.alpha = alpha
        self.cycle = cycle
        self.num_steps: int = 0
        self.lr = init_lr
        
    def step(self) -> float:
        self.num_steps += 1
        if self.num_steps % self.cycle == 0:
            self.lr *= self.alpha
        
        return self.lr
    
    def get_stamp(self) -> int:
        return self.num_steps
    

class SquareRootScheduler(LRScheduler):
    def __init__(self, init_lr: float) -> None:
        super().__init__()
        self.init_lr: float = init_lr
        self.num_steps: int = 0
        self.lr = init_lr
        
        
    def step(self) -> float:
        self.num_steps += 1
        self.lr = self.init_lr / (np.sqrt(self.num_steps))
        return self.lr
    
    
    def get_stamp(self) -> int:
        return self.num_steps
    
    
class CosineLRSccheduler(LRScheduler):
    def __init__(self, min_lr: float=1e-5, max_lr: float=1e-2, max_steps: int=10000) -> None:
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = 0
        self.max_steps = max_steps
        self.lr = max_lr
        
    def step(self) -> float:
        self.num_steps += 1
        self.lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos((self.num_steps * np.pi)/self.max_steps))
        return self.lr
    
    
    def get_stamp(self) -> int:
        return self.num_steps