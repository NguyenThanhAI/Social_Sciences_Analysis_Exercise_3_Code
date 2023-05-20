from abc import ABC, abstractmethod


class LRScheduler(ABC):
    
    @abstractmethod
    def step(self):
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