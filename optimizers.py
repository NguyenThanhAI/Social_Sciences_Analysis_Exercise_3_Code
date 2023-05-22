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
    
    
class AdamOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler, beta_1: float=0.5, beta_2: float=0.99) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m: np.ndarray = None
        self.v: np.ndarray = None
        
    def init_state(self, weights: np.ndarray) -> None:
        self.m = np.zeros_like(weights)
        self.v = np.zeros_like(weights)
    
    
    @staticmethod
    def adam_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = beta_1 * m + (1 - beta_1) * dweights
        v = beta_2 * v + (1 - beta_2) * dweights**2

        m_hat = m / (1 - beta_1**t)
        v_hat = v / (1 - beta_2**t)

        p = m_hat / (np.sqrt(v_hat) + epsilon)

        return p, m, v
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        
        if self.m is None or self.v is None:
            self.init_state(weights=weights)
        
        lr = self.lrscheduler.step()
        t = self.lrscheduler.get_stamp()
        
        p, self.m, self.v = self.adam_step(dweights=dweights, m=self.m, v=self.v, t=t,
                                           beta_1=self.beta_1, beta_2=self.beta_2)
        
        weights -= lr * p
        
        return weights
    
    
class AvagradOptimizer(Optimizer):
    
    def __init__(self, lrscheduler: LRScheduler, beta_1: float=0.5, beta_2: float=0.99) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m: np.ndarray = None
        self.v: np.ndarray = None
        
    def init_state(self, weights: np.ndarray) -> None:
        self.m = np.zeros_like(weights)
        self.v = np.zeros_like(weights)
    
    @staticmethod    
    def avagrad_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> np.ndarray:
        d = dweights.shape[0]
        m = beta_1 * m + (1 - beta_1) * dweights

        eta = 1 / (np.sqrt(v) + epsilon)

        p = (eta / np.linalg.norm(eta/np.sqrt(d))) * m

        v = beta_2 * v + (1 - beta_2) * dweights**2

        return p, m, v
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        
        if self.m is None or self.v is None:
            self.init_state(weights=weights)
            
        
        lr = self.lrscheduler.step()
        
        p, self.m, self.v = self.avagrad_step(dweights=dweights, m=self.m, v=self.v, beta_1=self.beta_1, beta_2=self.beta_2)
        
        weights -= lr * p
        
        return weights
        

class RadamOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler, beta_1: float=0.5, beta_2: float=0.99) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m: np.ndarray = None
        self.v: np.ndarray = None
        
    def init_state(self, weights: np.ndarray) -> None:
        self.m = np.zeros_like(weights)
        self.v = np.zeros_like(weights)
        
        
    @staticmethod
    def radam_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho_inf = (2 / (1 - beta_2)) - 1

        m = beta_1 * m + (1 - beta_1) * dweights
        v = beta_2 * v + (1 - beta_2) * dweights**2

        m_hat = m / (1 - beta_1**t)

        rho = rho_inf - (2 * t * beta_2**t) / (1 - beta_2**t)

        if rho > 4:
            v_hat = np.sqrt(v / (1 - beta_2**t))
            r = np.sqrt(((rho - 4) * (rho - 2) * rho_inf)/((rho_inf - 4) * (rho_inf - 2) * rho))
            p = (r * m_hat) / (v_hat + epsilon)
        else:
            p = m_hat

        return p, m, v
    
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        if self.m is None or self.v is None:
            self.init_state(weights=weights)
        
        lr = self.lrscheduler.step()
        t = self.lrscheduler.get_stamp()
        
        p, self.m, self.v = self.radam_step(dweights=dweights, m=self.m, v=self.v, t=t,
                                            beta_1=self.beta_1, beta_2=self.beta_2)
        
        weights -= lr * p
        
        return weights
    
    
class AdamaxOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler, beta_1: float=0.5, beta_2: float=0.99) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m: np.ndarray = None
        self.u: np.ndarray = None
        
        
    def init_state(self, weights: np.ndarray) -> None:
        self.m = np.zeros_like(weights)
        self.u = np.zeros_like(weights)
        
    @staticmethod
    def adamax_step(dweights: np.ndarray, m: np.ndarray, u: float, t: int, beta_1: float=0.9, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, float]:

        m = beta_1 * m + (1 - beta_1) * dweights
        u = np.maximum(beta_2 * u, np.max(dweights))

        p =  m / ((1 - beta_1**t) * u + epsilon)

        return p, m, u
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        if self.m is None or self.u is None:
            self.init_state(weights=weights)
            
        lr = self.lrscheduler.step()
        t = self.lrscheduler.get_stamp()
        
        p, self.m, self.u = self.adamax_step(dweights=dweights, m=self.m, u=self.u, t=t,
                                             beta_1=self.beta_1, beta_2=self.beta_2)
        
        weights -= lr * p
        
        return weights
    
    
    
class NadamOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler, beta_1: float=0.5, beta_2: float=0.99) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m: np.ndarray = None
        self.v: np.ndarray = None
        
    def init_state(self, weights: np.ndarray) -> None:
        self.m = np.zeros_like(weights)
        self.v = np.zeros_like(weights)
        
    @staticmethod
    def nadam_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = beta_1 * m + (1 - beta_1) * dweights
        v = beta_2 * v + (1 - beta_2) * dweights**2

        m_hat = m / (1 - beta_1**t)
        v_hat = v / (1 - beta_2**t)

        p = (beta_1 * m_hat + ((1 - beta_1)/(1 - beta_1**t))*dweights) / (np.sqrt(v_hat) + epsilon)

        return p, m, v
    
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        if self.m is None or self.v is None:
            self.init_state(weights=weights)
        
        lr = self.lrscheduler.step()
        t = self.lrscheduler.get_stamp()
        
        p, self.m, self.v = self.nadam_step(dweights=dweights, m=self.m, v=self.v, t=t,
                                            beta_1=self.beta_1, beta_2=self.beta_2)
        
        weights -= lr * p
        
        return weights
    
    
class AMSGradOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler, beta_1: float=0.5, beta_2: float=0.99) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m: np.ndarray = None
        self.v: np.ndarray = None
        self.v_hat: np.ndarray = None
        
        
    def init_state(self, weights: np.ndarray) -> None:
        self.m = np.zeros_like(weights)
        self.v = np.zeros_like(weights)
        self.v_hat = np.zeros_like(weights)
        
    @staticmethod
    def amsgrad_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, v_hat: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = beta_1 * m + (1 - beta_1) * dweights
        v = beta_2 * v + (1 - beta_2) * dweights**2

        v_hat = np.maximum(v_hat, v)

        p =  m / (np.sqrt(v_hat) + epsilon)

        return p, m, v, v_hat
    
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        if self.m is None or self.v is None or self.v_hat is None:
            self.init_state(weights=weights)
        
        lr = self.lrscheduler.step()
        t = self.lrscheduler.get_stamp()
        
        p, self.m, self.v, self.v_hat = self.amsgrad_step(dweights=dweights, m=self.m, v=self.v, v_hat=self.v_hat, t=t,
                                                          beta_1=self.beta_1, beta_2=self.beta_2)
        
        weights -= lr * p
        
        return weights
    
    
class AdaBeliefOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler, beta_1: float=0.5, beta_2: float=0.99) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m: np.ndarray = None
        self.v: np.ndarray = None
        
    def init_state(self, weights: np.ndarray) -> None:
        self.m = np.zeros_like(weights)
        self.v = np.zeros_like(weights)
        
    @staticmethod
    def adabelief_step(dweights: np.ndarray, m: np.ndarray, v: np.ndarray, t: int, beta_1: float=0.5, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = beta_1 * m + (1 - beta_1) * dweights
        v = beta_2 * v + (1 - beta_2) * (dweights - m) ** 2

        m_hat = m / (1 - beta_1**t)
        v_hat = v / (1 - beta_2**t)

        p = m_hat / (np.sqrt(v_hat) + epsilon)

        return p, m, v
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        if self.m is None or self.v is None:
            self.init_state(weights=weights)
        
        lr = self.lrscheduler.step()
        t = self.lrscheduler.get_stamp()
        
        p, self.m, self.v = self.adabelief_step(dweights=dweights, m=self.m, v=self.v, t=t,
                                                beta_1=self.beta_1, beta_2=self.beta_2)
        
        weights -= lr * p
        
        return weights
    
    
class AdagradOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.v: np.ndarray = None
        
    
    def init_state(self, weights: np.ndarray) -> None:
        self.v = np.zeros_like(weights)
        
        
    @staticmethod
    def adagrad_step(dweights: np.ndarray, v: np.ndarray, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
        v = v + dweights**2

        p = dweights / (np.sqrt(v) + epsilon)

        return p, v
    
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.init_state(weights=weights)
        
        lr = self.lrscheduler.step()
        
        p, self.v = self.adagrad_step(dweights=dweights, v=self.v)
        
        weights -= lr * p
        
        return weights
    
    
class RMSPropOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler, beta_2=0.9) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_2 = beta_2
        self.v: np.ndarray = None
        
    
    def init_state(self, weights: np.ndarray) -> None:
        self.v = np.zeros_like(weights)
    
    @staticmethod
    def rmsprop_step(dweights: np.ndarray, v: np.ndarray, beta_2: float=0.9, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
        v = beta_2 * v + (1 - beta_2) * dweights ** 2

        p = dweights / (np.sqrt(v) + epsilon)

        return p, v
    
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.init_state(weights=weights)
        
        lr = self.lrscheduler.step()
        
        p, self.v = self.rmsprop_step(dweights=dweights, v=self.v, beta_2=self.beta_2)
        
        weights -= lr * p
        
        return weights
    
    
class MomentumOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler, beta_1=0.9) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_1 = beta_1
        self.m: np.ndarray = None
        
    
    def init_state(self, weights: np.ndarray) -> None:
        self.m = np.zeros_like(weights)
        
        
    @staticmethod
    def momentum_step(dweights: np.ndarray, m: np.ndarray, beta_1: float=0.9) -> Tuple[np.ndarray, np.ndarray]:
        m = beta_1 * m + (1 - beta_1) * dweights

        p = m

        return p, m
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.init_state(weights=weights)
        
        lr = self.lrscheduler.step()
        
        p, self.m = self.momentum_step(dweights=dweights, m=self.m, beta_1=self.beta_1)
        
        weights -= lr * p
        
        return weights
    
    
class AdadeltaOptimizer(Optimizer):
    def __init__(self, lrscheduler: LRScheduler, beta_2=0.9) -> None:
        super().__init__()
        self.lrscheduler = lrscheduler
        self.beta_2 = beta_2
        self.v: np.ndarray = None
        self.d: np.ndarray = None
        
    def init_state(self, weights: np.ndarray) -> None:
        self.v = np.zeros_like(weights)
        self.d = np.zeros_like(weights)
        
    
    @staticmethod
    def adadelta_step(dweights: np.ndarray, v: np.ndarray, d: np.ndarray, alpha: float, beta_2: float=0.99, epsilon: float=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = beta_2 * v + (1 - beta_2) * dweights ** 2

        p = (np.sqrt(d + epsilon) * dweights) / (np.sqrt(v + epsilon))

        delta_w = - alpha * p

        d = beta_2 * d + (1 - beta_2) * delta_w ** 2

        return p, v, d
    
    
    def step(self, weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
        if self.v is None or self.d is None:
            self.init_state(weights=weights)
        
        lr = self.lrscheduler.step()
        t = self.lrscheduler.get_stamp()
        
        p, self.v, self.d = self.adadelta_step(dweights=dweights, 
                                               v=self.v, 
                                               d=self.d, 
                                               alpha=lr,
                                               beta_2=self.beta_2)
        
        weights -= lr * p
        
        return weights
    
    