from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

from schedulers import *
from optimizers import *


def pca(x: np.ndarray, alpha: float=0.95) -> Tuple[np.ndarray, np.ndarray]:
    
    mu = np.mean(x, axis=0, keepdims=True)

    x_mu = x - mu

    cov_matrix = np.matmul(x_mu.T, x_mu) / x.shape[0]

    w, v = np.linalg.eig(cov_matrix)

    order = np.argsort(w)[::-1]

    w = w[order]
    v = v[:, order]

    rate = np.cumsum(w) / np.sum(w)

    r = np.where(rate >= alpha)

    U = v[:, :(r[0][0] + 1)]

    reduced_x = np.matmul(x, U)

    #print(reduced_x)

    return U, reduced_x

def plot_roc_curve(tpr, fpr):
    roc_auc = auc(fpr, tpr)
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
    plt.legend(loc = "lower right")
    plt.plot([0, 1], [0, 1],"r--")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.grid()
    plt.show()


def create_data(csv_path: str, split_ratio: float=0.4, normalize: str="standardize", chosen_features: str=None, random_state: int=42, use_bias: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    
    x = df.drop("Class", axis=1).to_numpy()
    y = df["Class"]
    y = LabelEncoder().fit_transform(y)
    
    if normalize.lower() == "identity":
        x = x
    elif normalize.lower() == "minmax":
        x = (x - np.min(x, axis=0, keepdims=True))/(np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))
        
    elif normalize.lower() == "standardize":
        mean_x = np.mean(x, axis=0)
        variance = np.sum((x - mean_x[np.newaxis, :])**2, axis=0)/(x.shape[0]-1)
        std = np.sqrt(variance)
        x = (x - mean_x[np.newaxis, :]) / std[np.newaxis, :]
    else:
        raise ValueError("No normalizing initializer name {}".format(normalize))
    
    if chosen_features is None:
        pass
    elif chosen_features.startswith("pca"):
        U, x = pca(x=x, alpha=0.95)
    else:
        x = x[:, list(map(lambda x: int(x.strip()[1:])-1, chosen_features.split("+")))]
        
    if use_bias:
        ones = np.ones(shape=[x.shape[0], 1], dtype=np.float32)
        x = np.append(x, ones, axis=1)
     
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=split_ratio, stratify=y, random_state=random_state)
    
    return x_train, y_train, x_val, y_val


class AverageMeter(object):

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def construct_lrscheduler(lrscheduler_type: str, init_lr: float) -> LRScheduler:
    if lrscheduler_type == "fixed":
        return LRFixedScheduler(init_lr=init_lr)
    elif lrscheduler_type == "inverse_decay":
        return InverseDecayScheduler(init_lr=init_lr)
    elif lrscheduler_type == "cyclic":
        return CyclicScheduler(max_lr=init_lr)
    elif lrscheduler_type == "linear_decay":
        return LinearLRDecay(init_lr=init_lr)
    else:
        raise ValueError("{} is not supported scheduler type".format(lrscheduler_type))

    
def construct_optimizer(optimizer_type: str, lrscheduler: LRScheduler) -> Optimizer:
    if optimizer_type == "gradient_descent":
        return GradientDescentOptimizer(lrscheduler=lrscheduler)
    elif optimizer_type == "adam":
        return AdamOptimizer(lrscheduler=lrscheduler, beta_1=0.5, beta_2=0.99)
    elif optimizer_type == "avagrad":
        return AvagradOptimizer(lrscheduler=lrscheduler, beta_1=0.5, beta_2=0.9)
    elif optimizer_type == "radam":
        return RadamOptimizer(lrscheduler=lrscheduler, beta_1=0.5, beta_2=0.99)
    elif optimizer_type == "adamax":
        return AdamaxOptimizer(lrscheduler=lrscheduler, beta_1=0.5, beta_2=0.9)
    elif optimizer_type == "nadam":
        return NadamOptimizer(lrscheduler=lrscheduler, beta_1=0.5, beta_2=0.99)
    elif optimizer_type == "amsgrad":
        return AMSGradOptimizer(lrscheduler=lrscheduler, beta_1=0.5, beta_2=0.99)
    elif optimizer_type == "adabelief":
        return AdaBeliefOptimizer(lrscheduler=lrscheduler, beta_1=0.5, beta_2=0.99)
    elif optimizer_type == "adagrad":
        return AdagradOptimizer(lrscheduler=lrscheduler)
    elif optimizer_type == "rmsprop":
        return RMSPropOptimizer(lrscheduler=lrscheduler, beta_2=0.90)
    elif optimizer_type == "momentum":
        return MomentumOptimizer(lrscheduler=lrscheduler, beta_1=0.9)
    elif optimizer_type == "adadelta":
        return AdadeltaOptimizer(lrscheduler=lrscheduler, beta_2=0.99)
    else:
        raise ValueError("{} is not supported optimizer type".format(optimizer_type))
    