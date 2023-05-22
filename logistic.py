import copy
from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_curve, auc

from utils import *


class Logistic(object):
    def __init__(self, optimizer_type: str, lrscheduler_type: str, seed: int, regularizer: str=None, init_lr: float=1e-3, weight_decay: float=1e-4, max_iter: int=10000, w_tol: float=1e-4, grad_tol: float=1e-4, limit_tol: int=5, use_bias: bool=True, initializer: str="xavier", num_steps_per_eval: int=100, verbosity: bool=True) -> None:
        lrscheduler = construct_lrscheduler(lrscheduler_type=lrscheduler_type, init_lr=init_lr)
        self.optimizer = construct_optimizer(optimizer_type=optimizer_type, lrscheduler=lrscheduler)
        
        self.seed = seed
        np.random.seed(self.seed)
        self.init_lr = init_lr
        self.regularizer = regularizer
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.w_tol = w_tol
        self.grad_tol = grad_tol
        self.limit_tol = limit_tol
        self.use_bias = use_bias
        self.initializer = initializer
        
        self.weights = None
        self.prev_weights = None
        
        self.num_steps_per_eval = num_steps_per_eval
        self.verbosity = verbosity
        
        self.train_loss_meter = AverageMeter(name="train_loss", fmt=":.4f")
        self.val_loss_meter = AverageMeter(name="val_loss", fmt=":.4f")
        
        self.train_loss = {}
        self.val_loss = {}
        self.train_auc = {}
        self.val_auc = {}
        self.train_aic = {}
        self.val_aic = {}
        self.train_bic = {}
        self.val_bic = {}
        
        self.max_auc = 0
        self.max_weights: np.ndarray = None
        
        
    def init_weights(self, x: np.ndarray) -> None:

        if self.use_bias:
            if self.initializer == "xavier":
                self.weights = np.random.normal(loc=0, scale=np.sqrt(2/(x.shape[1])), size=x.shape[1]-1)
            else:
                self.weights = np.random.rand(x.shape[1]-1)

            self.weights = np.append(self.weights, 0.)

        else:
            if self.initializer == "xavier":
                weights = np.random.normal(loc=0, scale=np.sqrt(2/(x.shape[1])), size=x.shape[1])
            else:
                weights = np.random.rand(x.shape[1])

        self.prev_weights = np.zeros_like(self.weights)
    
    
    @staticmethod
    def sigmoid(xw: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-xw))
    
    
    def predict(self, x: np.ndarray,threshold: float=0.8, return_prob: bool=False):
        if len(x.shape) > 1:
            assert x.shape[1] == self.weights.shape[0]
            xw = np.sum(x * self.weights[np.newaxis, :], axis=1)
        else:
            assert x.shape[0] == self.weights.shape[0]
            xw = np.sum(x * self.weights)

        prob = self.sigmoid(xw=xw)
        if return_prob:
            return prob
        #print("prob: {}".format(prob))
        else:
            output = np.where(prob > threshold, 1, 0)
            return output
        
    
    def sigmoid_cross_entropy_with_logits(self, xw: np.ndarray, y: np.ndarray) -> float:
        a = self.sigmoid(xw=xw)
        cost = - np.mean(y * np.log(a + 1e-8) + (1 - y) * np.log(1-a + 1e-8))
        return cost
    
    
    @staticmethod
    def sigmoid_cross_entropy_truncated(xw: np.ndarray, y: np.ndarray):
        return np.mean(- y * xw + np.log(1+np.exp(xw)))
    
    
    def sigmoid_cross_entropy_with_x_y(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x.shape) > 1:
            assert x.shape[1] == self.weights.shape[0]
            xw = np.sum(x * self.weights[np.newaxis, :], axis=1)
        else:
            assert x.shape[0] == self.weights.shape[0]
            xw = np.sum(x * self.weights)
        #return sigmoid_cross_entropy_with_logits(xw=xw, y=y)
        return self.sigmoid_cross_entropy_truncated(xw=xw, y=y)
    
    
    def derivative_cost_wrt_params(self, x: np.ndarray, y: np.ndarray):
        #assert x.shape[0] == y.shape[0]
        if len(x.shape) > 1:
            assert x.shape[1] == self.weights.shape[0]
            xw = np.sum(x * self.weights[np.newaxis, :], axis=1)
            sig = self.sigmoid(xw=xw)
            return np.mean(- y[:, np.newaxis] * x + sig[:, np.newaxis] * x, axis=0)
        else:
            assert x.shape[0] == self.weights.shape[0]
            xw = np.sum(x * self.weights)
            sig = sig(xw=xw)
            return - y[np.newaxis] * x + sig[np.newaxis] * x
        
    def get_total_cost(self, raw_cost: float) -> float:
        if self.regularizer == "l2":
            return raw_cost + self.weight_decay * np.sum(self.weights**2)
        elif self.regularizer == "l1":
            return raw_cost + self.weight_decay * np.sum(np.abs(self.weights))
        else:
            return raw_cost
        
    
    def get_final_grad(self, raw_grad: np.ndarray) -> np.ndarray:
        assert self.weights.shape[0] == raw_grad.shape[0]
        if self.regularizer == "l2":
            return raw_grad + 2 * self.weight_decay * self.weights
        else:
            return raw_grad
        
        
    def soft_threshold(self) -> float:
        return np.where(self.weights > self.weight_decay, self.weights - self.weight_decay, np.where(self.weights < - self.weight_decay, self.weights + self.weight_decay, 0))
        
        
    def step(self, x: np.ndarray, y: np.ndarray):
        dweights = self.derivative_cost_wrt_params(x=x, y=y)
        
        if self.regularizer == "l2":
            dweights = self.get_final_grad(raw_grad=dweights)
            
        self.weights = self.optimizer.step(weights=self.weights, dweights=dweights)
        
        if self.regularizer == "l1":
            self.weights = self.soft_threshold()
        
        diff_weights = np.sqrt(np.sum((self.weights - self.prev_weights)**2))
        grad_norm = np.sqrt(np.sum(dweights**2))
        
        self.prev_weights = copy.deepcopy(self.weights)
        
        return diff_weights, grad_norm
    
    
    def compute_auc_roc(self, x: np.ndarray, y: np.ndarray):
        y_pred = self.predict(x=x, return_prob=True)
        fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_pred)
        area_roc = auc(fpr, tpr)
        return area_roc
    

    def compute_aic(self, y_true: np.ndarray, y_pred_prob: np.ndarray, sample_size: int=0):
        p = self.weights.shape[0]
        sample_size = y_true.shape[0]
        log_likelihood_elements = y_true*np.log(y_pred_prob + 1e-8) + (1-y_true)*np.log(1-y_pred_prob + 1e-8)
        if sample_size > 0:
            return -2 * sum(log_likelihood_elements) + np.log(sample_size) * p
        else:
            return -2 * sum(log_likelihood_elements) + 2 * p
        
        
    def compute_metrics(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        y_pred = self.predict(x=x, return_prob=True)
        fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_pred)
        area_roc = auc(fpr, tpr)
        aic = self.compute_aic(y_true=y, y_pred_prob=y_pred)
        bic = self.compute_aic(y_true=y, y_pred_prob=y_pred, sample_size=y.shape[0])
        
        return area_roc, aic, bic
    
    
    def plot_loss(self):
        plt.title("Loss")
        plt.plot(self.train_loss.keys(), self.train_loss.values(), label="train loss")
        plt.plot(self.val_loss.keys(), self.val_loss.values(), label="val loss")
        plt.legend(loc ="lower right")
        plt.ylabel("Loss")
        plt.xlabel("Step")
        plt.grid()
        plt.show()
        
        
    def plot_roc(self, x: np.ndarray, y: np.ndarray):
        y_pred = self.predict(x=x, return_prob=True)
        fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_pred)
        plot_roc_curve(tpr=tpr, fpr=fpr)
        
    def restore_best_weights(self) -> None:
        if self.max_weights is not None:
            self.weights = self.max_weights
    
    
    def fit(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> None:
        
        self.init_weights(train_x)
        
        stop_tol = 0
        auc_tol = 0
        
        for i in tqdm(range(self.max_iter)):
            diff_weights, grad_norm = self.step(x=train_x, y=train_y)
            
            train_loss = self.sigmoid_cross_entropy_with_x_y(x=train_x, y=train_y)
            self.train_loss[i+1] = train_loss
            self.train_loss_meter.update(val=train_loss)
            
            val_loss = self.sigmoid_cross_entropy_with_x_y(x=val_x, y=val_y)
            self.val_loss[i+1] = val_loss
            self.val_loss_meter.update(val=val_loss)
            

            if (i + 1) % self.num_steps_per_eval == 0:
                try:
                    train_auc, train_aic, train_bic = self.compute_metrics(x=train_x, y=train_y)
                    self.train_auc[i+1] = train_auc
                    self.train_aic[i+1] = train_aic
                    self.train_bic[i+1] = train_bic
                    val_auc, val_aic, val_bic = self.compute_metrics(x=val_x, y=val_y)
                    self.val_auc[i+1] = val_auc
                    self.val_aic[i+1] = val_aic
                    self.val_bic[i+1] = val_bic
                    if val_auc > self.max_auc:
                        self.max_auc = val_auc
                        self.max_weights = copy.deepcopy(self.weights)
                        auc_tol = 0
                    else:
                        auc_tol += 1
                    if self.verbosity == True:
                        print("Step {}, train AUC: {:.3f}, train AIC: {:.3f}, train BIC: {:.3f}, val AUC: {:.3f}, val AIC: {:.3f}, val BIC: {:.3f}, diff weights: {:.10f}, grad norm: {:.3f}".format(i + 1, train_auc, train_aic, train_bic, val_auc, val_aic, val_bic, diff_weights, grad_norm))

                    if auc_tol >= self.limit_tol:
                        print("AUC-ROC does not improve. Stop training")
                        break
                except ValueError:
                    break
                
            if diff_weights <= self.w_tol and grad_norm <= self.grad_tol:
                stop_tol += 1
            else:
                stop_tol = 0
                
            if stop_tol >= self.limit_tol:
                break
            
        self.restore_best_weights()
        if self.verbosity:
            print("Model weights: {}".format(self.weights.tolist()))
        print("Max val auc: {}".format(self.max_auc))
        if self.verbosity:
            self.plot_loss()
            self.plot_roc(x=val_x, y=val_y)