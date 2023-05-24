import os
import math

import argparse

import itertools

import pickle

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from logistic import Logistic

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, GroupKFold, RepeatedKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve, precision_score, recall_score, classification_report

from utils import *

font = {"size": 8}

matplotlib.rc("font", **font)

def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--custom_model_path", type=str, default="models/config.pkl")
    parser.add_argument("--sklearn_model_path", type=str, default="sklearn_models/sklearn_config.pkl")
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    args = get_args()
    
    custom_model_path = args.custom_model_path
    sklearn_model_path = args.sklearn_model_path
    csv_path = "data/Sonar.csv"
    #chosen_features = None
    
    with open(custom_model_path, "rb") as f:
        custom_models_container = pickle.load(f)
        
    with open(sklearn_model_path, "rb") as f:
        sklearn_models_container = pickle.load(f)
    
    #normalize = custom_models_container["config"]["l2"][None]["normalize"]
    #
    #print(normalize)
    
        
    #train_x, train_y, val_x, val_y = create_data(csv_path=csv_path, chosen_features=chosen_features, normalize=normalize)
        
    #print(custom_models_container["model"]["l2"][None].compute_auc_roc(x=val_x, y=val_y), custom_models_container["auc"]["l2"][None])
    
    regularizers_list = [None, "l1", "l2"]
    
    
    models_feature_chosen = [None, 
                            "V2 + V4 + V10 + V11 + V17 + V20 + V21 + V22 + V29 + V30 + V31 + V32 + V34 + V35 + V36 + V49 + V50",
                            "V11 + V47 + V36 + V45 + V4 + V15 + V21 + V51 + V8 + V49 + V50 + V1 + V3 + V52 + V54 + V23 + V29 + V31 + V12 + V30 + V32 + V53 + V7 + V16 + V9 + V26 + V37 + V34 + V35 + V38 + V6 + V40 + V59 + V19 + V56",
                            "V1 + V3 + V7 + V9 + V12 + V13 + V14 + V16 + V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V27 + V29 + V30 + V31 + V32 + V33 + V34 + V37 + V39 + V40 + V41 + V42 + V43 + V46 + V48 + V50 + V51 + V52 + V57 + V58",
                            "pca_95"]
    
    names_model = {None: "full",
                   "V2 + V4 + V10 + V11 + V17 + V20 + V21 + V22 + V29 + V30 + V31 + V32 + V34 + V35 + V36 + V49 + V50": "m.bo",
                   "V1 + V3 + V7 + V9 + V12 + V13 + V14 + V16 + V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V27 + V29 + V30 + V31 + V32 + V33 + V34 + V37 + V39 + V40 + V41 + V42 + V43 + V46 + V48 + V50 + V51 + V52 + V57 + V58": "m.b",
                   "V11 + V47 + V36 + V45 + V4 + V15 + V21 + V51 + V8 + V49 + V50 + V1 + V3 + V52 + V54 + V23 + V29 + V31 + V12 + V30 + V32 + V53 + V7 + V16 + V9 + V26 + V37 + V34 + V35 + V38 + V6 + V40 + V59 + V19 + V56": "m.f",
                   "pca_95": "pca_95"}
    
    names_regu = {None: "",
                  "l1": "Lasso",
                  "l2": "Ridge"}
    
    plt.figure(figsize=(10, 10))
    plt.title("Receiver Operating Characteristic")
    for regularizer in regularizers_list:
        for chosen_features in models_feature_chosen:
            normalize = custom_models_container["config"][regularizer][chosen_features]["normalize"]
            model = custom_models_container["model"][regularizer][chosen_features]
            train_x, train_y, val_x, val_y = create_data(csv_path=csv_path, chosen_features=chosen_features, normalize=normalize)
            print("regularizer: {}, features: {}, {:.5f}".format(regularizer, chosen_features, np.abs(model.compute_auc_roc(x=val_x, y=val_y) - custom_models_container["auc"][regularizer][chosen_features])))
            y_pred = model.predict(x=val_x, return_prob=True)
            fpr, tpr, thresholds = roc_curve(y_true=val_y, y_score=y_pred)
            area_roc, aic, bic = model.compute_metrics(x=val_x, y=val_y)
            area_roc = auc(fpr, tpr)
            aic = compute_aic(y_true=val_y, y_pred_prob=y_pred, p=train_x.shape[1] + 1)
            bic = compute_aic(y_true=val_y, y_pred_prob=y_pred, p=train_x.shape[1] + 1, sample_size=val_y.shape[0])
            gmeans = compute_gmeans(tpr=tpr, fpr=fpr)
            
            idx = np.argmax(gmeans)
            prob_star = thresholds[idx]
            gmean_star = gmeans[idx]
            
            #print("regularizer: {}, features: {} tpr: {}, fpr: {}, thres: {}, gmeans: {}, ind: {}".format(regularizer, chosen_features, tpr, fpr, thresholds, gmeans, idx))
            
            plt.plot(fpr, tpr, label = "{} - {}, AUC={:.2f}, p*={:.2f}, gmean*={:.2f}, AIC={:.2f}, BIC={:.2f}".format(names_model[chosen_features], names_regu[regularizer], area_roc, prob_star, gmean_star, aic, bic))
            #plt.scatter(fpr[idx], tpr[idx], marker="o", color="red", label="Best")
    
    regularizers_list = ["none", "l1", "l2"]
    names_regu = {"none": "",
                  "l1": "Lasso",
                  "l2": "Ridge"}
    models_feature_chosen = [None, 
                            "V2 + V4 + V10 + V11 + V17 + V20 + V21 + V22 + V29 + V30 + V31 + V32 + V34 + V35 + V36 + V49 + V50",
                            "pca_95"]
    for regularizer in regularizers_list:
        for chosen_features in models_feature_chosen:
            normalize = sklearn_models_container["config"][regularizer][chosen_features]["normalize"]
            clf = sklearn_models_container["model"][regularizer][chosen_features]
            train_x, train_y, val_x, val_y = create_data(csv_path=csv_path, chosen_features=chosen_features, normalize=normalize)
            y_pred_prob = clf.predict_proba(val_x)[:, 1]
            fpr, tpr, thresholds = roc_curve(val_y, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            print("regularizer: {}, features: {}, {:.5f}".format(regularizer, chosen_features, np.abs(roc_auc - sklearn_models_container["auc"][regularizer][chosen_features])))
            aic = compute_aic(val_y, y_pred_prob, val_x.shape[1] + 1, 0)
            bic = compute_aic(val_y, y_pred_prob, val_x.shape[1] + 1, val_x.shape[0])
            
            gmeans = compute_gmeans(tpr=tpr, fpr=fpr)
            idx = np.argmax(gmeans)
            prob_star = thresholds[idx]
            gmean_star = gmeans[idx]
            
            #print("regularizer: {}, features: {} tpr: {}, fpr: {}, thres: {}, gmeans: {}, ind: {}".format(regularizer, chosen_features, tpr, fpr, thresholds, gmeans, idx))
            
            plt.plot(fpr, tpr, label = "sklearn_{} - {}, AUC={:.2f}, p*={:.2f}, gmean*={:.2f}, AIC={:.2f}, BIC={:.2f}".format(names_model[chosen_features], names_regu[regularizer], area_roc, prob_star, gmean_star, aic, bic))
    plt.legend(loc = "lower right")
    plt.plot([0, 1], [0, 1],"r--")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.grid()
    plt.show()
            
            
    
    