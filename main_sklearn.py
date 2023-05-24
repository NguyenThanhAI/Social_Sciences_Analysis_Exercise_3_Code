import os
import argparse
from tqdm import tqdm
import pickle

import time
import copy
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, classification_report, auc

from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--csv_path", type=str, default="data/Sonar.csv")
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    
    args = get_args()
    
    csv_path = args.csv_path
    
    N = 1000
    
    regularizers_list = ["none", "l1", "l2"]
    
    models_feature_chosen = [None, 
                            "V2 + V4 + V10 + V11 + V17 + V20 + V21 + V22 + V29 + V30 + V31 + V32 + V34 + V35 + V36 + V49 + V50",
                            #"V11 + V47 + V36 + V45 + V4 + V15 + V21 + V51 + V8 + V49 + V50 + V1 + V3 + V52 + V54 + V23 + V29 + V31 + V12 + V30 + V32 + V53 + V7 + V16 + V9 + V26 + V37 + V34 + V35 + V38 + V6 + V40 + V59 + V19 + V56",
                            #"V1 + V3 + V7 + V9 + V12 + V13 + V14 + V16 + V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V27 + V29 + V30 + V31 + V32 + V33 + V34 + V37 + V39 + V40 + V41 + V42 + V43 + V46 + V48 + V50 + V51 + V52 + V57 + V58",
                            "pca_95"]
    
    normalize_strategy = ["minmax", "standardize"]
    
    l1_optimizers_list = ["liblinear", "saga"]
    general_optimizers_list = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    Cs_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    
    max_config = {}
    max_model = {}
    max_val_auc = {}
    
    
    for regularizer in regularizers_list:
        max_config[regularizer] = {}
        max_model[regularizer] = {}
        max_val_auc[regularizer] = {}
        
        if regularizer == "l1":
            optimizers_list = copy.deepcopy(l1_optimizers_list)
        else:
            optimizers_list = copy.deepcopy(general_optimizers_list)
        
        for chosen_features in models_feature_chosen:
            max_val_auc[regularizer][chosen_features] = 0
            max_config[regularizer][chosen_features] = {"state": None, "normalize": None, "optimizer": None, "C": None}
            max_model[regularizer][chosen_features] = None
            print("Regularizer: {}, Model: {}".format(regularizer, chosen_features))
            for normalize in normalize_strategy:
                train_x, train_y, val_x, val_y = create_data(csv_path=csv_path, chosen_features=chosen_features, normalize=normalize)
                for state in tqdm(range(N)):
                    #accu_time = 0
                    #start = time.time()
                     
                    for optimizer in optimizers_list:
                        #start = time.time()   
                        for C in Cs_list:
                            #print("Regularizer: {}, Model: {}, normalize: {}, state: {}, optimizer: {}, C: {}".format(regularizer, chosen_features, normalize, state, optimizer, C))
                            #start = time.time()
                            try:
                                
                                clf = LogisticRegression(penalty=regularizer, C=C, solver=optimizer, random_state=state)
                                clf.fit(train_x, train_y)
                                y_pred = clf.predict_proba(val_x)[:, 1]
                                auc_roc = roc_auc_score(val_y, y_pred)

                                if auc_roc > max_val_auc[regularizer][chosen_features]:
                                    max_config[regularizer][chosen_features]["normalize"] = normalize
                                    max_config[regularizer][chosen_features]["optimizer"] = optimizer
                                    max_config[regularizer][chosen_features]["state"] = state
                                    max_config[regularizer][chosen_features]["C"] = C
                                    max_val_auc[regularizer][chosen_features] = auc_roc
                                    max_model[regularizer][chosen_features] = clf
                                #end = time.time()
                                #accu_time += end - start
                            except ValueError:
                                continue
                            #end = time.time()
                            #accu_time += end - start
                        #end = time.time()
                        #accu_time += end - start
                    #end = time.time()
                    #accu_time = end - start
                    #print("Accumulated time: {}".format(accu_time))
                                
            print("Regularizers: {}, Model: {}, Max config: {}, max AUC-ROC: {}".format(regularizer, chosen_features, max_config[regularizer][chosen_features], max_val_auc[regularizer][chosen_features]))
            
        with open("sklearn_models/sklearn_config_{}.pkl".format(str(regularizer)), mode="wb") as f:
            pickle.dump({"config": max_config, "model": max_model, "auc": max_val_auc}, f)
            
    print("Max configuration: {}".format(max_config))
    with open("sklearn_models/sklearn_config.pkl", mode="wb") as f:
           pickle.dump({"config": max_config, "model": max_model, "auc": max_val_auc}, f)