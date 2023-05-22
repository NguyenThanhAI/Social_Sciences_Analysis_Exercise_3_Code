import os
import argparse
from tqdm import tqdm

from logistic import Logistic

from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--csv_path", type=str, default="data/Sonar.csv")
    # "V2 + V4 + V10 + V11 + V17 + V20 + V21 + V22 + V29 + V30 + V31 + V32 + V34 + V35 + V36 + V49 + V50"
    # "V11 + V47 + V36 + V45 + V4 + V15 + V21 + V51 + V8 + V49 + V50 + V1 + V3 + V52 + V54 + V23 + V29 + V31 + V12 + V30 + V32 + V53 + V7 + V16 + V9 + V26 + V37 + V34 + V35 + V38 + V6 + V40 + V59 + V19 + V56"
    # "V1 + V3 + V7 + V9 + V12 + V13 + V14 + V16 + V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V27 + V29 + V30 + V31 + V32 + V33 + V34 + V37 + V39 + V40 + V41 + V42 + V43 + V46 + V48 + V50 + V51 + V52 + V57 + V58"
    # "V11 + V47 + V36 + V45 + V4 + V15 + V21 +  V51 + V8 + V49 + V50 + V1 + V3 + V52 + V54 + V23 + V29 +  V31 + V12 + V30 + V32 + V7 + V16 + V9 + V26 + V37 + V35"
    parser.add_argument("--chosen_features", type=str, default="V11 + V47 + V36 + V45 + V4 + V15 + V21 +  V51 + V8 + V49 + V50 + V1 + V3 + V52 + V54 + V23 + V29 +  V31 + V12 + V30 + V32 + V7 + V16 + V9 + V26 + V37 + V35")
    parser.add_argument("--normalize", type=str, default="minmax")
    
    args = parser.parse_args()

    return args    


if __name__ == "__main__":
    
    #np.random.seed(42)
    
    args = get_args()
    
    csv_path = args.csv_path
    chosen_features = args.chosen_features
    normalize = args.normalize
    N = 1000
    
    train_x, train_y, val_x, val_y = create_data(csv_path=csv_path, chosen_features=chosen_features, normalize=normalize)
    
    lr_scheduler_list = ["fixed", "cyclic", "inverse_decay", "exponential_decay", "factor_decay", "squareroot", "cosine"]

    learning_rate_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5, 10]

    optimizer_list = ["Gradient_Descent", "Adam", "Avagrad", "RAdam", "Momentum", 
                      "Adagrad", "RMSProp", "Adadelta", "Adamax", "Nadam", "AMSGrad", "AdaBelief"]
    
    max_config = {"optimizer": None, "scheduler": None, "init_lr": None}
    max_val_auc = 0
    #for i in tqdm(range(N)):
    for optimizer in optimizer_list:
        for scheduler in lr_scheduler_list:
            for init_lr in learning_rate_list:
                print("Optimizer: {}, scheduler: {}, init_lr: {}".format(optimizer, scheduler, init_lr))
                
                logistic = Logistic(optimizer_type=optimizer, 
                                    lrscheduler_type=scheduler,
                                    seed=42,
                                    init_lr=init_lr, # l1 thường lr phải lớn 1 chút
                                    max_iter=10000,
                                    num_steps_per_eval=1000,
                                    regularizer=None, # l1 thì hội tụ rất nhanh nên số max_iter chỉ nhỏ
                                    weight_decay=1e-3,
                                    verbosity=False)

                logistic.fit(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
                
                if logistic.max_auc > max_val_auc:
                    max_config["optimizer"] = optimizer
                    max_config["scheduler"] = scheduler
                    max_config["init_lr"] = init_lr
                    max_val_auc = logistic.max_auc
                    
    print("Max config: {}, max AUC-ROC: {}".format(max_config, max_val_auc))