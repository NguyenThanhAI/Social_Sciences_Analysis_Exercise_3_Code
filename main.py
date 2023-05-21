import os
import argparse
from tqdm import tqdm

from logistic import Logistic

from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--csv_path", type=str, default="data/Sonar.csv")
    parser.add_argument("--chosen_features", type=str, default="pca")
    
    args = parser.parse_args()

    return args    


if __name__ == "__main__":
    
    #np.random.seed(42)
    
    args = get_args()
    
    csv_path = args.csv_path
    chosen_features = args.chosen_features
    N = 1000
    
    train_x, train_y, val_x, val_y = create_data(csv_path=csv_path, chosen_features=chosen_features)
    
    #for i in tqdm(range(N)):
    logistic = Logistic(optimizer_type="adam", 
                        lrscheduler_type="fixed",
                        seed=42,
                        init_lr=1e-4, # l1 thường lr phải lớn 1 chút
                        max_iter=100000,
                        num_steps_per_eval=5000,
                        regularizer="l2", # l1 thì hội tụ rất nhanh nên số max_iter chỉ nhỏ
                        weight_decay=1e-3,
                        verbosity=True)
    
    logistic.fit(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)