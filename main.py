import os
import argparse

from logistic import Logistic

from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--csv_path", type=str, default="data/Sonar.csv")
    parser.add_argument("--chosen_features", type=str, default=None)
    
    args = parser.parse_args()

    return args    


if __name__ == "__main__":
    
    np.random.seed(42)
    
    args = get_args()
    
    csv_path = args.csv_path
    chosen_features = args.chosen_features
    
    train_x, train_y, val_x, val_y = create_data(csv_path=csv_path, chosen_features=chosen_features)
    
    logistic = Logistic(optimizer_type="gradient_descent", 
                        lrscheduler_type="fixed", 
                        init_lr=1e-4,
                        max_iter=100000,
                        num_steps_per_eval=10000)
    
    logistic.fit(train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)