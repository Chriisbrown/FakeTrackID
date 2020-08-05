mport pandas as pd
from sklearn.model_selection import train_test_split
import xgboost
from sklearn import metrics
import time
import sys
import os
import util_funcs
import yaml
import joblib
import gc
from train import get_features

config = {
    # We pick the Bayes algorithm:
    "algorithm": "bayes",

    # Declare your hyperparameters in the Vizier-inspired format:
    "parameters": {
        "n_estimators": {"type": "integer", "min": 100, "max": 101},
        "max_depth": {"type": "integer", "min": 3, "max": 4},
        "learning_rate": {"type": "float", "min": 0.01, "max": 1.0, "scalingType": "uniform"},
        "gamma": {"type": "float", "min": 0.0, "max": 0.00001, "scalingType": "uniform"},
        "min_child_weight": {"type": "integer", "min": 1, "max": 10},
        "subsample": {"type": "float", "min": 0.1, "max": 1.0, "scalingType": "uniform"},
        "alpha": {"type": "float", "min": 0.0, "max": 1.0, "scalingType": "uniform"},


        #"activation":{"type": "categorical","values":["elu","sigmoid"]}
    },

    # Declare what we will be optimizing, and how:
    "spec": {
    "metric": "ROC",
        "objective": "maximize",
    },
}



parameters = open("parameters.yml")
yamlparameters = yaml.load(parameters,Loader=yaml.FullLoader)
opt = Optimizer(config, api_key=yamlparameters["comet_api_key"], project_name="newhitmaskGBDTclassifier",auto_metric_logging=True)

X_train, X_test, y_train, y_test  = get_features(yamlparameters["DataDir"])


for experiment in opt.get_experiments():
    model = xgboost.XGBClassifier(booster="gbtree",verbosity=1,objective="binary:logistic",tree_method="exact",
                                  learning_rate=experiment.get_parameter("learning_rate"),
                                  gamma=experiment.get_parameter("gamma"),
                                  n_estimators=experiment.get_parameter("n_estimators"),
                                  max_depth=experiment.get_parameter("max_depth"),
                                  min_child_weight=experiment.get_parameter("min_child_weight"),
                                  subsample=experiment.get_parameter("subsample"),
                                  alpha=experiment.get_parameter("alpha"),n_jobs=4)

    model.fit(X_train,y_train)

    y_predict = model.predict(X_test)

    binary_accuracy = metrics.balanced_accuracy_score(y_test,y_predict)
    y_predict_prob = model.predict_proba(X_test)[:,1]
    auc = metrics.roc_auc_score(y_test,y_predict_prob)
    print("AUC:",auc)
    print("ACC:",binary_accuracy)
    experiment.log_metric("ROC",auc)

    experiment.log_metric("Binary_Accuracy",binary_accuracy)
