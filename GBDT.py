import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
import os
import util_funcs
import yaml
import joblib
import gc
import util_funcs
import numpy as np
from sklearn import metrics
from train import get_features

def save_tree(model, outfile_name):
    joblib_file = outfile_name + ".pkl"
    joblib.dump(model,joblib_file)


if __name__ == "__main__":

    parameters = open("GBDTparameters.yml")
    yamlparameters = yaml.load(parameters,Loader=yaml.FullLoader)
    X_train,X_test,y_train,y_test = get_features(yamlparameters["DataDir"])
    '''
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_round = yamlparameters["Training_n_estimators"]
    

    params = {"max_depth":yamlparameters["Training_max_depth"],
                                "objective":'binary:logistic',
                                "tree_method":'exact',
                                "learning_rate":yamlparameters["Training_learning_rate"],
                                "gamma":yamlparameters["Training_gamma"],
                                "min_child_weight":yamlparameters["Training_min_child_weight"],
                                "subsample":yamlparameters["Training_subsample"],
                                "reg_alpha":yamlparameters["Training_alpha"],
                                #num_rounds=yamlparameters["Training_n_estimators"],
                                "n_jobs":4}

    model = xgb.train(params, dtrain, num_round)
    '''
    model = xgb.XGBClassifier(n_estimators = yamlparameters["Training_n_estimators"],
                              max_depth=yamlparameters["Training_max_depth"],
                              objective='binary:logistic',
                              tree_method='exact',
                              learning_rate=yamlparameters["Training_learning_rate"],
                              gamma=yamlparameters["Training_gamma"],
                              min_child_weight=yamlparameters["Training_min_child_weight"],
                              subsample=yamlparameters["Training_subsample"],
                              reg_alpha=yamlparameters["Training_alpha"],
                              n_jobs=4)
    model = model.fit(X_train,y_train)
    save_tree(model,yamlparameters["TrainDir"]+"/Classifier")

    #y_predict = model.predict(dtest)
    #y_predict_prob = model.predict(dtest)

    y_predict = model.predict(X_test)
    y_predict_prob = model.predict_proba(X_test)[:,1]

    print("ROC AUC:",metrics.roc_auc_score(y_test,y_predict_prob))
    binary_accuracy = metrics.accuracy_score(y_test,y_predict)
    print("ACC:",binary_accuracy)

    print("Predictions Made .....")
    plot_parameters = {'score':binary_accuracy,'lengths':len(X_train),'events':len(X_train)/184,'name':yamlparameters["xTree_name"]}
    auc = util_funcs.calculate_roc(model,X_test,y_test,0,plot_parameters,plot=True,NN=False)
    print("Results Printed .....")
