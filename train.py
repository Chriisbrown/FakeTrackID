from comet_ml import Experiment
import sys
import os
import keras
import numpy as np
# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
import h5py
from tensorflow.keras.optimizers import Adam, Nadam
import pandas as pd
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
import yaml
import models
import util_funcs
from callbacks import all_callbacks
from tensorflow.keras import callbacks
import gc
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from sklearn.metrics import roc_auc_score


# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

def print_model_to_json(keras_model, outfile_name):
    outfile = open(outfile_name,'w')
    jsonString = keras_model.to_json()
    import json
    with outfile:
        obj = json.loads(jsonString)
        json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
        outfile.write('\n')

def get_features(datafolder):
    # To use one data file:
    trackdf = util_funcs.load_transformed_data(datafolder)

    trackdf = trackdf.sample(frac=1).reset_index(drop=True)

    X = trackdf[["trk_matchtp_pdgid","LogChi","LogBendChi","LogChirphi","LogChirz", "trk_nstub",
                "pred_layer1","pred_layer2","pred_layer3","pred_layer4","pred_layer5","pred_layer6","pred_disk1","pred_disk2","pred_disk3","pred_disk4","pred_disk5","BigInvR","TanL","ModZ","pred_dtot","pred_ltot"]]
    y = trackdf["trk_fake"]
    
    del [trackdf]


    print("Data Imported .....")

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=seed)


    Z = X_train.join(y_train)
    Z = util_funcs.particleBalanceData(Z,seed)
    
    X_train = Z[["LogChi","LogBendChi","LogChirphi","LogChirz", "trk_nstub",
                "pred_layer1","pred_layer2","pred_layer3","pred_layer4","pred_layer5","pred_layer6","pred_disk1","pred_disk2","pred_disk3","pred_disk4","pred_disk5","BigInvR","TanL","ModZ","pred_dtot","pred_ltot"]].to_numpy()
    y_train = Z["trk_fake"].to_numpy()
    y_train[y_train > 0] = 1

    
    X_test = X_test[["LogChi","LogBendChi","LogChirphi","LogChirz", "trk_nstub",
                "pred_layer1","pred_layer2","pred_layer3","pred_layer4","pred_layer5","pred_layer6","pred_disk1","pred_disk2","pred_disk3","pred_disk4","pred_disk5","BigInvR","TanL","ModZ","pred_dtot","pred_ltot"]].to_numpy()

    y_test = y_test.to_numpy()
    y_test[y_test > 0] = 1

    del[X,Z]
    gc.collect()
        

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    

    parameters = open("parameters.yml")
    yamlparameters = yaml.load(parameters,Loader=yaml.FullLoader)
    experiment = Experiment(api_key=yamlparameters["comet_api_key"],project_name='qkeras',auto_param_logging=True)
    

    X_train, X_test, y_train, y_test  = get_features(yamlparameters["DataDir"])

    steps_per_epoch = int(len(X_train)/yamlparameters["Training_batch_size"])

    
    #pruning_params = {"pruning_schedule" : pruning_schedule.PolynomialDecay(initial_sparsity=0.0,
    #                                                                        final_sparsity=yamlparameters["Sparsity"],
    #                                                                        begin_step=yamlparameters["Pruning_begin_epoch"]*steps_per_epoch, 
    #                                                                        end_step=yamlparameters["Pruning_end_epoch"]*steps_per_epoch)}

    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(initial_sparsity=0.0,
                                                                             target_sparsity=yamlparameters["Sparsity"],
                                                                             begin_step=yamlparameters["Pruning_begin_epoch"]*steps_per_epoch, 
                                                                             end_step=yamlparameters["Pruning_end_epoch"]*steps_per_epoch,
                                                                             frequency=yamlparameters["Pruning_frequency"]*steps_per_epoch)}
    

    #keras_model = models.qdense_model(Input(shape=X_train.shape[1:]), 
                                       l1Reg=yamlparameters["Training_regularization"],
                                       bits=yamlparameters["Layer_bits"],
                                       ints=yamlparameters["Layer_ints"])
    #keras_model = prune.prune_low_magnitude(keras_model, **pruning_params)
    

    keras_model = models.dense_model_RegBN(Input(shape=X_train.shape[1:]), 
                                           l1Reg=yamlparameters["Training_regularization"])


    
    print(keras_model.summary())
    print_model_to_json(keras_model,yamlparameters["TrainDir"] +'/KERAS_model.json')

    startlearningrate=yamlparameters["Training_learning_rate"]
    
    adam = Adam(lr=startlearningrate,
                beta_1=yamlparameters["Training_learning_beta1"],
                beta_2=yamlparameters["Training_learning_beta2"],
                amsgrad=True)
    
    keras_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    callbacks=all_callbacks(stop_patience=yamlparameters["Training_early_stopping"], 
                            initial_lr=yamlparameters["Training_learning_rate"],
                            lr_factor=yamlparameters["Training_lr_factor"],
                            lr_patience=yamlparameters["Training_lr_patience"],
                            lr_epsilon=yamlparameters["Training_lr_min_delta"], 
                            lr_cooldown=yamlparameters["Training_lr_cooldown"], 
                            lr_minimum=yamlparameters["Training_lr_minimum"],
                            Prune_begin=yamlparameters["Pruning_begin_epoch"],
                            Prune_end=yamlparameters["Pruning_end_epoch"],
                            outputDir=yamlparameters["TrainDir"])

    callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())

    with experiment.train():
    
        keras_model.fit(X_train,y_train,
                        batch_size=yamlparameters["Training_batch_size"],
                        epochs=yamlparameters["Training_epochs"],
                        callbacks=callbacks.callbacks,
                        verbose=1,
                        validation_split=yamlparameters["Training_validation_split"],
                        shuffle=True)
 
    keras_model = strip_pruning(keras_model)
    keras_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    keras_model.save(yamlparameters["TrainDir"]+"/Best_model.h5")

    with experiment.test():
        y_predict = keras_model.predict(X_test,verbose=0)
        loss,binary_accuracy = keras_model.evaluate(X_test, y_test,verbose=0)
        auc = roc_auc_score(y_test,y_predict)
        print("AUC:",auc)
        print("ACC:",binary_accuracy)

        metrics = {
        'loss':loss,
        'accuracy':binary_accuracy,
        'ROC AUC':roc_auc_score
        }
        experiment.log_metrics(metrics)

    

    ####################################################################################################################################################
    print("Predictions Made .....")
    plot_parameters = {'score':binary_accuracy,'lengths':len(X_train),'events':len(X_train)/184,'name':yamlparameters["Network_Name"]}
    auc = util_funcs.calculate_roc(keras_model,X_test,y_test,0,plot_parameters,plot=True,NN=True)
    print("Results Printed .....")
    experiment.log_dataset_hash(X_train) #creates and logs a hash of your data



