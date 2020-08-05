from comet_ml import Optimizer
import yaml
import models
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import Input
from callbacks import all_callbacks
from tensorflow.keras import callbacks
from train import get_features
from sklearn.metrics import roc_auc_score

config = {
    # We pick the Bayes algorithm:
    "algorithm": "bayes",

    # Declare your hyperparameters in the Vizier-inspired format:
    "parameters": {
        "batch_size": {"type": "integer", "min": 151, "max": 252},
        "epochs": {"type": "integer", "min": 100, "max": 101},
        "learning_rate": {"type": "float", "min": 0.0001, "max": 0.001, "scalingType": "uniform"},
        "learning_beta1": {"type": "float", "min": 0.800, "max": 0.9999, "scalingType": "uniform"},
        "learning_beta2": {"type": "float", "min": 0.800, "max": 0.9999, "scalingType": "uniform"},
        "Regularization": {"type": "float", "min": 0.0001, "max": 0.01, "scalingType": "uniform"},
        "Adagrad":{"type":"categorical","values":["True","False"]},
        "Training_lr_factor": {"type": "float", "min": 0.49, "max": 0.5, "scalingType": "uniform"},
        "Training_lr_patience": {"type": "integer", "min": 10, "max": 11},
        #"Layer_bits":{"type":"integer","min":10,"max":16},
        #"Layer_ints":{"type":"integer","min":2,"max":6}

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
opt = Optimizer(config, api_key=yamlparameters["comet_api_key"], project_name="NNqhmv6",auto_metric_logging=True)

X_train, X_test, y_train, y_test  = get_features(yamlparameters["DataDir"])
    

for experiment in opt.get_experiments():
    keras_model = models.qdense_model(Input(shape=X_train.shape[1:]), l1Reg=experiment.get_parameter("Regularization"),bits=14,ints=2)
    #keras_model = models.dense_model(Input(shape=X_train.shape[1:]), l1Reg=experiment.get_parameter("Regularization"))
    startlearningrate=experiment.get_parameter("learning_rate")
    adam = Adam(lr=startlearningrate,beta_1=experiment.get_parameter("learning_beta1"),beta_2=experiment.get_parameter("learning_beta2"),amsgrad=experiment.get_parameter("Adagrad"))
    keras_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

    callbacks=all_callbacks(stop_patience=yamlparameters["Training_early_stopping"], 
                            lr_factor=experiment.get_parameter("Training_lr_factor"),
                            lr_patience=experiment.get_parameter("Training_lr_patience"),
                            lr_epsilon=yamlparameters["Training_lr_min_delta"], 
                            lr_cooldown=yamlparameters["Training_lr_cooldown"], 
                            lr_minimum=yamlparameters["Training_lr_minimum"],
                            outputDir="None")
    keras_model.fit(X_train, y_train, batch_size = experiment.get_parameter("batch_size"), epochs =  experiment.get_parameter("epochs"),
                        validation_split =  yamlparameters["Training_validation_split"], shuffle = True, callbacks =callbacks.callbacks,verbose=1)

    y_predict = keras_model.predict(X_test,verbose=0)
    loss,binary_accuracy = keras_model.evaluate(X_test, y_test,verbose=0)
    auc = roc_auc_score(y_test,y_predict)
    print("AUC:",auc)
    print("ACC:",binary_accuracy)
    experiment.log_metric("ROC",auc)
    experiment.log_metric("Loss",loss)
    experiment.log_metric("Binary_Accuracy",binary_accuracy)

