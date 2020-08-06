from comet_ml import Optimizer
import yaml
import models
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import Input
from callbacks import all_callbacks
from tensorflow.keras import callbacks
from train import get_features
from sklearn.metrics import roc_auc_score
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

parameters = open("parameters.yml")
yamlparameters = yaml.load(parameters,Loader=yaml.FullLoader)

config = {
    # We pick the Bayes algorithm:
    "algorithm": "bayes",

    # Declare your hyperparameters in the Vizier-inspired format:
    "parameters": {


        "learning_rate": {     "type": "float", "mu": yamlparameters["Training_learning_rate"], "sigma": 0.0001, "scalingType": "normal"},
        "Regularization": {    "type": "float", "min": 0.0001, "max": 0.01,  "scalingType": "uniform"},
        "pruning_begin_epoch":{"type": "int",   "min": 50,     "max": 200,   "scalingType": "uniform"},
        "pruning_end_epoch":  {"type": "int",   "min": 100,    "max": 800,   "scalingType": "uniform"},
        "pruning_lr_factor_1":{"type": "float", "min": 0.0,    "max": 1.0,   "scalingType": "uniform"},
        "pruning_lr_factor_2":{"type": "float", "min": 0.0,    "max": 1.0,   "scalingType": "uniform"},
        "pruning_lr_factor_3":{"type": "float", "min":-10.0,   "max": 10.0,  "scalingType": "uniform"},


    },

    # Declare what we will be optimizing, and how:
    "spec": {
    "metric": "ROC",
        "objective": "maximize",
    },
}




opt = Optimizer(config, api_key=yamlparameters["comet_api_key"], project_name="NNqhmv6",auto_metric_logging=True)

X_train, X_test, y_train, y_test  = get_features(yamlparameters["DataDir"])
    

for experiment in opt.get_experiments():
    steps_per_epoch = int(len(X_train)/yamlparameters["Training_batch_size"])

    pruning_params = {"pruning_schedule" : pruning_schedule.PolynomialDecay(initial_sparsity=0.0,
                                                                            final_sparsity=yamlparameters["Sparsity"],
                                                                            begin_step=experiment.get_parameter("pruning_begin_epoch")*steps_per_epoch, 
                                                                            end_step=experiment.get_parameter("pruning_end_epoch")*steps_per_epoch)}
    keras_model = models.qdense_model(Input(shape=X_train.shape[1:]), 
                                       l1Reg=experiment.get_parameter("Regularization"),
                                       bits=yamlparameters["Layer_bits"],
                                       ints=yamlparameters["Layer_ints"])
    keras_model = prune.prune_low_magnitude(keras_model, **pruning_params)

    startlearningrate=experiment.get_parameter("learning_rate")

    adam = Adam(lr=startlearningrate,
                beta_1=yamlparameters["Training_learning_beta1"],
                beta_2=yamlparameters["Training_learning_beta2"],
                amsgrad=True)

    keras_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

    callbacks=all_callbacks(stop_patience=yamlparameters["Training_early_stopping"], 
                            initial_lr=experiment.get_parameter("learning_rate"),
                            lr_factor=yamlparameters["Training_lr_factor"],
                            lr_patience=yamlparameters["Training_lr_patience"],
                            lr_epsilon=yamlparameters["Training_lr_min_delta"], 
                            lr_cooldown=yamlparameters["Training_lr_cooldown"], 
                            lr_minimum=yamlparameters["Training_lr_minimum"],
                            Prune_begin=experiment.get_parameter("pruning_begin_epoch"),
                            Prune_end=experiment.get_parameter("pruning_end_epoch"),
                            prune_lrs=[experiment.get_parameter("pruning_lr_factor_1"),
                                       experiment.get_parameter("pruning_lr_factor_2"),
                                       experiment.get_parameter("pruning_lr_factor_3")],
                            outputDir=yamlparameters["TrainDir"])

    callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())

    keras_model.fit(X_train, y_train, 
                    batch_size = experiment.get_parameter("batch_size"), 
                    epochs =  experiment.get_parameter("epochs"),
                    validation_split =  yamlparameters["Training_validation_split"], 
                    shuffle = True, 
                    callbacks =callbacks.callbacks,
                    verbose=0)
    
    model = strip_pruning(keras_model)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

    y_predict = model.predict(X_test,verbose=0)
    loss,binary_accuracy = keras_model.evaluate(X_test, y_test,verbose=0)
    auc = roc_auc_score(y_test,y_predict)
    print("AUC:",auc)
    print("ACC:",binary_accuracy)

    experiment.log_metric("ROC",auc)
    experiment.log_metric("Loss",loss)
    experiment.log_metric("Binary_Accuracy",binary_accuracy)

