import yaml
import sys
import os

if sys.argv[1] == "NN":

    parameters = open("parameters.yml")
    yamlparameters = yaml.load(parameters,Loader=yaml.FullLoader)

    os.system("mkdir "+ yamlparameters["TrainDir"])


    os.system("python train.py")


    os.system("mkdir NNOutput")
    os.system("mkdir NNOutput/Models")
    os.system("mkdir NNOutput/Images")

    os.system("mv "+ yamlparameters["TrainDir"]+"/KERAS_check_best_model_weights.h5 NNOutput/Models/Initial_model_weights.h5")
    os.system("mv "+ yamlparameters["TrainDir"]+"/KERAS_check_best_model.h5 NNOutput/Models/Initial_model.h5")
    os.system("mv "+ yamlparameters["TrainDir"]+"/KERAS_model.json NNOutput/Models/Initial_model.json")

    os.system("mv "+ yamlparameters["TrainDir"]+"/Best_model.h5 NNOutput/Models/Final_model.h5")

    os.system("mv "+yamlparameters["Network_Name"]+".png NNOutput/Images/FinalResult.png")
    os.system("cp parameters.yml NNOutput/parameters.yml")
    os.system("rm -r " +yamlparameters["TrainDir"])

if sys.argv[1] == "NNOptimiser":
    os.system("python optimiser.py")

if sys.argv[1] == "GBDT":

    parameters = open("GBDTparameters.yml")
    yamlparameters = yaml.load(parameters,Loader=yaml.FullLoader)

    os.system("mkdir "+ yamlparameters["TrainDir"])


    os.system("python GBDT.py")


    os.system("mkdir GBDTOutput")
    os.system("mkdir GBDTOutput/Models")
    os.system("mkdir GBDTOutput/Images")

    os.system("mv "+ yamlparameters["TrainDir"]+"/Classifier.pkl GBDTOutput/Models/GBDT.pkl")
    os.system("mv "+ yamlparameters["xTree_name"]+".png GBDTOutput/FinalResult.png")
    os.system("cp GBDTparameters.yml GBDTOutput/parameters.yml")
    os.system("rm -r " +yamlparameters["TrainDir"])

if sys.argv[1] == "GBDTOptimiser":
    os.system("python GBDToptimiser.py")