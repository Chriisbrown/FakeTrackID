import sys
import eval_funcs
from sklearn import metrics
import xgboost as xgb
import numpy as np
import Cut
import datasaver
import os

os.system("mkdir plots")


threshold = 0.5

from constraints import ZeroSomeWeights
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"ZeroSomeWeights": ZeroSomeWeights})

model1 = tf.keras.models.load_model("Models/NNpredlayersv1.h5")
parameters1 = ["LogChi","LogBendChi", "LogChirphi", "LogChirz" , "pred_nstubs",
               "pred_layer1","pred_layer2","pred_layer3","pred_layer4","pred_layer5",
               "pred_layer6","pred_disk1","pred_disk2","pred_disk3","pred_disk4",
               "pred_disk5","BigInvR","TanL","ModZ","pred_dtot","pred_ltot"]


'''
import joblib

model1 = joblib.load("Models/Classifier.pkl")
parameters1 = ["LogChi","LogBendChi", "LogChirphi", "LogChirz" , "pred_nstubs",
               "pred_layer1","pred_layer2","pred_layer3","pred_layer4","pred_layer5",
               "pred_layer6","pred_disk1","pred_disk2","pred_disk3","pred_disk4",
               "pred_disk5","BigInvR","TanL","ModZ","pred_dtot","pred_ltot"]

'''

try:
    trackdf = datasaver.load_transformed_data("../Data/hybridQuality1k")
    #trackdf = trackdf.sample(frac=1).reset_index(drop=True)
except KeyError:
    trackdf,events = datasaver.loadData("../Data/hybridQuality1k",transform=True)
    infs = np.where(np.asanyarray(np.isnan(trackdf)))[0]
    trackdf.drop(infs,inplace=True)
    trackdf.reset_index(inplace=True)
    trackdf = datasaver.predhitpattern(trackdf)
    datasaver.save_transformed_data(trackdf,"../Data/hybridQuality1k")




simple_cut = Cut.cut_based_classifier(2,2.4,15,40,2.4,4)
y_simple_cut = simple_cut.predict(trackdf)

trackdf["class_2"] = y_simple_cut
trackdf["class_output_2"] = y_simple_cut


y = trackdf["trk_fake"].to_numpy()
y[y>0] = 1
trackdf["trk_fake"] = y
X = trackdf[parameters1].to_numpy()

#dtest = xgb.DMatrix(X, label=y)
#trackdf["class_output_0"] = model1.predict_proba(X)[:,1]
trackdf["class_output_0"] = model1.predict(X)
trackdf["class_0"] = trackdf["class_output_0"]
trackdf["class_0"][trackdf["class_0"]>threshold] = 1
trackdf["class_0"][trackdf["class_0"]<=threshold] = 0









#X = trackdf[parameters2].to_numpy()
#dtest = xgb.DMatrix(X, label=y)
#trackdf["class_output_2"] = model2.predict_proba(X)[:,1]
#trackdf["class_output_1"] = model2.predict(X)
trackdf["class_output_1"] = trackdf["trk_MVA1"] 
trackdf["class_1"] = trackdf["class_output_1"]
trackdf["class_1"][trackdf["class_1"]>threshold] = 1
trackdf["class_1"][trackdf["class_1"]<=threshold] = 0

model_names = ("Keras_NN","CMSSW_NN","Chi2 Cut")

eval_funcs.roc_auc(trackdf,model_names)

eval_funcs.bins(trackdf,model_names,"eta")
#eval_funcs.bins(trackdf,model_names,"pt")
#eval_funcs.bins(trackdf,model_names,"phi")

#eval_funcs.own_roc(trackdf,model_names)

#eval_funcs.lepton_split(trackdf,model_names)






