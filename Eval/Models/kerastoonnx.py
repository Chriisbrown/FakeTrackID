'''
Helper script that converts a keras model saved in the h5 format to the onnx format
It will produce a file of name NN_model.onnx
'''

import numpy as np
import keras2onnx
import onnxruntime

import tensorflow as tf
from keras import backend as K

from tensorflow.keras.models import load_model
import numpy as np
import sys

from constraints import ZeroSomeWeights
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"ZeroSomeWeights": ZeroSomeWeights})

model = load_model(sys.argv[1])
X = np.array(np.random.rand(10, 21), dtype=np.float32)
print(model.predict(X))

# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

temp_model_file = 'NN_model.onnx'
keras2onnx.save_model(onnx_model, temp_model_file)
sess = onnxruntime.InferenceSession(temp_model_file)

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# This input name is needed in Classifier_cff as  NNIdONNXInputName
print(sess.get_inputs()[0].name)
print(label_name)
# The name of the output is needed in Clasifier_cff as NNIdONNXOutputName

# predict on random input and compare to previous keras model
for i in range(len(X)):
    pred_onx = sess.run([label_name], {input_name: X[i:i+1]})[0]
    print(pred_onx)

