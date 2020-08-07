from tensorflow.keras.constraints import *
from tensorflow.python.keras.layers.merge import multiply
import tensorflow.keras.backend as K


class ZeroSomeWeights(Constraint):
    def __init__(self,binary_tensor=None):
        self.binary_tensor = binary_tensor

    def __call__(self,w):
        if self.binary_tensor is not None:
            v = w * self.binary_tensor
        return v
    
    def get_config(self):
        return {'binary_tensor': self.binary_tensor}

zero_some_weights = ZeroSomeWeights