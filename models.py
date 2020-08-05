from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l1
import h5py
import tensorflow_model_optimization as tfmot


def dense_model_regBN(Inputs,l1Reg=0,h5fName=None):
    x = tfmot.sparsity.keras.prune_low_magnitude(Dense(21,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              name="Dense_Layer_1"))(Inputs)

    x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_1",)(x)

    x = Activation(activation='relu', activity_regularizer = l1(l1Reg),name="Relu_Layer_1")(x)
    #x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_1",)(x)    
    x = tfmot.sparsity.keras.prune_low_magnitude(Dense(22,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              name="Pruned_Dense_Layer_2"))(x)

    x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_2",)(x)

    x = Activation(activation='relu', activity_regularizer = l1(l1Reg),name="Relu_Layer_2")(x)
    #x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_2",)(x)
    x = tfmot.sparsity.keras.prune_low_magnitude(Dense(8,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              name="Dense_Layer_3"))(x)

    x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_3",)(x)
    
    x = Activation(activation='relu', activity_regularizer = l1(l1Reg),name="Relu_Layer_3")(x)
    #x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_3",)(x)
    x = Dense(1,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              name="Dense_Layer_4")(x)

    x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_4",)(x)

    predictions = Activation(activation='sigmoid', activity_regularizer = l1(l1Reg),name="Sigmoid_Output_Layer")(x)
    #predictions = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_4",)(x)
    model = Model(inputs=Inputs, outputs=predictions)

    return(model) 
def dense_model(Inputs,l1Reg=0,h5fName=None):
    x = Dense(21,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              name="Dense_Layer_1")(Inputs)

    x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_1",)(x)

    x = Activation(activation='relu',name="Relu_Layer_1")(x)

    x = Dense(22,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              name="Dense_Layer_2")(x)

    x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_2",)(x)

    x = Activation(activation='relu',name="Relu_Layer_2")(x)

    x = Dense(8,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              name="Dense_Layer_3")(x)

    x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_3")(x)

    x = Activation(activation='relu',name="Relu_Layer_3")(x)

    x = Dense(1,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              name="Dense_Layer_4")(x)

    x = BatchNormalization(beta_regularizer=l1(l1Reg), gamma_regularizer=l1(l1Reg),name="BN_Layer_4")(x)

    predictions = Activation(activation='sigmoid', name="Sigmoid_Output_Layer")(x)

    model = Model(inputs=Inputs, outputs=predictions)

    return(model)


from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
def qdense_model(Inputs,l1Reg=0,bits=6,ints=0,h5fName=None):
    x = QDense(21,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              kernel_quantizer=quantized_bits(bits,ints,alpha=1),
              bias_quantizer=quantized_bits(6,0,alpha=1),
              name="Dense_Layer_1")(Inputs)

  
    x = QActivation(activation=quantized_relu(bits,ints),name="Relu_Layer_1")(x)
    
    x = QDense(22,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              kernel_quantizer=quantized_bits(bits,ints,alpha=1), 
              bias_quantizer=quantized_bits(bits,ints,alpha=1),
              name="Dense_Layer_2")(x)

    
    x = QActivation(activation=quantized_relu(bits,ints),name="Relu_Layer_2")(x)
    
    x = QDense(8,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              kernel_quantizer=quantized_bits(bits,ints,alpha=1),
              bias_quantizer=quantized_bits(bits,ints,alpha=1),
              name="Dense_Layer_3")(x)

   
    x = QActivation(activation=quantized_relu(bits),name="Relu_Layer_3")(x)
    
    x = QDense(1,activation=None, kernel_initializer='lecun_uniform',
              kernel_regularizer = l1(l1Reg),
              bias_regularizer = l1(l1Reg),
              kernel_quantizer=quantized_bits(bits,ints,alpha=1),
              bias_quantizer=quantized_bits(bits,ints,alpha=1),
              name="Dense_Layer_4")(x)

    #x = QActivation("quantized_bits(20,5)",name="Final_quantization")(x)
   
    predictions = Activation(activation='sigmoid',name="Sigmoid_Output_Layer")(x)
   
    model = Model(inputs=Inputs, outputs=predictions)

    return(model)

