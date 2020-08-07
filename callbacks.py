from tensorflow.keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
import tensorflow as tf
import math

class EarlyStopAfterPrune(EarlyStopping):
    def __init__(self, monitor='val_loss',
             min_delta=0, patience=0, verbose=0, mode='auto', start_epoch = 100): # add argument for starting epoch
        super(EarlyStopAfterPrune, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class LR_schedule(Callback):
    def __init__(self,initial_lr,Prune_begin,Prune_end,lr_factor_1,lr_factor_2,lr_factor_3):
         super(LR_schedule, self).__init__()
         self.initial_lr = initial_lr
         self.Prune_begin = Prune_begin
         self.Prune_end = Prune_end
         self.lr_factor_1 = lr_factor_1
         self.lr_factor_2 = lr_factor_2
         self.lr_factor_3 = lr_factor_3

    def on_epoch_begin(self, epoch,logs=None):
        if epoch > self.Prune_begin:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr*self.lr_factor_1+self.intial_lr*self.lr_factor_2*math.exp(-epoch*self.lr_factor_3))

        if epoch == self.Prune_end:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)



class all_callbacks(object):
    def __init__(self,
                 initial_lr=0.001,
                 stop_patience=10,
                 lr_factor=0.5,
                 lr_patience=1,
                 lr_epsilon=0.001,
                 lr_cooldown=4,
                 lr_minimum=1e-5,
                 Prune_begin=100,
                 Prune_end=650,
                 prune_lrs=[1,0,0],
                 outputDir=''):
        

        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, 
                                mode='min', verbose=1, min_delta=lr_epsilon,
                                 cooldown=lr_cooldown, min_lr=lr_minimum)

        self.modelbestcheck=ModelCheckpoint(outputDir+"/KERAS_check_best_model.h5", 
                                        monitor='val_loss', verbose=1, 
                                        save_best_only=True)

        self.modelbestcheckweights=ModelCheckpoint(outputDir+"/KERAS_check_best_model_weights.h5", 
                                            monitor='val_loss', verbose=1, 
                                            save_best_only=True,save_weights_only=True)
                  
        self.stopping = EarlyStopAfterPrune(monitor='val_loss', 
                                      patience=stop_patience, 
                                      verbose=1, mode='min', start_epoch=Prune_end)

        self.learning_schedule = LR_schedule(initial_lr=initial_lr,
                                     Prune_begin=Prune_begin,Prune_end=Prune_end,
                                     lr_factor_1=prune_lrs[0],lr_factor_2=prune_lrs[1],lr_factor_3=prune_lrs[2])


  
        self.callbacks=[
           self.modelbestcheck,self.modelbestcheckweights, self.reduce_lr,self.stopping,self.learning_schedule
        ]

        if outputDir == "None":
           self.callbacks=[
           self.reduce_lr,self.stopping,self.learning_schedule
        ]
