import pandas as pd
import numpy as np
import time
import pickle


def PVTracks(dataframe):
    return (dataframe[dataframe["trk_fake"]==1])

def pileupTracks(dataframe):
    return (dataframe[dataframe["trk_fake"]==2])

def fakeTracks(dataframe):
    return (dataframe[dataframe["trk_fake"]==0])

def genuineTracks(dataframe):
    return (dataframe[dataframe["trk_fake"] != 0])


def balanceData(dataframe,random_state=4):

    genuine = genuineTracks(dataframe)
    numgenuine = len(genuine)

    fake = fakeTracks(dataframe)
    numfake = len(fake)

    fraction = numfake/numgenuine

    genuine = genuine.sample(frac=fraction, replace=True, random_state=random_state)


    df = pd.concat([fake,genuine],ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def particleBalanceData(dataframe,random_state=4):
    electrons = dataframe[(dataframe["trk_matchtp_pdgid"] == -11) | (dataframe["trk_matchtp_pdgid"] == 11)]
    muons = dataframe[(dataframe["trk_matchtp_pdgid"] == -13) | (dataframe["trk_matchtp_pdgid"] == 13)]
    hadrons = dataframe[(dataframe["trk_matchtp_pdgid"] != -13) & (dataframe["trk_matchtp_pdgid"] != 13) & (dataframe["trk_matchtp_pdgid"] != 11) & (dataframe["trk_matchtp_pdgid"] != -11)]
    fakes = dataframe[dataframe["trk_fake"] == 0]
   
    efraction = len(muons)/len(electrons)    
    hfraction = len(muons)/len(hadrons)
    ffraction = 3*len(muons)/len(fakes)

    electrons = electrons.sample(frac=efraction,replace=True,random_state=random_state)
    hadrons = hadrons.sample(frac=hfraction,replace=True,random_state=random_state)
    fakes = fakes.sample(frac=ffraction,replace=True,random_state=random_state)


    df = pd.concat([electrons,muons,hadrons,fakes],ignore_index=True)

    del [electrons,muons,hadrons,fakes]
    
    df = df.sample(frac=1).reset_index(drop=True)

    return df
    


def load_transformed_data(name):
    store = pd.HDFStore(name+'.h5') 
    dataframe = store['df']
    store.close()
    return dataframe

def calculate_roc(model,X_test,y_test,start_time,plot_parameters,plot=False,NN=False): 
    from sklearn import metrics 
    
    if NN:
        A =  model.predict(X_test)[:,0]
        y_predict = model.predict(X_test)
    else:
        #A = model.predict(X_test)
        A = model.predict_proba(X_test)[:,1]
        y_predict = model.predict(X_test)

    B = y_test

    fpr, tpr, thresholds = metrics.roc_curve(y_test ,A, pos_label=1)
    auc = metrics.roc_auc_score(y_test,A)

    genuine = []
    fake = []

    for i in range(len(A)):
        if B[i] == 1:
            genuine.append(A[i])
        else:
            fake.append(A[i])

    end_time = time.time()-start_time
    if plot == True:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import mplhep as hep
        plt.style.use(hep.cms.style.ROOT)
        fig, ax = plt.subplots(1,2, figsize=(18,9)) 
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)

        ax[0].set_title("Accuracy Score: %.3f"%plot_parameters["score"] + " in %.3f"%end_time + " Seconds",loc='left',fontsize=20)
        ax[0].plot(fpr,tpr,label=str(plot_parameters["name"])+ " AUC: %.3f"%auc)
        ax[0].set_xlim([0,0.3])
        ax[0].set_ylim([0.8,1.0])
        #ax[0].plot(fpr,fpr,"--",color='r',label="Random Guess: 0.5")
        ax[0].set_xlabel("Fake Positive Rate",ha="right",x=1,fontsize=16)
        ax[0].set_ylabel("Identification Efficiency",ha="right",y=1,fontsize=16)
        ax[0].legend()
        ax[0].grid()

        ax[1].set_title(str(plot_parameters["lengths"]) + " Tracks" ,loc='left',fontsize=20)
        ax[1].hist(genuine,color='g',bins=20,range=(0,1),alpha=0.5,label="Genuine",density=True)
        ax[1].hist(fake,color='r',bins=20,range=(0,1),alpha=0.5,label="Fake",density=True)
        ax[1].grid()
        ax[1].set_xlabel(str(plot_parameters["name"])+ " Positive Class Probability",ha="right",x=1,fontsize=1)
        ax[1].set_ylabel("a.u.",ha="right",y=1,fontsize=1)
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(str(plot_parameters["name"]) + ".png",dpi=600)

    else:
        print("Results for " + str(plot_parameters["name"]))
        print("Accuracy: %.3f"%plot_parameters["score"])
        print("Time: %.3f"%end_time + " Seconds")
        print("ROC AUC: %.3f"%auc)
        print(str(plot_parameters["lengths"]) + " Tracks in " + str(plot_parameters["events"])+ " Events")
    
    return(auc)
