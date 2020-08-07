import numpy as np


class cut_based_classifier:
    def __init__(self,pt,eta,z0,chi2,bendchi2,nstub):
        
        self.pt = pt
        self.eta = eta
        self.z0 = z0
        self.chi2 = chi2
        self.bendchi2 = bendchi2
        self.nstub = nstub

    def fit(self,X,y):
        pass

    def predict(self,dataframe):
        y = np.where(((dataframe["trk_pt"] > self.pt) & 
                      (np.abs(dataframe["trk_eta"]) < self.eta) &
                      (np.abs(dataframe["trk_z0"]) < self.z0) &
                      (dataframe["trk_chi2"] < self.chi2) &
                      (dataframe["trk_bendchi2"] < self.bendchi2) &
                      (dataframe["trk_nstub"] >= self.nstub)),1,0)



        return y

    def predict_proba(self,dataframe):
        return self.predict(dataframe)


class perfect_classifier:
    def __init__(self,neg_label=0):
        self.neg_label = neg_label


    def fit(self,X,y):
        pass

    def predict(self,dataframe):
        y = np.where((dataframe["trk_fake"] != self.neg_label) , np.ones(len(dataframe)), 0)
        return y

    def predict_proba(self,dataframe):
        return self.predict(dataframe)

