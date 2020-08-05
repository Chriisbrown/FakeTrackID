import uproot
import pandas as pd
import numpy as np
import time
import pickle
import sys

def pttoR(pt):
    B = 3.8112 #Tesla for CMS magnetic field

    return 1000*abs((B*(3e8/1e11))/(2*pt))

def tanL(eta):
    return abs(np.sinh(eta))

def logChi(chi2):
    return np.log(chi2)

def sum_digits3(n):
   r = 0
   while n:
       r, n = r + n % 10, n // 10
   return r
 

def nhotdisk(dataframe):

    disk = dataframe["trk_dhits"].astype(str).to_numpy()


    disk_array = np.zeros([5,len(disk)])


    for i in range(len(disk)):

        for k in range(len(disk[i])):
            disk_array[k,i] = disk[i][-(k+1)]


    dataframe["disk1"] = disk_array[0,:]
    dataframe["disk2"] = disk_array[1,:]
    dataframe["disk3"] = disk_array[2,:]
    dataframe["disk4"] = disk_array[3,:]
    dataframe["disk5"] = disk_array[4,:]


    return(dataframe)

def nhotlayer(dataframe):
    layer = dataframe["trk_lhits"].astype(str).to_numpy()

    layer_array = np.zeros([6,len(layer)])

    for i in range(len(layer)):
        for j in range(len(layer[i])):

            layer_array[j,i] = layer[i][-(j+1)]



    dataframe["layer1"] = layer_array[0,:]
    dataframe["layer2"] = layer_array[1,:]
    dataframe["layer3"] = layer_array[2,:]
    dataframe["layer4"] = layer_array[3,:]
    dataframe["layer5"] = layer_array[4,:]
    dataframe["layer6"] = layer_array[5,:]


    return(dataframe)

def predhitpattern(dataframe):
    dataframe["trk_hitpattern"] = dataframe["trk_hitpattern"].apply(bin)
    hitpat = dataframe["trk_hitpattern"].astype(str).to_numpy()
    hit_array = np.zeros([7,len(hitpat)])
    expanded_hit_array = np.zeros([12,len(hitpat)])
    ltot = np.zeros(len(hitpat))
    dtot = np.zeros(len(hitpat))
    for i in range(len(hitpat)):
        for k in range(len(hitpat[i])-2):
            hit_array[k,i] = hitpat[i][-(k+1)]
    
    eta_bins = [0.0,0.2,0.41,0.62,0.9,1.26,1.68,2.08,2.5]
    conversion_table = np.array([[0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  6,  7,  8,  9 ],
                                 [0, 1,  7,  8,  9, 10,  11],
                                 [0, 6,  7,  8,  9, 10,  11]])
    
    for i in range(len(hitpat)):
        for j in range(8):
            if ((abs(dataframe["trk_eta"][i]) >= eta_bins[j]) & (abs(dataframe["trk_eta"][i]) < eta_bins[j+1])):
                for k in range(7):
                    expanded_hit_array[conversion_table[j][k]][i] = hit_array[k][i]


        ltot[i] = sum(expanded_hit_array[0:6,i])
        dtot[i] = sum(expanded_hit_array[6:11,i])
    
    dataframe["pred_layer1"] = expanded_hit_array[0,:]
    dataframe["pred_layer2"] = expanded_hit_array[1,:]
    dataframe["pred_layer3"] = expanded_hit_array[2,:]
    dataframe["pred_layer4"] = expanded_hit_array[3,:]
    dataframe["pred_layer5"] = expanded_hit_array[4,:]
    dataframe["pred_layer6"] = expanded_hit_array[5,:]
    dataframe["pred_disk1"] = expanded_hit_array[6,:]
    dataframe["pred_disk2"] = expanded_hit_array[7,:]
    dataframe["pred_disk3"] = expanded_hit_array[8,:]
    dataframe["pred_disk4"] = expanded_hit_array[9,:]
    dataframe["pred_disk5"] = expanded_hit_array[10,:]
    dataframe["pred_ltot"] = ltot
    dataframe["pred_dtot"] = dtot

    return dataframe


def transformData(dataframe):
    dataframe["InvR"] = dataframe["trk_pt"].apply(pttoR)
    dataframe["BigInvR"] = dataframe["InvR"]*1000
    dataframe["TanL"] = dataframe["trk_eta"].apply(tanL)
    dataframe["LogChi"] = dataframe["trk_chi2"].apply(logChi)
    dataframe["LogChirphi"] = dataframe["trk_chi2rphi"].apply(logChi)
    dataframe["LogChirz"] = dataframe["trk_chi2rz"].apply(logChi)
    dataframe["LogBendChi"] = np.log(dataframe["trk_bendchi2"])
    dataframe["ModZ"] = dataframe["trk_z0"].apply(np.abs)
    dataframe["dtot"] = dataframe["trk_dhits"].apply(sum_digits3)
    dataframe["ltot"] = dataframe["trk_lhits"].apply(sum_digits3)

    dataframe = nhotdisk(dataframe)
    dataframe = nhotlayer(dataframe)

    return dataframe


def loadData(name,transform=False):
    tracks = uproot.pandas.iterate(name+".root",treepath=b'L1TrackNtuple;1/eventTree;1',branches="trk*")
    events = (uproot.tree.numentries(name+".root",treepath=b'L1TrackNtuple;1/eventTree;1'))
    #Open file, find the track tuple and the event tree
    trackdf = pd.DataFrame()
    for i,df in enumerate(tracks):
        

        if transform:
             df = transformData(df)
             
        trackdf = trackdf.append(df,ignore_index=True)


    return trackdf,events

def save_transformed_data(dataframe,name):

    store = pd.HDFStore(name+'.h5')

    store['df'] = dataframe  # save it
    store.close()


if __name__ == "__main__":

    name = "Data/hybrid10kv11"

    trackdf,events = loadData(name,transform=True)
    infs = np.where(np.asanyarray(np.isnan(trackdf)))[0]
    trackdf.drop(infs,inplace=True)

    trackdf.reset_index(inplace=True)

    trackdf = predhitpattern(trackdf)
    save_transformed_data(trackdf,name)
    print("saved....")
