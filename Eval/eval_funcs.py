import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import pandas as pd
import uproot
import os
plt.style.use(hep.cms.style.ROOT)

def streamlined(trackdf,threshold,model_names):
    from sklearn.metrics import confusion_matrix

    totalstrue = len(trackdf[trackdf["trk_fake"]==1])
    totalsfalse= len(trackdf[trackdf["trk_fake"]==0])

    TP = np.zeros([len(model_names)])
    FN = np.zeros([len(model_names)])
    TN = np.zeros([len(model_names)])
    FP = np.zeros([len(model_names)])

    for i in range(len(model_names)):
        trackdf["class_"+str(i)] = trackdf["class_output_"+str(i)]
        trackdf["class_"+str(i)][trackdf["class_"+str(i)]>threshold] = 1
        trackdf["class_"+str(i)][trackdf["class_"+str(i)]<=threshold] = 0

        TN[i], FP[i], FN[i], TP[i] = confusion_matrix(trackdf["trk_fake"],trackdf["class_"+str(i)]).ravel()

    return (TP,FN,TN,FP,totalstrue,totalsfalse)

def newfull_rates(trackdf,model_names,threshold):

    tprs = np.zeros([len(model_names)])
    fprs = np.zeros([len(model_names)])
    tnrs = np.zeros([len(model_names)])
    fnrs = np.zeros([len(model_names)])
    totalstrue = np.zeros([len(model_names)])
    totalsfalse = np.zeros([len(model_names)])

    TP,FN,TN,FP,tottrue,totfalse = streamlined(trackdf,threshold,model_names)
    tprs[:] = TP/(TP+FN)
    fprs[:] = FP/(FP+TN)
    tnrs[:] = TN/(TP+FP)
    fnrs[:] = FN/(TN+TP)
    totalstrue[:] = tottrue
    totalsfalse[:] = totfalse

    return tprs,fprs,tnrs,fnrs,totalstrue,totalsfalse
       
def roc_auc(trackdf,model_names):

    from sklearn import metrics
    fpr = []
    tpr = []
    auc = []
    genuine_array = []
    fake_array = []
    for i in range(len(model_names)):
        temp_fpr, temp_tpr, thresholds = metrics.roc_curve(trackdf["trk_fake"] ,trackdf["class_output_"+str(i)], pos_label=1)
        temp_auc = metrics.roc_auc_score(trackdf["trk_fake"],trackdf["class_output_"+str(i)])
        auc.append(temp_auc)
        fpr.append(temp_fpr)
        tpr.append(temp_tpr)
        
        
    
    
        genuine = []
        fake = []


        for j in range(len(trackdf["class_output_"+str(i)])):
            if trackdf["trk_fake"][j] == 1:
                genuine.append(trackdf["class_output_"+str(i)][j])
            else:
                fake.append(trackdf["class_output_"+str(i)][j])
                
        genuine_array.append(genuine)
        fake_array.append(fake)

    
        fig, ax = plt.subplots(1,2, figsize=(18,9)) 
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)

        ax[0].set_title("Balanced Accuracy Score: %.3f"%metrics.balanced_accuracy_score(trackdf["trk_fake"],trackdf["class_"+str(i)]) ,loc='left',fontsize=20)
        ax[0].plot(temp_fpr,temp_tpr,label=model_names[i] + "AUC: %.3f"%temp_auc)
        ax[0].set_xlim([0.0,0.3])
        ax[0].set_ylim([0.7,1.0])
        ##ax[0].plot(fpr,fpr,"--",color='r',label="Random Guess: 0.5")
        ax[0].set_xlabel("False Positive Rate",ha="right",x=1,fontsize=16)
        ax[0].set_ylabel("Identification Efficiency",ha="right",y=1,fontsize=16)
        ax[0].legend()
        ax[0].grid()

        ax[1].set_title("Tested on: "+str(len(trackdf)) + " Tracks" ,loc='left',fontsize=20)
        ax[1].hist(genuine,color='g',bins=20,range=(0,1),alpha=1,label="Genuine",density=True,histtype="step",linewidth=1)
        ax[1].hist(fake,color='r',bins=20,range=(0,1),alpha=1,label="Fake",density=True,histtype="step",linewidth=1)
        ax[1].grid()
        ax[1].set_xlabel(model_names[i]+ " Output",ha="right",x=1,fontsize=16)
        ax[1].set_ylabel("a.u.",ha="right",y=1,fontsize=16)
        ax[1].legend()

        plt.tight_layout()
        #plt.savefig("plots/"+model_names[i] + "highres.png",dpi=600)
        plt.savefig("plots/"+model_names[i] + ".png",dpi=100)





    fig, ax = plt.subplots(figsize=(12,12)) 
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title("Reciever Operating Characteristic Curves" ,loc='left',fontsize=20)

    for i in range(len(model_names)-1):
        print(fpr[i],tpr[i],auc[i])
        ax.plot(fpr[i],tpr[i],label=model_names[i]+ " AUC: %.3f"%auc[i])
    
    ax.scatter(fpr[2], tpr[2],color='g',s=50,marker='x',label="$\chi^2$ "+ " AUC: %.3f"%auc[2])
    ax.set_xlim([0.0,0.3])
    ax.set_ylim([0.7,1.0])
    ##ax[0].plot(fpr,fpr,"--",color='r',label="Random Guess: 0.5")
    ax.set_xlabel("False Positive Rate",ha="right",x=1,fontsize=16)
    ax.set_ylabel("Identification Efficiency",ha="right",y=1,fontsize=16)
    ax.legend()
    ax.grid()


    plt.tight_layout()
    #plt.savefig("plots/"+"all" + "highres.png",dpi=600)
    plt.savefig("plots/"+"all" + "lowres.png",dpi=100)


def bins(trackdf,model_names,parameter,threshold=0.5):

    if parameter=="eta":
        plot_type_name = "$\eta$"
        bin_range = np.linspace(-2.34,2.34,20)

    if parameter=="pt":
        plot_type_name = "$p_T$"
        bin_range = np.linspace(0,200,10)

    if parameter=="phi":
    
        plot_type_name = "$\phi$"
        bin_range = np.linspace(-3.141,3.141,10)
        

    plot_name = parameter
    bin_width = abs(bin_range[0]-bin_range[-1])/(len(bin_range)*2)

    temp_df = pd.DataFrame()
    
    tprs = np.zeros([len(model_names),len(bin_range)-1,2])
    fprs = np.zeros([len(model_names),len(bin_range)-1,2])
    tnrs = np.zeros([len(model_names),len(bin_range)-1,2])
    fnrs = np.zeros([len(model_names),len(bin_range)-1,2])

    for i in range(len(bin_range)-1):
        temp_df = trackdf[(trackdf["trk_"+parameter] > bin_range[i]-bin_width/2) & (trackdf["trk_"+parameter] <= bin_range[i]+bin_width/2)]

        temp_tprs,temp_fprs,temp_tnrs,temp_fnrs,totalstrue,totalsfalse = newfull_rates(temp_df,model_names,threshold)

        tprs[:,i,0] = temp_tprs
        tprs[:,i,1] = np.sqrt((temp_tprs*(1-temp_tprs))/totalstrue)

        fprs[:,i,0] = temp_fprs
        fprs[:,i,1] = np.sqrt((temp_fprs*(1-temp_fprs))/totalstrue)

        tnrs[:,i,0] = temp_tnrs
        tnrs[:,i,1] = np.sqrt((temp_tnrs*(1-temp_tnrs))/totalsfalse)

        fnrs[:,i,0] = temp_fnrs
        fnrs[:,i,1] = np.sqrt((temp_fnrs*(1-temp_fnrs))/totalsfalse)
    
    bin_range =  bin_range[0:-1]
        

    fig, ax = plt.subplots(figsize=(12,12)) 
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title("True Positive Rate vs " +plot_type_name ,loc='left',fontsize=20)

    for i in range(len(model_names)-1):
        ax.errorbar(bin_range,tprs[i,:,0],yerr=tprs[i,:,1],xerr=bin_width, label=model_names[i])
    ax.errorbar(bin_range,tprs[2,:,0],yerr=tprs[2,:,1],xerr=bin_width, label="$\chi^2$ ")

    ax.set_xlabel(plot_type_name,ha="right",x=1,fontsize=16)
    ax.set_ylabel("TPR",ha="right",y=1,fontsize=16)
    ax.legend()
    ax.grid()


    plt.tight_layout()
    #plt.savefig("plots/"+"tprvs"+plot_name + "_highres.png",dpi=600)
    plt.savefig("plots/"+"tprvs"+plot_name + "_lowres.png",dpi=100)

    ###############################################################################################
    fig, ax = plt.subplots(figsize=(12,12)) 
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title("False Positive Rate vs " +plot_type_name ,loc='left',fontsize=20)

    for i in range(len(model_names)-1):
        ax.errorbar(bin_range,fprs[i,:,0],yerr=fprs[i,:,1],xerr=bin_width, label=model_names[i])
    ax.errorbar(bin_range,fprs[2,:,0],yerr=fprs[2,:,1],xerr=bin_width, label="$\chi^2$ ")

    ax.set_xlabel(plot_type_name,ha="right",x=1,fontsize=16)
    ax.set_ylabel("FPR",ha="right",y=1,fontsize=16)
    ax.legend()
    ax.grid()


    plt.tight_layout()
    #plt.savefig("plots/"+"fprvs"+plot_name + "_highres.png",dpi=600)
    plt.savefig("plots/"+"fprvs"+plot_name + "_lowres.png",dpi=100)

    ############################################################################################

    fig, ax = plt.subplots(figsize=(12,12)) 
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title("True Negative Rate vs "+plot_type_name ,loc='left',fontsize=20)

    for i in range(len(model_names)-1):
        ax.errorbar(bin_range,tnrs[i,:,0],yerr=tnrs[i,:,1],xerr=bin_width, label=model_names[i])
    ax.errorbar(bin_range,tnrs[2,:,0],yerr=tnrs[2,:,1],xerr=bin_width, label="$\chi^2$ ")

    ax.set_xlabel(plot_type_name,ha="right",x=1,fontsize=16)
    ax.set_ylabel("TNR",ha="right",y=1,fontsize=16)
    ax.legend()
    ax.grid()


    plt.tight_layout()
    #plt.savefig("plots/"+"tnrvs"+plot_name + "_highres.png",dpi=600)
    plt.savefig("plots/"+"tnrvs"+plot_name + "_lowres.png",dpi=100)

    ############################################################################################

    fig, ax = plt.subplots(figsize=(12,12)) 
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title("False Negative Rate vs "+plot_type_name ,loc='left',fontsize=20)

    for i in range(len(model_names)-1):
        ax.errorbar(bin_range,fnrs[i,:,0],yerr=fnrs[i,:,1],xerr=bin_width, label=model_names[i])
    ax.errorbar(bin_range,fnrs[2,:,0],yerr=fnrs[2,:,1],xerr=bin_width, label="$\chi^2$ ")

    ax.set_xlabel(plot_type_name,ha="right",x=1,fontsize=16)
    ax.set_ylabel("FNR",ha="right",y=1,fontsize=16)
    ax.legend()
    ax.grid()


    plt.tight_layout()
    #plt.savefig("plots/"+"fnrvs"+plot_name + "_highres.png",dpi=600)
    plt.savefig("plots/"+"fnrvs"+plot_name + "_lowres.png",dpi=100)

    ############################################################################################


def own_roc(trackdf,model_names):
    thresholds = np.linspace(0,1,100)
    tprs = np.zeros([len(model_names),len(thresholds),2])
    fprs = np.zeros([len(model_names),len(thresholds),2])
    tnrs = np.zeros([len(model_names),len(thresholds),2])
    fnrs = np.zeros([len(model_names),len(thresholds),2])

    for i in range(len(thresholds)):

        temp_tprs,temp_fprs,temp_tnrs,temp_fnrs,totalstrue,totalsfalse = newfull_rates(trackdf,model_names,thresholds[i])

        tprs[:,i,0] = temp_tprs
        tprs[:,i,1] = np.sqrt((temp_tprs*(1-temp_tprs))/totalstrue)

        fprs[:,i,0] = temp_fprs
        fprs[:,i,1] = np.sqrt((temp_fprs*(1-temp_fprs))/totalstrue)

        tnrs[:,i,0] = temp_tnrs
        tnrs[:,i,1] = np.sqrt((temp_tnrs*(1-temp_tnrs))/totalsfalse)

        fnrs[:,i,0] = temp_fnrs
        fnrs[:,i,1] = np.sqrt((temp_fnrs*(1-temp_fnrs))/totalsfalse)


    fig, ax = plt.subplots(figsize=(12,12)) 
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title("Reciever Operating Characteristic Curves" ,loc='left',fontsize=20)

    for i in range(len(model_names)-1):
        
        ax.plot(fprs[i,:,0],tprs[i,:,0],label=model_names[i])
        ax.plot(fprs[i,:,0]+1.96*tprs[i,:,1],tprs[i,:,0]-1.96*tprs[i,:,1],"--")
        ax.plot(fprs[i,:,0]-1.96*tprs[i,:,1],tprs[i,:,0]+1.96*tprs[i,:,1],"--")
        ax.fill_between(fprs[i,:,0], tprs[i,:,0]-1.96*tprs[i,:,1], tprs[i,:,0]+1.96*tprs[i,:,1],color="g",alpha=0.1)
    
    ax.errorbar(fprs[2,1,0],tprs[2,1,0], xerr=1.96*fprs[2,1,1], yerr=1.96*tprs[2,1,1],color='g',label="$\chi^2$")

    ax.set_xlim([0.0,0.3])
    ax.set_ylim([0.7,1.0])
    #ax.plot(fprs[0,:,0],fprs[0,:,0],"--",color='r',label="Random Guess: 0.5")
    ax.set_xlabel("False Positive Rate",ha="right",x=1,fontsize=16)
    ax.set_ylabel("Identification Efficiency",ha="right",y=1,fontsize=16)
    ax.legend()
    ax.grid()


    plt.tight_layout()
    plt.savefig("plots/"+"ownROC" + "highres.png",dpi=600)
    plt.savefig("plots/"+"ownROC" + "lowres.png",dpi=100)


def lepton_split(trackdf,model_names):
    electrondf = trackdf[(trackdf["trk_matchtp_pdgid"]==-11)|(trackdf["trk_matchtp_pdgid"]==11)]
    muondf = trackdf[(trackdf["trk_matchtp_pdgid"]==-13)|(trackdf["trk_matchtp_pdgid"]==13)]
    hadrondf = trackdf[(trackdf["trk_matchtp_pdgid"]!=-11)|(trackdf["trk_matchtp_pdgid"]!=11)|(trackdf["trk_matchtp_pdgid"]!=-13)|(trackdf["trk_matchtp_pdgid"]!=13)]

    noElectrons = len(electrondf)
    noMuons = len(muondf)
    noHadrons = len(hadrondf)

    ElectronsTP = np.zeros([len(model_names),100])
    MuonsTP = np.zeros([len(model_names),100])
    HadronsTP = np.zeros([len(model_names),100])
    

    thresholds = np.linspace(0,1,100)
    for i in range(len(model_names)):
        for j in range(len(thresholds)):
            electrondf["class_"+str(i)] = electrondf["class_output_"+str(i)]
            electrondf["class_"+str(i)][electrondf["class_"+str(i)]>thresholds[j]] = 1
            electrondf["class_"+str(i)][electrondf["class_"+str(i)]<=thresholds[j]] = 0

            muondf["class_"+str(i)] = muondf["class_output_"+str(i)]
            muondf["class_"+str(i)][muondf["class_"+str(i)]>thresholds[j]] = 1
            muondf["class_"+str(i)][muondf["class_"+str(i)]<=thresholds[j]] = 0

            hadrondf["class_"+str(i)] = hadrondf["class_output_"+str(i)]
            hadrondf["class_"+str(i)][hadrondf["class_"+str(i)]>thresholds[j]] = 1
            hadrondf["class_"+str(i)][hadrondf["class_"+str(i)]<=thresholds[j]] = 0            
            
            ElectronsTP[i,j] = len(electrondf[electrondf["class_"+str(i)] >= thresholds[j]])/noElectrons
            MuonsTP[i,j] = len(muondf[muondf["class_"+str(i)] >= thresholds[j]])/noMuons
            HadronsTP[i,j] = len(hadrondf[hadrondf["class_"+str(i)] >= thresholds[j]])/noHadrons


    fig, ax = plt.subplots(1,1, figsize=(12,12)) 
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_title(str(noElectrons) + " Electron Tracks "+ str(noMuons) + " Muon Tracks "  + str(noHadrons) + " Hadrons",loc='left',fontsize=20)
    
    ax.plot(thresholds,ElectronsTP[0],color='r',label="Electrons "+model_names[0]+" Cut")
    
    
    ax.plot(thresholds,MuonsTP[0],color='g',label="Muons "+model_names[0]+" Cut")
    
    
    ax.plot(thresholds,HadronsTP[0],color='c',label="Hadrons "+model_names[0]+" Cut")
    
    
    ax.plot(thresholds[1:],ElectronsTP[2][1:],color='r',linestyle='--',label="Electrons $\chi^2$ Cut")
    plt.fill_between(thresholds[1:], ElectronsTP[2][1:]-2.56*1/np.sqrt(ElectronsTP[2][1:]*noElectrons), ElectronsTP[2][1:]+2.56*1/np.sqrt(ElectronsTP[2][1:]*noElectrons),color="r",alpha=0.1)
    
    ax.plot(thresholds[1:],MuonsTP[2][1:],color='g',linestyle='--',label="Muons $\chi^2$ Cut")
    plt.fill_between(thresholds[1:], MuonsTP[2][1:]-2.56*1/np.sqrt(MuonsTP[2][1:]*noMuons), MuonsTP[2][1:]+2.56*1/np.sqrt(MuonsTP[2][1:]*noMuons),color="g",alpha=0.1)
    
    ax.plot(thresholds[1:],HadronsTP[2][1:],color='c',linestyle='--',label="Hadrons $\chi^2$ Cut")
    plt.fill_between(thresholds[1:], HadronsTP[2][1:]-2.56*1/np.sqrt(HadronsTP[2][1:]*noHadrons), HadronsTP[2][1:]+2.56*1/np.sqrt(HadronsTP[2][1:]*noHadrons),color="c",alpha=0.1)
    
    plt.fill_between(thresholds, ElectronsTP[0]-2.56*1/np.sqrt(ElectronsTP[0]*noElectrons), ElectronsTP[0]+2.56*1/np.sqrt(ElectronsTP[0]*noElectrons),color="r",alpha=0.1)
    plt.fill_between(thresholds, MuonsTP[0]-2.56*1/np.sqrt(MuonsTP[0]*noMuons), MuonsTP[0]+2.56*1/np.sqrt(MuonsTP[0]*noMuons),color="g",alpha=0.1)
    plt.fill_between(thresholds, HadronsTP[0]-2.56*1/np.sqrt(HadronsTP[0]*noHadrons), HadronsTP[0]+2.56*1/np.sqrt(HadronsTP[0]*noHadrons),color="c",alpha=0.1)
    
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.1)
    ax.grid()
    ax.set_xlabel("Threshold",ha="right",x=1,fontsize=16)
    ax.set_ylabel("Efficiency",ha="right",y=1,fontsize=16)
    ax.legend(prop={'size':16})

    plt.tight_layout()
    plt.savefig("plots/"+model_names[0]+"particleefficiency.png",dpi=100)
    plt.savefig("plots/"+model_names[0]+"particleefficiencyhr.png",dpi=600)
    
    ################################################################################################################################################

    fig, ax = plt.subplots(1,1, figsize=(12,12)) 
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)


    ax.set_title(str(noElectrons) + " Electron Tracks "+ str(noMuons) + " Muon Tracks "  + str(noHadrons) + " Hadrons",loc='left',fontsize=20)
    ax.plot(thresholds,ElectronsTP[1],color='r',label="Electrons "+model_names[1]+" Cut")
    
    
    ax.plot(thresholds,MuonsTP[1],color='g',label="Muons "+model_names[1]+" Cut")
    
    
    ax.plot(thresholds,HadronsTP[1],color='c',label="Hadrons "+model_names[1]+" Cut")
    
    
    ax.plot(thresholds[1:],ElectronsTP[2][1:],color='r',linestyle='--',label="Electrons $\chi^2$ Cut")
    plt.fill_between(thresholds[1:], ElectronsTP[2][1:]-2.56*1/np.sqrt(ElectronsTP[2][1:]*noElectrons), ElectronsTP[2][1:]+2.56*1/np.sqrt(ElectronsTP[2][1:]*noElectrons),color="r",alpha=0.1)
    
    ax.plot(thresholds[1:],MuonsTP[2][1:],color='g',linestyle='--',label="Muons $\chi^2$ Cut")
    plt.fill_between(thresholds[1:], MuonsTP[2][1:]-2.56*1/np.sqrt(MuonsTP[2][1:]*noMuons), MuonsTP[2][1:]+2.56*1/np.sqrt(MuonsTP[2][1:]*noMuons),color="g",alpha=0.1)
    
    ax.plot(thresholds[1:],HadronsTP[2][1:],color='c',linestyle='--',label="Hadrons $\chi^2$ Cut")
    plt.fill_between(thresholds[1:], HadronsTP[2][1:]-2.56*1/np.sqrt(HadronsTP[2][1:]*noHadrons), HadronsTP[2][1:]+2.56*1/np.sqrt(HadronsTP[2][1:]*noHadrons),color="c",alpha=0.1)
    
    plt.fill_between(thresholds[:-3], ElectronsTP[1][:-3]-2.56*1/np.sqrt(ElectronsTP[1][:-3]*noElectrons), ElectronsTP[1][:-3]+2.56*1/np.sqrt(ElectronsTP[1][:-3]*noElectrons),color="r",alpha=0.1)
    plt.fill_between(thresholds[:-3], MuonsTP[1][:-3]-2.56*1/np.sqrt(MuonsTP[1][:-3]*noMuons), MuonsTP[1][:-3]+2.56*1/np.sqrt(MuonsTP[1][:-3]*noMuons),color="g",alpha=0.1)
    plt.fill_between(thresholds, HadronsTP[1]-2.56*1/np.sqrt(HadronsTP[1]*noHadrons), HadronsTP[1]+2.56*1/np.sqrt(HadronsTP[1]*noHadrons),color="c",alpha=0.1)
    
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.1)
    ax.grid()
    ax.set_xlabel("Threshold",ha="right",x=1,fontsize=16)
    ax.set_ylabel("Efficiency",ha="right",y=1,fontsize=16)
    ax.legend(prop={'size':16})
    plt.tight_layout()
    plt.savefig("plots/"+model_names[1]+"particleefficiency.png",dpi=100)
    plt.savefig("plots/"+model_names[1]+"particleefficiencyhr.png",dpi=600)
    ################################################################################################################################################
    
    fig, ax = plt.subplots(1,1, figsize=(12,12)) 
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title(str(noElectrons) + " Electron Tracks ",loc='left',fontsize=20)
    ax.scatter(thresholds,ElectronsTP[0],color='orange',label="Electrons "+model_names[0]+" Cut")
    ax.scatter(thresholds,ElectronsTP[1],color='b',label="Electrons "+model_names[1]+" Cut")
    ax.plot(thresholds[1:],ElectronsTP[2][1:],color='g',linestyle='-',label="Electrons $\chi^2$ Cut")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.grid()
    ax.set_xlabel("Threshold",ha="right",x=1,fontsize=16)
    ax.set_ylabel("Efficiency",ha="right",y=1,fontsize=16)
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/"+"bothelectronefficiencyhd.png",dpi=600)
    plt.savefig("plots/"+"bothelectronefficiency.png",dpi=100)
    
    fig, ax = plt.subplots(1,1, figsize=(12,12)) 
    ax.set_title(str(noMuons) + " Muon Tracks "  ,loc='left',fontsize=20)
    ax.scatter(thresholds,MuonsTP[0],color='orange',label="Muons "+model_names[0]+" Cut")
    ax.scatter(thresholds,MuonsTP[1],color='b',label="Muons "+model_names[1]+" Cut")
    ax.plot(thresholds[1:],MuonsTP[2][1:],color='g',linestyle='-',label="Muons $\chi^2$ Cut")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.grid()
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Efficiency")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/"+"bothmuonefficiency.png",dpi=600)


    fig, ax = plt.subplots(1,1, figsize=(12,12)) 
    ax.set_title(str(noHadrons) + " Hadrons",loc='left',fontsize=20)
    ax.scatter(thresholds,HadronsTP[0],color='orange',label="Hadrons "+model_names[0]+" Cut")
    ax.scatter(thresholds,HadronsTP[1],color='b',label="Hadrons "+model_names[1]+" Cut")
    ax.plot(thresholds[1:],HadronsTP[2][1:],color='g',linestyle='-',label="Hadrons $\chi^2$ Cut")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.grid()
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Efficiency")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/"+"bothhadronefficiency.png",dpi=600)
    

 
    








