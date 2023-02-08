#!/usr/bin/env python
# coding: utf-8


#import python toolkits
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA,LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
import  xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.svm import SVC
import random
import math
from tqdm import tqdm
warnings.filterwarnings("ignore")


loc_path='../final_coding/'
#read data
TCGA_data=pd.read_csv(loc_path+"TCGA_dataset/gene_sel_data/train.csv",header=None)
TCGA_data=TCGA_data.values

TCGA_label=pd.read_csv(loc_path+"TCGA_dataset/train_TCGA_label.csv",header=None)
TCGA_label=list(TCGA_label.loc[:,0])


 #define the five-fold stratified sampling function
 def str_sam_get_5(label_loc,stra_sam_rete_loc,class_all):
     class_loc={}
     for i in range(class_all):
         class_loc[i]=[]
     for i in range(len(label_loc)):
         class_loc[label_loc[i]].append(i)
     str_sam_sel=[]
     class_num=[]
     for i in range(class_all):
         class_num.append(math.ceil(stra_sam_rate*len(class_loc[i])))
     for i in range(5):
         str_sam_loc=[]
         for j in range(class_all):
             if len(class_loc[j])>=class_num[j]:
                 strs=random.sample(class_loc[j],class_num[j])
                 str_sam_loc.extend(strs)
                 class_loc[j]=list(set(class_loc[j])-set(strs))
             else:
                 str_sam_loc.extend(class_loc[j])
         str_sam_sel.append(str_sam_loc)
     return str_sam_sel



 #get the accuarcy of model
 def acc_get(train,train_label,test,test_label,mx,mi,g):
     smote=SMOTE(k_neighbors=5)
     X_train,y_train=smote.fit_resample(train,train_label)
     X_test,y_test=test,test_label
   
     std=StandardScaler()
     X_train=std.fit_transform(X_train)
     X_test=std.transform(X_test)
    
     #train the model
     clf_xg=XGBClassifier(gamma=g,max_depth=mx,min_child_weight=mi,
                          learning_rate=0.4,booster='gbtree',n_jobs=-1)
     clf_xg.fit(X_train,y_train)
     c1=accuracy_score(y_test,clf_xg.predict(X_test))
    
     return c1


 #run the stratified sampling function
 stra_sam_rate=0.2
 TCGA_fold_loc=str_sam_get_5(TCGA_label,stra_sam_rate,15)


 #gamma:(0,3,0.2)
 #max_depth:(2,20,2)
 #min_child_weight:(1,6,0.5)
 gamma_num=np.arange(2,3,0.2)
 ma_num=np.arange(12,20,2)
 mi_num=np.arange(1,6,0.5)

 parameters=[]
 acc_val=[]
 for gamma in tqdm(gamma_num):
     for a in tqdm(ma_num):
         for i in tqdm(mi_num):
             acc1=0
             for i in range(5):
                 train_loc=TCGA_fold_loc[i]
                 test_loc=list(set(np.arange(len(TCGA_label)))-set(train_loc))
                 train_TCGA=TCGA_data[train_loc,:]
                 test_TCGA=TCGA_data[test_loc,:]
                 TCGA_label=np.array(TCGA_label)
                 train_TCGA_label=TCGA_label[train_loc]
                 test_TCGA_label=TCGA_label[test_loc]
                    
                 a1=acc_get(train_TCGA,train_TCGA_label,test_TCGA,test_TCGA_label,a,i,gamma)
                 acc1+=a1
             parameters.append([gamma,a,i])
             acc_val.append(acc1/5)



 #print the best parameters
 loc=acc_val.index(max(acc_val))
 print(max(acc_val))
 print(parameters[loc])


#training all the training data and write the parameters of model to file
smote=SMOTE(k_neighbors=5)
X_train,y_train=smote.fit_resample(TCGA_data,TCGA_label)

std=StandardScaler()
X_train=std.fit_transform(X_train)

#training model
ga,ma_num,mi_num=parameters[loc][0],parameters[loc][1],parameters[loc][2]
clf_xg=XGBClassifier(gamma=ga,max_depth=ma_num,min_child_weight=mi_num,
                     learning_rate=0.4,booster='gbtree',n_jobs=-1)
clf_xg.fit(X_train,y_train)


import pickle
with open("./model_file/TCGA_clf_xg_all.pickle",'wb') as f:
    pickle.dump(clf_xg,f)


 #stander the testing data
 TCGA_test=pd.read_csv(loc_path+"TCGA_dataset/gene_sel_data/test.csv",header=None)
 TCGA_test=TCGA_test.values
 print(TCGA_test.shape)

 TCGA_test=std.transform(TCGA_test)
 TCGA_test=pd.DataFrame(TCGA_test)
 TCGA_test.to_csv(loc_path+"TCGA_dataset/gene_sel_data/test_standard.csv",header=None,index=0)


 #training all the training data and write the parameters of model to file

 smote=SMOTE(k_neighbors=5)
 X_train,y_train=smote.fit_resample(TCGA_data,TCGA_label)

 std=StandardScaler()
 X_train=std.fit_transform(X_train)

 clf_knn=KNeighborsClassifier(15,'distance',n_jobs=-1)
 clf_knn.fit(X_train,y_train)
 with open("./model_file/TCGA_clf_knn_all.pickle",'wb') as f:
     pickle.dump(clf_knn,f)
 #
 clf_lin = SVC(decision_function_shape='ovo',kernel='linear',probability=True,random_state=42)
 clf_lin.fit(X_train,y_train)
 with open("./model_file/TCGA_clf_lin_all.pickle",'wb') as f:
     pickle.dump(clf_lin,f)


 clf_rbf = SVC(decision_function_shape='ovo',kernel='rbf',probability=True,random_state=42)
 clf_rbf.fit(X_train,y_train)
 with open("./model_file/TCGA_clf_rbf_all.pickle",'wb') as f:
     pickle.dump(clf_rbf,f)

