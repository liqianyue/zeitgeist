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
warnings.filterwarnings("ignore")



loc_path='../final_coding/'
#read data set 
TCGA_data=pd.read_csv(loc_path+"TCGA_dataset/train_TCGA.csv",header=None)
GEO_data=pd.read_csv(loc_path+"GEO_dataset/train_GEO.csv",header=None)
TCGA_data=TCGA_data.values
GEO_data=GEO_data.values

TCGA_label=pd.read_csv(loc_path+"TCGA_dataset/train_TCGA_label.csv",header=None)
GEO_label=pd.read_csv(loc_path+"GEO_dataset/train_GEO_label.csv",header=None)
TCGA_label=list(TCGA_label.loc[:,0])
GEO_label=list(GEO_label.loc[:,0])

TCGA_gene_name=pd.read_csv(loc_path+"TCGA_dataset/TCGA_gene_name.csv",header=None)
GEO_gene_name=pd.read_csv(loc_path+"GEO_dataset/GEO_gene_name.csv",header=None)
TCGA_gene_name=list(TCGA_gene_name.loc[:,0])
GEO_gene_name=list(GEO_gene_name.loc[:,0])


TCGA_gene_impor=pd.read_csv(loc_path+"TCGA_dataset/TCGA_gene_importance.csv")
GEO_gene_impor=pd.read_csv(loc_path+"GEO_dataset/GEO_gene_importance.csv")




#define of a five-fold stratified sampling function
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


#get tbe accuarcy of model 
def acc_get(train,train_label,test,test_label):
    smote=SMOTE(k_neighbors=5)
    X_train,y_train=smote.fit_resample(train,train_label)
    X_test,y_test=test,test_label
    
    std=StandardScaler()
    X_train=std.fit_transform(X_train)
    X_test=std.transform(X_test)
    
    #train the model
    clf_xg=XGBClassifier(max_depth=15,learning_rate=0.4,booster='gbtree')
    clf_xg.fit(X_train,y_train)
    c1=accuracy_score(y_test,clf_xg.predict(X_test))
    print(1)
    
    clf_knn=KNeighborsClassifier(15,'distance')
    clf_knn.fit(X_train,y_train)
    c2=accuracy_score(y_test,clf_knn.predict(X_test))
    print(2)
    #
    clf_lin = SVC(decision_function_shape='ovo',kernel='linear',probability=True,random_state=42)
    clf_lin.fit(X_train,y_train)
    c3=accuracy_score(y_test,clf_lin.predict(X_test))
    print(3)
    
    clf_rbf = SVC(decision_function_shape='ovo',kernel='rbf',probability=True,random_state=42)
    clf_rbf.fit(X_train,y_train)
    c4=accuracy_score(y_test,clf_rbf.predict(X_test))
    print(4)
    
    return c1,c2,c3,c4


#preform stratified sampling

#sampling ratio
stra_sam_rate=0.2

#number of dataset categories
label_TCGA=15
TCGA_fold_loc=str_sam_get_5(TCGA_label,stra_sam_rate,label_TCGA)

#number of selected gene symbol :from 10 to 1000 step length is 10
gene_sel_num=np.arange(10,1000,10)
acc_val=[[],[],[],[]]
for num in gene_sel_num:
    gene_loc=[]
    gene_sel=list(TCGA_gene_impor.loc[:num-1,'gene_name'].values)
    for i in gene_sel:
        gene_loc.append(TCGA_gene_name.index(i))
    acc1,acc2,acc3,acc4=0,0,0,0
    for i in range(5):
        train_loc=TCGA_fold_loc[i]
        test_loc=list(set(np.arange(len(TCGA_label)))-set(train_loc))
        train_TCGA=TCGA_data[train_loc,:]
        train_TCGA=train_TCGA[:,gene_loc]
        test_TCGA=TCGA_data[test_loc,:]
        test_TCGA=test_TCGA[:,gene_loc]
        TCGA_label=np.array(TCGA_label)
        train_TCGA_label=TCGA_label[train_loc]
        test_TCGA_label=TCGA_label[test_loc]
        
        a1,a2,a3,a4=acc_get(train_TCGA,train_TCGA_label,test_TCGA,test_TCGA_label)
        acc1+=a1
        acc2+=a2
        acc3+=a3
        acc4+=a4
    acc_val[0].append(acc1/5)
    acc_val[1].append(acc2/5)
    acc_val[2].append(acc3/5)
    acc_val[3].append(acc4/5)

#write the accuarcy of models
acc_val_n=np.array(acc_val).T
acc_val_n=pd.DataFrame(acc_val_n)
acc_val_n.to_csv(loc_path+"TCGA_dataset/gene_sel_acc_val.csv",
               header=['xgboost','knn','svm_lin','svm_rbf'])


#get the accuarcy of model
def acc_get(train,train_label,test,test_label):
    smote=SMOTE(k_neighbors=2)
    X_train,y_train=smote.fit_resample(train,train_label)
    X_test,y_test=test,test_label
    
    std=StandardScaler()
    X_train=std.fit_transform(X_train)
    X_test=std.transform(X_test)
    
    #train the model
    clf_xg=XGBClassifier(max_depth=15,learning_rate=0.4,n_jobs=-1)
    clf_xg.fit(X_train,y_train)
    c1=accuracy_score(y_test,clf_xg.predict(X_test))

    
    clf_knn=KNeighborsClassifier(15,'distance',n_jobs=-1)
    clf_knn.fit(X_train,y_train)
    c2=accuracy_score(y_test,clf_knn.predict(X_test))



    clf_lg=LogisticRegression()
    clf_lg.fit(X_train,y_train)
    c3=accuracy_score(y_test,clf_lg.predict(X_test))

    
    clf_rbf = SVC(decision_function_shape='ovo',kernel='rbf',probability=True,random_state=42)
    clf_rbf.fit(X_train,y_train)
    c4=accuracy_score(y_test,clf_rbf.predict(X_test))

    return c1,c2,c3,c4


#preform stratified sampling

#sampling ratio
stra_sam_rate=0.2

#number of dataset categories
label_GEO=10
GEO_fold_loc=str_sam_get_5(GEO_label,stra_sam_rate,label_GEO)

#number of selected gene symbol :from 10 to 600 step length is 10
gene_sel_num=np.arange(10,600,10)
acc_val=[[],[],[],[]]
for num in gene_sel_num:
    gene_loc=[]
    gene_sel=list(GEO_gene_impor.loc[:num-1,'gene_name'].values)
    for i in gene_sel:
        gene_loc.append(GEO_gene_name.index(i))
    print(len(gene_loc))
    acc1,acc2,acc3,acc4=0,0,0,0
    for i in range(5):
        train_loc=GEO_fold_loc[i]
        test_loc=list(set(np.arange(len(GEO_label)))-set(train_loc))
        train_GEO=GEO_data[train_loc,:]
        train_GEO=train_GEO[:,gene_loc]
        test_GEO=GEO_data[test_loc,:]
        test_GEO=test_GEO[:,gene_loc]
        GEO_label=np.array(GEO_label)
        train_GEO_label=GEO_label[train_loc]
        test_GEO_label=GEO_label[test_loc]
        
        a1,a2,a3,a4=acc_get(train_GEO,train_GEO_label,test_GEO,test_GEO_label)
        acc1+=a1
        acc2+=a2
        acc3+=a3
        acc4+=a4
    acc_val[0].append(acc1/5)
    acc_val[1].append(acc2/5)
    acc_val[2].append(acc3/5)
    acc_val[3].append(acc4/5)


#write the accuarcy of models
acc_val_n=np.array(acc_val).T
acc_val_n=pd.DataFrame(acc_val_n)
acc_val_n.to_csv(loc_path+"GEO_dataset/gene_sel_acc_val.csv",
               header=['xgboost','knn','logistics regression','svm_rbf'])



##gene selected of step two training set and testing set
#TCGA：200
#GEO：390
TCGA_test_data=pd.read_csv(loc_path+"TCGA_dataset/test_TCGA.csv",header=None)
GEO_test_data=pd.read_csv(loc_path+"GEO_dataset/test_GEO.csv",header=None)
TCGA_test_data=TCGA_test_data.values
GEO_test_data=GEO_test_data.values


sel_num=200
gene_loc=[]
gene_sel=list(TCGA_gene_impor.loc[:sel_num-1,'gene_name'].values)
for i in gene_sel:
        gene_loc.append(TCGA_gene_name.index(i))
TCGA_data=TCGA_data[:,gene_loc]
print(TCGA_data.shape)

TCGA_data=pd.DataFrame(TCGA_data)
TCGA_data.to_csv(loc_path+"TCGA_dataset/gene_sel_data/train.csv",header=None,index=0)
TCGA_test_data=TCGA_test_data[:,gene_loc]
print(TCGA_test_data.shape)
TCGA_test_data=pd.DataFrame(TCGA_test_data)
TCGA_test_data.to_csv(loc_path+"TCGA_dataset/gene_sel_data/test.csv",header=None,index=0)

sel_num=390
gene_loc=[]
gene_sel=list(GEO_gene_impor.loc[:sel_num-1,'gene_name'].values)
for i in gene_sel:
        gene_loc.append(GEO_gene_name.index(i))
GEO_data=GEO_data[:,gene_loc]
print(GEO_data.shape)
GEO_data=pd.DataFrame(GEO_data)
GEO_data.to_csv(loc_path+"GEO_dataset/gene_sel_data/train.csv",header=None,index=0)


GEO_test_data=GEO_test_data[:,gene_loc]
print(GEO_test_data.shape)
GEO_test_data=pd.DataFrame(GEO_test_data)
GEO_test_data.to_csv(loc_path+"GEO_dataset/gene_sel_data/test.csv",header=None,index=0)

