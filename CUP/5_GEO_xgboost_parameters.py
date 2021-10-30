#!/usr/bin/env python
# coding: utf-8

# In[28]:


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
import umap
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
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")


# In[21]:


#read data
GEO_data=pd.read_csv("./GEO_dataset/gene_sel_data/train.csv",header=None)
GEO_data=GEO_data.values

GEO_label=pd.read_csv("./GEO_dataset/train_GEO_label.csv",header=None)
GEO_label=list(GEO_label.loc[:,0])


# In[3]:


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


# In[4]:


#get accuacy of model
def acc_get(train,train_label,test,test_label,mx,mi,g):
    #smote=SMOTE(k_neighbors=7)
    smote=SMOTE(k_neighbors=2)
    X_train,y_train=smote.fit_resample(train,train_label)
    X_test,y_test=test,test_label
    
    std=StandardScaler()
    X_train=std.fit_transform(X_train)
    X_test=std.transform(X_test)
    
    #模型训练
    clf_xg=XGBClassifier(gamma=g,max_depth=mx,min_child_weight=mi,
                         learning_rate=0.4,booster='gbtree',n_jobs=-1)
    clf_xg.fit(X_train,y_train)
    c1=accuracy_score(y_test,clf_xg.predict(X_test))
    
    return c1


# In[5]:


#run the strarified sampling function
stra_sam_rate=0.2
GEO_fold_loc=str_sam_get_5(GEO_label,stra_sam_rate,11)


# In[15]:


#gamma
#max_depth:ma_num
#min_child_weight:min_num
gamma_num=np.arange(0,3,0.2)
ma_num=np.arange(2,20,1)
mi_num=np.arange(1,6,0.4)

parameters=[]
acc_val=[]
for gamma in tqdm(gamma_num):
    for a in tqdm(ma_num):
        for i in tqdm(mi_num):
            acc1=0
            for i in range(5):
                train_loc=GEO_fold_loc[i]
                test_loc=list(set(np.arange(len(GEO_label)))-set(train_loc))
                train_GEO=GEO_data[train_loc,:]
                test_GEO=GEO_data[test_loc,:]
                GEO_label=np.array(GEO_label)
                train_GEO_label=GEO_label[train_loc]
                test_GEO_label=GEO_label[test_loc]
                    
                a1=acc_get(train_GEO,train_GEO_label,test_GEO,test_GEO_label,a,i,gamma)
                acc1+=a1
            parameters.append([gamma,a,i])
            acc_val.append(acc1/5)


# In[ ]:


#print best parameters
loc1=acc_val.index(max(acc_val))
print(parameters[loc1])


# In[22]:


#training the all training set and write the final parameters to files
smote=SMOTE(k_neighbors=2)
X_train,y_train=smote.fit_resample(GEO_data,GEO_label)

std=StandardScaler()
X_train=std.fit_transform(X_train)

#model training
clf_xg=XGBClassifier(gamma=0.0,max_depth=19,min_child_weight=4,
                     learning_rate=0.4,booster='gbtree',n_jobs=-1)
clf_xg.fit(X_train,y_train)


# In[24]:


import pickle
with open("./model_file/TCGA_clf_xg_all.pickle",'wb') as f:
    pickle.dump(clf_xg,f)


# In[27]:


#stander the testing set
GEO_test=pd.read_csv("./GEO_dataset/gene_sel_data/test.csv",header=None)
GEO_test=GEO_test.values
print(GEO_test.shape)

GEO_test=std.transform(GEO_test)
GEO_test=pd.DataFrame(GEO_test)
GEO_test.to_csv("./GEO_dataset/gene_sel_data/test_standard.csv",header=None,index=0)


# In[29]:


#training the all training set and write the final parameters to files
smote=SMOTE(k_neighbors=2)
X_train,y_train=smote.fit_resample(GEO_data,GEO_label)

std=StandardScaler()
X_train=std.fit_transform(X_train)

clf_knn=KNeighborsClassifier(15,'distance')
clf_knn.fit(X_train,y_train)
with open("./model_file/GEO_clf_knn_all.pickle",'wb') as f:
    pickle.dump(clf_knn,f)
#
clf_lg=LogisticRegression()
clf_lg.fit(X_train,y_train)
with open("./model_file/GEO_clf_lg_all.pickle",'wb') as f:
    pickle.dump(clf_lg,f)


clf_rbf = SVC(decision_function_shape='ovo',kernel='rbf',probability=True,random_state=42)
clf_rbf.fit(X_train,y_train)
with open("./model_file/GEO_clf_rbf_all.pickle",'wb') as f:
    pickle.dump(clf_rbf,f)

