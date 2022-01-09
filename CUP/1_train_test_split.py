#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import python toolkit
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
import warnings
from sklearn.svm import SVC
import random
import math
import os
warnings.filterwarnings("ignore")


#home directory
loc_path='../final_coding/'

#check the file directory
if not os.path.exists(loc_path+"TCGA_dataset"):
    print("please create 'TCGA_dataset' fold firstly")
if not os.path.exists(loc_path+"GEO_dataset"):
    print("please create 'GEO_dataset' fold firstly")
if not os.path.exists(loc_path+"model_file"):
    print("please create 'model_file' fold firstly")



#data import
TCGA_data=pd.read_csv(loc_path+"TCGA_data.csv",index_col=0)
print(TCGA_data.shape)
TCGA_label=pd.read_csv(loc_path+"TCGA_label.csv",header=None)
print(TCGA_label.shape)
GEO_data=pd.read_csv(loc_path+"GEO_data.csv",index_col=0)
print(GEO_data.shape)
GEO_label=pd.read_csv(loc_path+"GEO_label.csv",header=None)
print(GEO_label.shape)



#gene symbol name import 
TCGA_gene_name=TCGA_data.columns
TCGA_gene_name=pd.DataFrame(TCGA_gene_name)
TCGA_gene_name.to_csv(loc_path+"TCGA_dataset/TCGA_gene_name.csv",header=None,index=0)

GEO_gene_name=GEO_data.columns
GEO_gene_name=pd.DataFrame(GEO_gene_name)
GEO_gene_name.to_csv(loc_path+"GEO_dataset/GEO_gene_name.csv",header=None,index=0)



#Missing value filling
TCGA_data_1=(TCGA_data.fillna(0)).values
print(TCGA_data_1.shape)
TCGA_label=list(TCGA_label.loc[:,0].values)
print(len(TCGA_label))


GEO_data=(GEO_data.fillna(0)).values
print(GEO_data.shape)
GEO_label=list(GEO_label.loc[:,0].values)
print(len(GEO_label))



#define stratified sampling function
def str_sam_get(label_loc,stra_sam_rate_loc_1,stra_sam_rate_loc_2,class_all):
    class_loc={}
    for i in range(class_all):
        class_loc[i]=[]
    for i in range(len(label_loc)):
        class_loc[label_loc[i]].append(i)
    str_sam_sel=[]
    for i in range(class_all):
        if (len(class_loc[i])<(len(label_loc)*0.2)):
            loc_count=math.ceil(stra_sam_rate_loc_1*len(class_loc[i]))
        else:
            loc_count=math.ceil(stra_sam_rate_loc_2*len(class_loc[i]))
        str_sam_sel.extend(random.sample(class_loc[i],loc_count))
    label_loc=list(set(np.arange(len(label_loc)))-set(str_sam_sel))
    return label_loc,str_sam_sel



#preform stratified sampling

#sampling ratio: normal size and innormal size
stra_sam_rate_1=0.2
stra_sam_rate_2=0.5

#number of dataset categories
label_TCGA=15
label_GEO=10

train_loc_TCGA,test_loc_TCGA=str_sam_get(TCGA_label,stra_sam_rate_1,stra_sam_rate_2,label_TCGA)
train_loc_GEO,test_loc_GEO=str_sam_get(GEO_label,stra_sam_rate_1,stra_sam_rate_2,label_GEO)

train_TCGA=TCGA_data_1[train_loc_TCGA,:]
test_TCGA=TCGA_data_1[test_loc_TCGA,:]
TCGA_label=np.array(TCGA_label)
train_TCGA_label=TCGA_label[train_loc_TCGA]
test_TCGA_label=TCGA_label[test_loc_TCGA]


train_GEO=GEO_data[train_loc_GEO,:]
test_GEO=GEO_data[test_loc_GEO,:]
GEO_label=np.array(GEO_label)
train_GEO_label=GEO_label[train_loc_GEO]
test_GEO_label=GEO_label[test_loc_GEO]


#files output

train_TCGA=pd.DataFrame(train_TCGA)
test_TCGA=pd.DataFrame(test_TCGA)
train_TCGA_label=pd.DataFrame(train_TCGA_label)
test_TCGA_label=pd.DataFrame(test_TCGA_label)
train_TCGA.to_csv(loc_path+"TCGA_dataset/train_TCGA.csv",header=None,index=0)
test_TCGA.to_csv(loc_path+"TCGA_dataset/test_TCGA.csv",header=None,index=0)
train_TCGA_label.to_csv(loc_path+"TCGA_dataset/train_TCGA_label.csv",header=None,index=0)
test_TCGA_label.to_csv(loc_path+"TCGA_dataset/test_TCGA_label.csv",header=None,index=0)
print(train_TCGA.shape)
print(test_TCGA.shape)



train_GEO=pd.DataFrame(train_GEO)
test_GEO=pd.DataFrame(test_GEO)
train_GEO_label=pd.DataFrame(train_GEO_label)
test_GEO_label=pd.DataFrame(test_GEO_label)
train_GEO.to_csv(loc_path+"GEO_dataset/train_GEO.csv",header=None,index=0)
test_GEO.to_csv(loc_path+"GEO_dataset/test_GEO.csv",header=None,index=0)
train_GEO_label.to_csv(loc_path+"GEO_dataset/train_GEO_label.csv",header=None,index=0)
test_GEO_label.to_csv(loc_path+"GEO_dataset/test_GEO_label.csv",header=None,index=0)
print(train_GEO.shape)
print(test_GEO.shape)

