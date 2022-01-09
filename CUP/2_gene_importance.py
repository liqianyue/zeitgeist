#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import pickle
warnings.filterwarnings("ignore")




loc_path='../final_coding/'
#read the training set data
train_TCGA=pd.read_csv(loc_path+"TCGA_dataset/train_TCGA.csv",header=None,index_col=0)
train_TCGA_label=pd.read_csv(loc_path+"TCGA_dataset/train_TCGA_label.csv",header=None)
train_GEO=pd.read_csv(loc_path+"GEO_dataset/train_GEO.csv",header=None,index_col=0)
train_GEO_label=pd.read_csv(loc_path+"GEO_dataset/train_GEO_label.csv",header=None)

#read gene symbol names
TCGA_gene_name=pd.read_csv(loc_path+'TCGA_dataset/TCGA_gene_name.csv',header=None)
GEO_gene_name=pd.read_csv(loc_path+'GEO_dataset/GEO_gene_name.csv',header=None)



#oversampling and standardization

smote=SMOTE(k_neighbors=7)
X_train,y_train=smote.fit_resample(train_TCGA,train_TCGA_label)

std=StandardScaler()
X_train=std.fit_transform(X_train)

#train the xgboost model

'''Attention: we have saved the model when first training time, you can read the parameters of the trained model directly----TCGA clf_xg:
with open(loc_path+"model_file/TCGA_clf_xg.pickle",'r') as f:
    clf_xg=pickle.load(f) '''
clf_xg=XGBClassifier(max_depth=15,learning_rate=0.4,booster='gbtree')
clf_xg.fit(X_train,y_train)


#get the importance level of each gene and rank them

xg_impo=np.argsort(clf_xg.feature_importances_)
xg_impo=xg_impo[::-1]


TCGA_gene_name_import=[[],[]]
gene_name=list(TCGA_gene_name.loc[:,0].values)
gene_impor=clf_xg.feature_importances_
for i in range(len(xg_impo)):
    TCGA_gene_name_import[0].append(gene_name[xg_impo[i]])
    TCGA_gene_name_import[1].append(gene_impor[xg_impo[i]])
TCGA_gene_name_import=np.array(TCGA_gene_name_import)
TCGA_gene_name_import=pd.DataFrame(TCGA_gene_name_import.T)
TCGA_gene_name_import.to_csv(loc_path+"TCGA_dataset/TGCA_gene_importance.csv",
                             header=['gene_name','importance'],index=0)


#write the model parameters
import pickle
with open("./model_file/TCGA_clf_xg.pickle",'wb') as f:
    pickle.dump(clf_xg,f)


###run anagin in the dataset of GEO


smote=SMOTE(k_neighbors=7)
X_train,y_train=smote.fit_resample(train_GEO,train_GEO_label)

std=StandardScaler()
X_train=std.fit_transform(X_train)


#with open(loc_path+"model_file/GEO_clf_xg.pickle",'r') as f:
#    clf_xg=pickle.load(f)
clf_xg=XGBClassifier(max_depth=15,learning_rate=0.4,booster='gbtree')
clf_xg.fit(X_train,y_train)


xg_impo=np.argsort(clf_xg.feature_importances_)
xg_impo=xg_impo[::-1]


GEO_gene_name_import=[[],[]]
gene_name=list(GEO_gene_name.loc[:,0].values)
gene_impor=clf_xg.feature_importances_
for i in range(len(xg_impo)):
    GEO_gene_name_import[0].append(gene_name[xg_impo[i]])
    GEO_gene_name_import[1].append(gene_impor[xg_impo[i]])
GEO_gene_name_import=np.array(GEO_gene_name_import)
GEO_gene_name_import=pd.DataFrame(GEO_gene_name_import.T)
GEO_gene_name_import.to_csv(loc_path+"GEO_dataset/GEO_gene_importance.csv",
                             header=['gene_name','importance'],index=0)


with open("./model_file/GEO_clf_xg.pickle",'wb') as f:
    pickle.dump(clf_xg,f)

