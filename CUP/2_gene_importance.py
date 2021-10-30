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
import umap
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


# In[ ]:


#read the training set data
train_TCGA=pd.read_csv("./TCGA_dataset/train_TCGA.csv",index_col=0)
train_TCGA_label=pd.read_csv("./TCGA_dataset/train_TCGA_label.csv",header=None)
GEO_data=pd.read_csv("./GEO_dataset/train_GEO.csv",index_col=0)
GEO_label=pd.read_csv("./GEO_dataset/train_GEO_label.csv",header=None)

#read gene symbol names
TCGA_gene_name=pd.read_csv('./TCGA_dataset/TCGA_gene_name.csv',header=None)
GEO_gene_name=pd.read_csv('./GEO_dataset/GEO_gene_name.csv',header=None)


# In[ ]:


#oversampling and standardization

smote=SMOTE(k_neighbors=7)
X_train,y_train=smote.fit_resample(train_TCGA,train_TCGA_label)

std=StandardScaler()
X_train=std.fit_transform(X_train)

#train the xgboost model

#Attention: to save the second training time, you can read the parameters of the 
#trained model directly----TCGA clf_xg:
#with open("./model_file//TCGA_clf_xg.pickle",'r') as f:
#    clf_xg=pickle.load(f)

clf_xg=XGBClassifier(max_depth=15,learning_rate=0.4,booster='gbtree')
clf_xg.fit(X_train,y_train)


# In[ ]:


#get the importance level of each gene and rank them

xg_impo=np.argsort(clf_xg.feature_importances_)
xg_impo=xg_impo[::-1]


# In[13]:


TCGA_gene_name_import=[[],[]]
gene_name=list(TCGA_gene_name.loc[:,0].values)
gene_impor=clf_xg.feature_importances_
for i in range(len(xg_impo)):
    TCGA_gene_name_import[0].append(gene_name[xg_impo[i]])
    TCGA_gene_name_import[1].append(gene_impor[xg_impo[i]])
TCGA_gene_name_import=np.array(TCGA_gene_name_import)
TCGA_gene_name_import=pd.DataFrame(TCGA_gene_name_import.T)
TCGA_gene_name_import.to_csv("./TCGA_dataset/TGCA_gene_importance.csv",
                             header=['gene_name','importance'],index=0)


# In[14]:


#write the model parameters
import pickle
with open("./model_file//TCGA_clf_xg.pickle",'wb') as f:
    pickle.dump(clf_xg,f)


# In[1]:


###run anagin in the dataset of GEO


# In[ ]:


smote=SMOTE(k_neighbors=7)
X_train,y_train=smote.fit_resample(train_GEO,train_GEO_label)

std=StandardScaler()
X_train=std.fit_transform(X_train)


#with open("./model_file/GEO_clf_xg.pickle",'r') as f:
#    clf_xg=pickle.load(f)

clf_xg=XGBClassifier(max_depth=15,learning_rate=0.4,booster='gbtree')
clf_xg.fit(X_train,y_train)


# In[ ]:


xg_impo=np.argsort(clf_xg.feature_importances_)
xg_impo=xg_impo[::-1]


# In[ ]:


GEO_gene_name_import=[[],[]]
gene_name=list(GEO_gene_name.loc[:,0].values)
gene_impor=clf_xg.feature_importances_
for i in range(len(xg_impo)):
    GEO_gene_name_import[0].append(gene_name[xg_impo[i]])
    GEO_gene_name_import[1].append(gene_impor[xg_impo[i]])
GEO_gene_name_import=np.array(GEO_gene_name_import)
GEO_gene_name_import=pd.DataFrame(GEO_gene_name_import.T)
GEO_gene_name_import.to_csv("./GEO_dataset/GEO_gene_importance.csv",
                             header=['gene_name','importance'],index=0)


# In[ ]:


with open("./model_file/GEO_clf_xg.pickle",'wb') as f:
    pickle.dump(clf_xg,f)

