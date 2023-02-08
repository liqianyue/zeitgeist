#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


sns.set(style='darkgrid',font_scale=1.3)
plt.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False

loc_path='../final_coding/'

#TCGA_gene
data=pd.read_csv(loc_path+"TCGA_dataset/gene_sel_acc_val.csv",index_col=0)
data.shape

x=np.arange(10,1000,10)
plt.figure(figsize=(17,8))
plt.plot(x,data.loc[:,'xgboost'].values,color='r',linestyle='-',label='down',linewidth=2)
plt.plot(x,data.loc[:,'knn'].values,color='m',linestyle='-',label='down',linewidth=2)
plt.plot(x,data.loc[:,'svm_rbf'].values,color='darkcyan',linestyle='-',label='down',linewidth=2)
plt.plot(x,data.loc[:,'svm_lin'].values,color='b',linestyle='-',label='down',linewidth=2)
plt.legend(labels=['xgboost','knn','svm_rbf','svm_lin'],loc='lower right')
plt.xlabel("gene numbers top x")
plt.ylabel("accuarcy")
plt.yticks(np.arange(0.6,1.05,0.05))
plt.xticks(np.arange(0,3000,200))
plt.savefig(loc_path+"TCGA_dataset/plot/TCGA_feature_sel.jpg",dpi=175)


#GEO_gene
data=pd.read_csv(loc_path+"GEO_dataset/gene_sel_acc_val.csv",index_col=0)
data.shape

x=np.arange(10,1000,10)
plt.figure(figsize=(17,8))
plt.plot(x,data.loc[:,'xgboost'].values,color='r',linestyle='-',label='down',linewidth=2)
plt.plot(x,data.loc[:,'knn'].values,color='m',linestyle='-',label='down',linewidth=2)
plt.plot(x,data.loc[:,'svm_rbf'].values,color='darkcyan',linestyle='-',label='down',linewidth=2)
plt.plot(x,data.loc[:,'logistics regression'].values,color='b',linestyle='-',label='down',linewidth=2)
plt.legend(labels=['xgboost','knn','logistics regression','svm_rbf'],loc='lower right')
plt.xlabel("gene numbers top x")
plt.ylabel("accuarcy")
plt.yticks(np.arange(0.4,1,0.05))
plt.xticks(np.arange(0,3000,200))
plt.savefig(loc_path+"GEO_dataset/GEO_feature_sel.jpg",dpi=175)



T=pd.read_csv(loc_path+"TCGA_dataset/TCGA_gene_importance.csv")
T=T.loc[:,'gene_name'].values[:800]
with open(loc_path+"TCGA_dataset/TCGA_gene_sel_name.txt","w") as f:
    for i in T:
        f.write(i)
        f.write(",")
f.close()



T=pd.read_csv(loc_path+"GEO_dataset/GEO_gene_importance.csv")
T=T.loc[:,'gene_name'].values[:500]

with open(loc_path+"GEO_dataset/GEO_gene_sel_name.txt","w") as f:
    for i in T:
        f.write(i)
        f.write(",")
f.close()



#TCGA_gene
data=pd.read_csv(loc_path+"TCGA_dataset/gene_sel_data/train.csv",header=None)
data=data.values
print(data.shape)
gene_name=pd.read_csv(loc_path+"TCGA_dataset/TCGA_gene_importance.csv")
gene_name=gene_name.loc[:,'gene_name'].values[:60]
print(len(gene_name))
label=pd.read_csv(loc_path+"TCGA_dataset/train_TCGA_label.csv",header=None)
label=label.loc[:,0].values
label


name=['liver','adrenal gland','bladder','breast','cervix',
 'esophagus','pancreas','prostate','stomach','testis',
 'thyroid','thymus','uterus','eye','colorectal']



num_l=[0]*15
gene_expre=np.zeros((15,60))
for i in range(data.shape[0]):
    num_l[int(label[i])]+=1
    gene_expre[int(label[i])]+=data[i,:60]
for i in range(len(num_l)):
    gene_expre[i]=gene_expre[i]/num_l[i]



for i in range(gene_expre.shape[0]):
    for j in range(gene_expre.shape[1]):
        gene_expre[i,j]=1-math.exp(-gene_expre[i,j])


f,ax=plt.subplots(figsize=(40,14))
sns.heatmap(gene_expre,cmap='RdBu_r',ax=ax,linewidths=0.08,
            xticklabels=gene_name,yticklabels=name,center=0.48)
plt.savefig("./TCGA_dataset/plot/TCGA_gene_express.jpg",dpi=175)
plt.show()


#GEO_gene
data=pd.read_csv("./GEO_dataset/gene_sel_data/train.csv",header=None)
data=data.values
print(data.shape)
gene_name=pd.read_csv("./GEO_dataset/GEO_gene_importance.csv")
gene_name=gene_name.loc[:,'gene_name'].values[:60]
print(len(gene_name))
label=pd.read_csv("./GEO_dataset/train_GEO_label.csv",header=None)
label=label.loc[:,0].values
label




name=['skin','kindey','colorectum','liver','thyroid','breast','lung',
      'bone','eye','other']



num_l=[0]*10
gene_expre=np.zeros((11,60))
for i in range(data.shape[0]):
    num_l[int(label[i])]+=1
    gene_expre[int(label[i])]+=data[i,:60]
for i in range(len(num_l)):
    gene_expre[i]=gene_expre[i]/num_l[i]



for i in range(gene_expre.shape[0]):
    for j in range(gene_expre.shape[1]):
        if gene_expre[i,j]>600:
            gene_expre[i,j]=600+(gene_expre[i,j]/300)


f,ax=plt.subplots(figsize=(40,12))
sns.heatmap(gene_expre,cmap='RdBu_r',ax=ax,linewidths=0.08,
            xticklabels=gene_name,yticklabels=name)
plt.savefig("./GEO_dataset/plot/GEO_gene_express.jpg",dpi=175)

