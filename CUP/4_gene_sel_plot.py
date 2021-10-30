#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[2]:


sns.set(style='darkgrid',font_scale=1.3)
plt.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False


# In[157]:


#TCGA_gene
data=pd.read_csv("./TCGA_dataset/gene_sel_acc_val.csv",index_col=0)
data.shape

x=np.arange(100,3000,100)
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
plt.savefig("./TCGA_dataset/plot/TCGA_feature_sel.jpg",dpi=175)

plt.show()


# In[24]:


#GEO_gene
data=pd.read_csv("./GEO_dataset/gene_sel_acc_val.csv",index_col=0)
data.shape

x=np.arange(100,3000,100)
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
plt.savefig("./GEO_dataset/GEO_feature_sel.jpg",dpi=175)
plt.show()


# In[130]:


T_data=pd.read_csv("./TCGA_label.csv",header=None)
T_data=T_data.loc[:,0].values
T=Counter(T_data)
T


# In[131]:


name=['liver','adrenal gland','bladder','breast','cervix',
 'esophagus','pancreas','prostate','stomach','testis',
 'thyroid','thymus','uterus','eye','colorectal']
num=[]
for i in range(15):
    num.append(T[i])
num


# In[133]:


plt.figure(figsize=(13,8))
plt.barh(name,num,0.6)
plt.title("Cancer sample number",x=0.4,y=1.02)
plt.savefig("./TCGA_dataset/plot/TCGA_cancer_number.jpg",dpi=175)


# In[134]:


G_data=pd.read_csv("./GEO_infor.csv",index_col=0)
site=G_data.loc[:,'primary_site'].values


# In[135]:


from collections import Counter
s=Counter(site)


# In[136]:


s


# In[137]:


name=[]
num=[]
for i in s.keys():
    name.append(i)
    num.append(s[i])


# In[ ]:





# In[138]:


plt.figure(figsize=(13,8))
plt.barh(name,num,0.6)
plt.title("Cancer sample number",x=0.4,y=1.02)
plt.savefig("./GEO_dataset/plot/GEO_cancer_number.jpg",dpi=125)


# In[141]:


T=pd.read_csv("./TCGA_dataset/gene_sel_acc_val.csv",index_col=0)
T


# In[142]:


T=pd.read_csv("./GEO_dataset/gene_sel_acc_val.csv",index_col=0)
T


# In[158]:


T=pd.read_csv("./TCGA_dataset/TCGA_gene_importance.csv")
T=T.loc[:,'gene_name'].values[:800]
with open("./TCGA_dataset/TCGA_gene_sel_name.txt","w") as f:
    for i in T:
        f.write(i)
        f.write(",")
f.close()


# In[151]:


T=pd.read_csv("./GEO_dataset/GEO_gene_importance.csv")
T=T.loc[:,'gene_name'].values[:500]

with open("./GEO_dataset/GEO_gene_sel_name.txt","w") as f:
    for i in T:
        f.write(i)
        f.write(",")
f.close()


# In[214]:


#TCGA_gene
data=pd.read_csv("./TCGA_dataset/gene_sel_data/train.csv",header=None)
data=data.values
print(data.shape)
gene_name=pd.read_csv("./TCGA_dataset/TCGA_gene_importance.csv")
gene_name=gene_name.loc[:,'gene_name'].values[:60]
print(len(gene_name))
label=pd.read_csv("./TCGA_dataset/train_TCGA_label.csv",header=None)
label=label.loc[:,0].values
label


# In[215]:


name=['liver','adrenal gland','bladder','breast','cervix',
 'esophagus','pancreas','prostate','stomach','testis',
 'thyroid','thymus','uterus','eye','colorectal']


# In[220]:


num_l=[0]*15
gene_expre=np.zeros((15,60))
for i in range(data.shape[0]):
    num_l[int(label[i])]+=1
    gene_expre[int(label[i])]+=data[i,:60]
for i in range(len(num_l)):
    gene_expre[i]=gene_expre[i]/num_l[i]


# In[221]:


for i in range(gene_expre.shape[0]):
    for j in range(gene_expre.shape[1]):
        gene_expre[i,j]=1-math.exp(-gene_expre[i,j])


# In[250]:


f,ax=plt.subplots(figsize=(40,14))
sns.heatmap(gene_expre,cmap='RdBu_r',ax=ax,linewidths=0.08,
            xticklabels=gene_name,yticklabels=name,center=0.48)
plt.savefig("./TCGA_dataset/plot/TCGA_gene_express.jpg",dpi=175)
plt.show()


# In[3]:


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


# In[4]:


name=['skin','kindey','colorectum','liver','thyroid','breast','lung',
      'other1','tongue','eye','other2']


# In[37]:


num_l=[0]*11
gene_expre=np.zeros((11,60))
for i in range(data.shape[0]):
    num_l[int(label[i])]+=1
    gene_expre[int(label[i])]+=data[i,:60]
for i in range(len(num_l)):
    gene_expre[i]=gene_expre[i]/num_l[i]


# In[38]:


for i in range(gene_expre.shape[0]):
    for j in range(gene_expre.shape[1]):
        if gene_expre[i,j]>600:
            gene_expre[i,j]=600+(gene_expre[i,j]/300)
#        else:
#            gene_expre[i,j]=1/(1+math.exp(-abs(gene_expre[i,j])))


# In[40]:


f,ax=plt.subplots(figsize=(40,12))
sns.heatmap(gene_expre,cmap='RdBu_r',ax=ax,linewidths=0.08,
            xticklabels=gene_name,yticklabels=name)
plt.savefig("./GEO_dataset/plot/GEO_gene_express.jpg",dpi=175)
plt.show()


# In[264]:


Counter(label)

