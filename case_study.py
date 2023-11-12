#!/usr/bin/env python
# coding: utf-8

# In[634]:


from numpy.random import seed
import csv
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import scipy.sparse as sp
import math
import copy

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA

import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from pytorchtools import BalancedDataParallel
from radam import RAdam
import torch.nn.functional as F

import networkx as nx

import warnings

warnings.filterwarnings("ignore")

import os
from tensorboardX import SummaryWriter


# In[635]:


file_path="/home/dqw_yjw/DDIMDL/"


# In[636]:


seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def prepare(df_drug, feature_list, mechanism, action, drugA, drugB):
    d_label = {}
    d_feature = {}

    # Transfrom the interaction event to number
    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    event_num = len(count)
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  # vector=[]
    for i in feature_list:
        # vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
        tempvec = feature_vector(i, df_drug)
        vector = np.hstack((vector, tempvec))
    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    # Use the dictionary to obtain feature vector and label
    new_feature = []
    new_label = []

    for i in range(len(d_event)):
        temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
        new_feature.append(temp)
        new_label.append(d_label[d_event[i]])

    new_feature = np.array(new_feature)  # 323539*....
    new_label = np.array(new_label)  # 323539

    return new_feature, new_label, event_num


# In[639]:


def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0])))
    return sim_matrix


# In[640]:


class DDIDataset(Dataset):
    def __init__(self, x, y):
        self.len = len(x)
        self.x_data = torch.from_numpy(x)

        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# In[641]:


class DDIDataset_test(Dataset):
    def __init__(self,x):
        self.len=len(x)
        self.x_data=torch.from_numpy(x)
        
    def __getitem__(self,index):
        return self.x_data[index]
    def __len__(self):
        return self.len


# In[642]:


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


# In[107]:


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X


# In[108]:


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# In[109]:


class AE1(torch.nn.Module):  # Joining together
    def __init__(self, vector_size):
        super(AE1, self).__init__()

        self.vector_size = vector_size

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.att2 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)

        self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.att2(X)
        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)

        return X, X_AE


# In[110]:


class AE2(torch.nn.Module):  # twin network
    def __init__(self, vector_size):
        super(AE2, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE // 2) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE // 2) // 2)

        self.att2 = EncoderLayer((self.vector_size + len_after_AE // 2) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE // 2) // 2, len_after_AE // 2)

        self.l3 = torch.nn.Linear(len_after_AE // 2, (self.vector_size + len_after_AE // 2) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE // 2) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE // 2) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]

        X1 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X1 = self.att2(X1)
        X1 = self.l2(X1)
        X_AE1 = self.dr(self.bn3(self.ac(self.l3(X1))))
        X_AE1 = self.l4(X_AE1)

        X2 = self.dr(self.bn1(self.ac(self.l1(X2))))
        X2 = self.att2(X2)
        X2 = self.l2(X2)
        X_AE2 = self.dr(self.bn3(self.ac(self.l3(X2))))
        X_AE2 = self.l4(X_AE2)

        X = torch.cat((X1, X2), 1)
        X_AE = torch.cat((X_AE1, X_AE2), 1)

        return X, X_AE


# In[111]:


class cov(torch.nn.Module):
    def __init__(self, vector_size):
        super(cov, self).__init__()

        self.vector_size = vector_size

        self.co2_1 = torch.nn.Conv2d(1, 1, kernel_size=(2, cov2KerSize))
        self.co1_1 = torch.nn.Conv1d(1, 1, kernel_size=cov1KerSize)
        self.pool1 = torch.nn.AdaptiveAvgPool1d(len_after_AE)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size // 2]
        X2 = X[:, self.vector_size // 2:]

        X = torch.cat((X1, X2), 0)

        X = X.view(-1, 1, 2, self.vector_size // 2)

        X = self.ac(self.co2_1(X))

        X = X.view(-1, self.vector_size // 2 - cov2KerSize + 1, 1)
        X = X.permute(0, 2, 1)
        X = self.ac(self.co1_1(X))

        X = self.pool1(X)

        X = X.contiguous().view(-1, len_after_AE)

        return X


# In[112]:


class ADDAE(torch.nn.Module):
    def __init__(self, vector_size):
        super(ADDAE, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.att1 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)
        # self.att2=EncoderLayer(len_after_AE//2,bert_n_heads)

        self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]
        X = X1 + X2

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.att1(X)
        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)
        X_AE = torch.cat((X_AE, X_AE), 1)

        return X, X_AE


# In[113]:


class BERT(torch.nn.Module):
    def __init__(self, input_dim, n_heads, n_layers, event_num):
        super(BERT, self).__init__()

        self.ae1 = AE1(input_dim)  # Joining together
        self.ae2 = AE2(input_dim)  # twin loss
        self.cov = cov(input_dim)  # cov
        self.ADDAE = ADDAE(input_dim)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.input_dim = input_dim

        self.layers = torch.nn.ModuleList([EncoderLayer(len_after_AE * 5, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(len_after_AE * 5)

        self.l1 = torch.nn.Linear(len_after_AE * 5, (len_after_AE * 5 + event_num) // 2)
        self.bn1 = torch.nn.BatchNorm1d((len_after_AE * 5 + event_num) // 2)

        self.l2 = torch.nn.Linear((len_after_AE * 5 + event_num) // 2, event_num)

        self.ac = gelu

    def forward(self, X):
        X1, X_AE1 = self.ae1(X)
        X2, X_AE2 = self.ae2(X)

        X3 = self.cov(X)

        X4, X_AE4 = self.ADDAE(X)

        X5 = X1 + X2 + X3 + X4

        X = torch.cat((X1, X2, X3, X4, X5), 1)

        for layer in self.layers:
            X = layer(X)
        X = self.AN(X)

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.l2(X)

        return X, X_AE1, X_AE2, X_AE4


class focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(focal_loss, self).__init__()

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1)  # [B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]

        preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss


class my_loss1(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()

        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4)
        return loss


class my_loss2(nn.Module):
    def __init__(self):
        super(my_loss2, self).__init__()

        self.criteria1 = focal_loss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4)
        return loss


def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y

def BERT_train(model, x_train, y_train):
    model_optimizer = RAdam(model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    x_train = np.vstack((x_train, np.hstack((x_train[:, len(x_train[0]) // 2:], x_train[:, :len(x_train[0]) // 2]))))
    y_train = np.hstack((y_train, y_train))
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    len_train = len(y_train)

    print("arg train len", len(y_train))


    train_dataset = DDIDataset(x_train, np.array(y_train))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


    for epoch in range(epo_num):
        if epoch < epoch_changeloss:
            my_loss = my_loss1()
        else:
            my_loss = my_loss2()

        running_loss = 0.0

        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
            x, y = data

            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(x.size()[0]).cuda()
            inputs = lam * x + (1 - lam) * x[index, :]

            targets_a, targets_b = y, y[index]

            inputs = inputs.to(device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)

            model_optimizer.zero_grad()
            # forward + backward+update
            X, X_AE1, X_AE2, X_AE4 = model(inputs.float())

            loss = lam * my_loss(X, targets_a, inputs, X_AE1, X_AE2, X_AE4) + (1 - lam) * my_loss(X, targets_b, inputs,
                                                                                                  X_AE1, X_AE2, X_AE4)

            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()
            
        print('epoch [%d] loss: %.6f' % (epoch+1,running_loss/len_train/2))
    torch.save(model.state_dict(), file_path+"case_study_model.pt")


# In[653]:
file_path="./"

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_n_heads=4
bert_n_layers=4
drop_out_rating=0.3
batch_size=256
len_after_AE=500
learn_rating=0.00001
epo_num=120
cross_ver_tim=5
cov2KerSize=50
cov1KerSize=25
calssific_loss_weight=5
epoch_changeloss=epo_num//2
weight_decay_rate=0.0001
feature_list = ["smile","target","enzyme"]

def main():
    
    conn = sqlite3.connect("/home/dqw_yjw/DDIMDL/event.db")
    df_drug = pd.read_sql('select * from drug;', conn)
    #print(df_drug)
    extraction = pd.read_sql('select * from extraction;', conn)
    #print(extraction)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    
    drugA_test=[]
    drugB_test=[]
    drug_drug_fea=[]
    drug_drug_score=[]
    drug_drug_type=[]
    train_drug_com=[]
    for i in range(len(drugA)):
        train_drug_com.append(drugA[i]+drugB[i])
        train_drug_com.append(drugB[i] + drugA[i])
    #print(train_drug_com)
    
    new_feature, new_label, event_num=prepare(df_drug,feature_list,mechanism,action,drugA,drugB)
    model=BERT(len(new_feature[0]),bert_n_heads,bert_n_layers,event_num)
    BERT_train(model,new_feature,new_label)
    
    
    
    
    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  #vector=[]
    d_feature={}
    for i in feature_list:
        vector = np.hstack((vector,feature_vector(i, df_drug)))
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]
    
    
    for i in range(len(np.array(df_drug['name']).tolist())):
        
        for j in range(i+1,len(np.array(df_drug['name']).tolist())):
            if (df_drug['name'][i]+df_drug['name'][j]) not in train_drug_com:
                #print(df_drug['name'][i]+df_drug['name'][j])
                drugA_test.append(df_drug['name'][i])
                drugB_test.append(df_drug['name'][j])
                drug_drug_fea.append(np.hstack((d_feature[df_drug['name'][i]],d_feature[df_drug['name'][j]])))
    print(len(drug_drug_fea))
            
    
            
    #print(len(np.array(df_drug['name']).tolist()))
    
    the_model = BERT(len(new_feature[0]),bert_n_heads,bert_n_layers,event_num)
    the_model = torch.nn.DataParallel(the_model) 
    the_model.load_state_dict(torch.load(file_path+"case_study_model.pt"))
    
    
    drug_drug_fea=np.array(drug_drug_fea)
    
    test_dataset = DDIDataset_test(drug_drug_fea)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    for batch_idx,data in enumerate(test_loader,0):
        inputs=data
        inputs=inputs.to(device)
        X,_,_,_=the_model(inputs.float())
        X=F.softmax(X).detach().cpu().numpy()
        drug_drug_type.append(np.argmax(X, axis=1).tolist())
        drug_drug_score.append(np.amax(X, axis=1).tolist())
    
    drug_drug_score=[i for j in drug_drug_score for i in j]
    drug_drug_type=[i for j in drug_drug_type for i in j]
    
    datalist=[]
    datalist.append(drugA_test)
    datalist.append(drugB_test)
    datalist.append(drug_drug_score)
    datalist.append(drug_drug_type)
    
    dataarray=np.array(datalist)
    
    dataarray=dataarray.T


    type0_drugA=[]
    type0_drugB = []
    type0_score = []

    type1_drugA = []
    type1_drugB = []
    type1_score = []

    type2_drugA = []
    type2_drugB = []
    type2_score = []

    type3_drugA = []
    type3_drugB = []
    type3_score = []

    type4_drugA = []
    type4_drugB = []
    type4_score = []

    for i in range(len(dataarray)):
        #print(dataarray[i][3])
        if dataarray[i][3]=='0':
            type0_drugA.append(dataarray[i][0])
            type0_drugB.append(dataarray[i][1])
            type0_score.append(float(dataarray[i][2]))
        elif dataarray[i][3]=='1':
            type1_drugA.append(dataarray[i][0])
            type1_drugB.append(dataarray[i][1])
            type1_score.append(float(dataarray[i][2]))
        elif dataarray[i][3]=='2':
            type2_drugA.append(dataarray[i][0])
            type2_drugB.append(dataarray[i][1])
            type2_score.append(float(dataarray[i][2]))
        elif dataarray[i][3]=='3':
            type3_drugA.append(dataarray[i][0])
            type3_drugB.append(dataarray[i][1])
            type3_score.append(float(dataarray[i][2]))
        elif dataarray[i][3]=='4':
            type4_drugA.append(dataarray[i][0])
            type4_drugB.append(dataarray[i][1])
            type4_score.append(float(dataarray[i][2]))

    datalist0 = []
    datalist0.append(type0_drugA)
    datalist0.append(type0_drugB)
    datalist0.append(type0_score)
    dataarray0 = np.array(datalist0)
    dataarray0 = dataarray0.T
    dataarray0=dataarray0[np.lexsort(dataarray0.T)][-20:,:]


    datalist1 = []
    datalist1.append(type1_drugA)
    datalist1.append(type1_drugB)
    datalist1.append(type1_score)
    dataarray1 = np.array(datalist1)
    dataarray1 = dataarray1.T
    dataarray1 = dataarray1[np.lexsort(dataarray1.T)][-20: , :]

    datalist2 = []
    datalist2.append(type2_drugA)
    datalist2.append(type2_drugB)
    datalist2.append(type2_score)
    dataarray2 = np.array(datalist2)
    dataarray2 = dataarray2.T
    dataarray2 = dataarray2[np.lexsort(dataarray2.T)][-20:, :]

    datalist3 = []
    datalist3.append(type3_drugA)
    datalist3.append(type3_drugB)
    datalist3.append(type3_score)
    dataarray3 = np.array(datalist3)
    dataarray3 = dataarray3.T
    dataarray3 = dataarray3[np.lexsort(dataarray3.T)][-20:, :]

    datalist4 = []
    datalist4.append(type4_drugA)
    datalist4.append(type4_drugB)
    datalist4.append(type4_score)
    dataarray4 = np.array(datalist4)
    dataarray4 = dataarray4.T
    dataarray4 = dataarray4[np.lexsort(dataarray4.T)][-20:, :]






    
    with open(file_path+"case_study_type0.csv","w",encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["drugA","drugB","score","type"])
        writer.writerows(dataarray0)
    with open(file_path+"case_study_type1.csv","w",encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["drugA","drugB","score","type"])
        writer.writerows(dataarray1)
    with open(file_path+"case_study_type2.csv","w",encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["drugA","drugB","score","type"])
        writer.writerows(dataarray2)
    with open(file_path+"case_study_type3.csv","w",encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["drugA","drugB","score","type"])
        writer.writerows(dataarray3)
    with open(file_path+"case_study_type4.csv","w",encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["drugA","drugB","score","type"])
        writer.writerows(dataarray4)


# In[654]:


main()

