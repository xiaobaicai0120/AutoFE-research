'''
Author: Jiaxin Zheng
Date: 2024-04-03 20:32:42
LastEditors: Jiaxin Zheng
LastEditTime: 2024-04-20 19:32:30
Description: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import torch.nn.functional as F
import numpy as np
import random
import os
import collections

from collections import Counter
import pandas as pd

import math
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt

import rootutils
rootutils.setup_root(__file__,indicator='.project-root', pythonpath=True)

class Args:
    mask_prob = 0.15
    seq_len = 41
    h_head = 32
    class_num_label = 2
    class_num_name = 11
    num_layers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = 0.2
    kernel_size = 3
    topk = 5
    turn_dim = cnn_out_channel = h_dim = 512
    folds = 10
    bidirectional = True
    batch_size = 512 # 512
    epochs = 500
    """1-增加对应的超参数"""
    aug_type_idx = 0 # 指定单独的物种索引
    aug_type_times = 1 
    single_name = "Ceq"  # 指定物种名称以保留tensorboard roc曲线等
    """2-增加存储地址对应的物种名称""" 
    token_dict ={'<PAD>': 0, '<CLS_label>': 1, '<CLS_name>': 2, '<MASK>': 3, 'A': 4, 'T': 5, 'C': 6, 'G': 7}

class Feature_extractor(nn.Module):
    def __init__(self,use_features=Args.h_dim,lstm_hidden=Args.h_dim,\
                      cnn_features=Args.cnn_out_channel,\
                      kernel_size=Args.kernel_size,turn_dim=Args.turn_dim):
        super(Feature_extractor, self).__init__()
        self.bidirectional = Args.bidirectional
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden
        if self.bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.lstm_turn = nn.Linear(self.direction*lstm_hidden, turn_dim)
        self.lstm = nn.GRU(use_features,lstm_hidden,batch_first=True,bidirectional=self.bidirectional)
        self.conv1d = nn.Conv1d(in_channels=use_features,out_channels=cnn_features,kernel_size=kernel_size,padding="same")
        self.globals = nn.Linear(use_features,lstm_hidden)

    def forward(self,batch_feature):

        feature_dim = batch_feature.dim()

        if feature_dim > 3:
            global_out = self.globals(batch_feature[:,0,:,:].squeeze())
            lstm_out,(lstm_f,lstm_b) = self.lstm(batch_feature[:,1,:,:].squeeze())
            lstm_out = self.lstm_turn(lstm_out)
            conv_out  = self.conv1d(batch_feature[:,2,:,:].squeeze().permute(0,2,1))
            conv_out  = conv_out.transpose(-1,-2)

        else:
            global_out = self.globals(batch_feature)
            lstm_out,(lstm_f,lstm_b) = self.lstm(batch_feature)
            lstm_out = self.lstm_turn(lstm_out)
            conv_out  = self.conv1d(batch_feature.permute(0,2,1))
            conv_out  = conv_out.transpose(-1,-2)

        return global_out,lstm_out,conv_out,lstm_f+lstm_b
    
class AutoFEPointer(nn.Module):
    def __init__(self,use_features=Args.h_dim,dropout_rate=Args.dropout,\
                 lstm_hidden=Args.h_dim,cnn_features=Args.cnn_out_channel,\
                 kernel_size=Args.kernel_size, turn_dim=Args.turn_dim):

        super(AutoFEPointer, self).__init__()

        self.feature_extract = Feature_extractor()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.multi_head_att = nn.MultiheadAttention(use_features,Args.h_head)
        self.init_matrix = torch.eye(turn_dim,requires_grad=True).to(Args.device)
#         self.init_matrix = torch.eye(turn_dim,requires_grad=True)
        self.rate_turn = nn.Linear(turn_dim,1)
        self.layer_norm = nn.LayerNorm(turn_dim)

    def forward(self,batch_feature):

        global_out,lstm_out,conv_out,lstm_f = self.feature_extract(batch_feature)
        all_features = torch.stack((global_out,lstm_out,conv_out),dim=1)
        all_features2 = torch.stack((global_out,conv_out),dim=1)
        all_features_high = torch.mean(all_features2,dim=-2)
        all_features_high = torch.cat((all_features_high,lstm_f.unsqueeze(1)),dim=1)
        scores_arrays = self.rate_turn(self.init_matrix)
        scores = torch.matmul(self.layer_norm(all_features_high),scores_arrays)/math.sqrt(scores_arrays.shape[0])
        scores = torch.softmax(torch.sum(scores,dim=-1),dim=-1)
        weighted_features = torch.sum(scores.unsqueeze(-1) * all_features_high, dim=-2)
        all_features = scores.unsqueeze(-1).unsqueeze(-1) * all_features
        return scores,weighted_features,all_features
    
class Block(nn.Module):
    def __init__(self,turn_dim=Args.turn_dim):
        super(Block, self).__init__()
        self.layernorm = nn.LayerNorm(turn_dim)
        self.vote_att = AutoFEPointer()
        self.fc1 = nn.Linear(turn_dim,turn_dim*4)
        self.fc2 = nn.Linear(turn_dim*4,turn_dim)
        self.relu = nn.ReLU()

    def forward(self, batch_feature):

        scores, weighted_features, all_features = self.vote_att(batch_feature)
        all_features = self.fc2(self.relu(self.fc1(all_features)))
        return self.layernorm(all_features),self.layernorm(weighted_features)

class Blocks(nn.Module):
    def __init__(self,class_num=Args.class_num_label,\
             type_num=Args.class_num_name,\
             turn_dim=Args.turn_dim, num_layers=Args.num_layers):
        super(Blocks,self).__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(turn_dim)
        self.type_embed = nn.Embedding(type_num, Args.h_dim)
        self.token_embedding = nn.Embedding(len(Args.token_dict),Args.h_dim)
        self.positional_embedding = nn.Embedding(Args.seq_len, Args.h_dim)
        self.fc = nn.Linear(turn_dim,class_num)

    def forward(self, batch_feature, pos, batch_type):
        type_embed = self.type_embed(batch_type)
        pos_embedding = self.positional_embedding(pos)
        batch_feature = self.token_embedding(batch_feature)
        batch_feature += pos_embedding
        batch_feature += type_embed.unsqueeze(1)
        batch_feature = self.layernorm(batch_feature)
        for block in self.blocks:
            batch_feature,weighted_features = block(batch_feature)
        weighted_features_n = self.layernorm(weighted_features)
        return weighted_features_n,self.fc(weighted_features)

def load_model():
    # Load model
    model = Blocks()
    ckpt_path = 'model/site_species/4mC/Ceq_acc.pkl'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    return Args, model
    
if __name__=='__main__':
    
    load_model()