import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from utils import * 
import time 

class GAT(nn.Module): 
    # Gated / Skip Connection / Attention 
    def __init__(self):
        
    def forward(self) : 

        return 



class Attention(nn.Module): 
    # 1) sigmoid or softmax
    # 2) residual attention or not  
    def __init__(self, sigmoid=False, identity=False):

        self.sigmoid = sigmoid 
        self.identity = identity

    def forward(self, x, adj): 
        self_attention = torch.matmul(x, x.transpose([0,2,1]))
        #self_attention = torch.einsum('ijl,ikl->ijk', (x,x))
        local_attention = torch.einsum('ijl,ilk->ijk',(adj, self_attention))

        if self.sigmoid : 
            attention = nn.Sigmoid(local_attention)
        else : 
            attention = F.Softmax(local_attention)
        
        if self.identity : 
            attention = adj + adj*attention
        
        return attention 

class 



# divide it by module? or not 
# I think module is better for readable code


# ref 
#https://github.com/seobeomjin/3D_AttentionBranch_GCN/blob/master/model-script/layers_dti.py
#https://github.com/Diego999/pyGAT/blob/master/layers.py
#https://github.com/SeungsuKim/CH485--AI-and-Chemistry/blob/master/Assignments/5.%20GCN/Assignment5_logP_GCN.ipynb