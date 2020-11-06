import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time

class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.n_in = n_in_feature 
        self.n_out = n_out_feature
        self.make_dim_same = nn.Linear(n_in_feature, n_out_feature)
        self.W = nn.Linear(n_out_feature, n_out_feature)
        #self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        if self.n_in!= self.n_out:
            x = self.make_dim_same(x)
        h = self.W(x)
        batch_size = h.size()[0]
        N = h.size()[1]
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        e = e + e.permute((0,2,1)) # e is already attention coefficient 
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, h)
        attention = attention*adj
        h_prime = F.relu(torch.einsum('aij,ajk->aik',(attention, h)))
       
        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1)) # [batch, n_atom, in_dim] 에서 in_dim 만큼 확장해서 
        retval = coeff*x+(1-coeff)*h_prime # 차원 동일하게 맞혀줄 필요 있다.
        return retval


class InnerProductDecoder(nn.Module): 
    """ Decoder for using inner product for prediction. """
    def __init__(self, indim, outdim, dropout_rate, act):
        super(InnerProductDecoder, self).__init__()
        self.linear = nn.Linear(indim, outdim)
        self.dropout_rate = dropout_rate
        self.activation = act 
    
    def forward(self, z): 
        z = self.linear(z)
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        out = self.act(torch.mm(z, z.t())) # what is the result if ,,,, I think it is just dot product
        return out 

