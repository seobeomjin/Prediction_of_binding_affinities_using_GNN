import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers_dti import GAT_gate

N_atom_features = 34  #28

class abggcn(torch.nn.Module):
    def __init__(self, args):
        super(abggcn, self).__init__()
        n_graph_layer = args.n_graph_layer
        if args.d_graph_layer_custom!= None : 
            d_graph_layer = args.d_graph_layer_custom
            print("length of the list has to be n_graph_layer+1, {}".format(n_graph_layer+1))
            print("custom graph dimension {}".format(d_graph_layer))
        else : 
            d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate 

        if args.d_graph_layer_custom!= None :
            self.layers1 = [d_graph_layer[i] for i in range(n_graph_layer+1)]
        else : 
            self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]

        self.gconv1 = nn.ModuleList([GAT_gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)]) 
        self.att_conv = GAT_gate(self.layers1[-2], self.layers1[-1])
        
        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i==0 else
                                 nn.Linear(d_FC_layer, 1) if i==n_FC_layer-1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])

        self.att_FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i==0 else
                                 nn.Linear(d_FC_layer, 1) if i==n_FC_layer-1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])
        
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        if args.d_graph_layer_custom!= None :
            self.embede = nn.Linear(2*N_atom_features, d_graph_layer[0], bias = False)
        else : 
            self.embede = nn.Linear(2*N_atom_features, d_graph_layer, bias = False)


        self.bn = nn.BatchNorm1d(1200)
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d((1,self.layers1[-1]))
        

    def embede_graph(self, data):
        c_hs, c_adjs1, c_adjs2, c_valid = data
        c_hs = self.embede(c_hs)
        hs_size = c_hs.size()
        c_adjs2 = torch.exp(-torch.pow(c_adjs2-self.mu.expand_as(c_adjs2), 2)/self.dev) + c_adjs1
        regularization = torch.empty(len(self.gconv1), device=c_hs.device)

        for k in range(len(self.gconv1)):
            c_hs1 = self.gconv1[k](c_hs, c_adjs1)
            c_hs2 = self.gconv1[k](c_hs, c_adjs2)
            c_hs = c_hs2-c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)

        #self.att = self.sigmoid(self.bn(c_hs))
        #c_hs = c_hs*c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1)) #leave only ligand data
        #c_hs = c_hs.sum(1) #[batch, n_out_features]
        return c_hs

    def fully_connected(self, c_hs):
        regularization = torch.empty(len(self.FC)*1-1, device=c_hs.device)

        for k in range(len(self.FC)):
            #c_hs = self.FC[k](c_hs)
            if k<len(self.FC)-1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)

        c_hs =  F.relu(c_hs) # <- torch.sigmoid (because our target label is a continuous value)

        return c_hs

    def att_fully_connected(self, c_hs):
        regularization = torch.empty(len(self.att_FC)*1-1, device=c_hs.device)

        for k in range(len(self.att_FC)):
            #c_hs = self.FC[k](c_hs)
            if k<len(self.att_FC)-1:
                c_hs = self.att_FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.att_FC[k](c_hs)

        c_hs =  F.relu(c_hs) # <- torch.sigmoid (because our target label is a continuous value)

        return c_hs


    def train_model(self, data):
        c_hs, c_adjs1, c_adjs2, c_valid = data
        c_hs = self.embede_graph(data)#embede a graph to a vector

        #attention prediction 
        att_c_hs1 = self.att_conv(c_hs, c_adjs1)
        att_c_hs2 = self.att_conv(c_hs, c_adjs2)
        att_c_hs = att_c_hs2 - att_c_hs1
        att_c_hs = F.dropout(att_c_hs, p=self.dropout_rate, training=self.training)
        
        att = self.sigmoid(self.bn(att_c_hs))

        att_c_hs = att_c_hs*c_valid.unsqueeze(-1).repeat(1, 1, att_c_hs.size(-1)) #leave only ligand data
        att_pred = self.gap(att_c_hs) # [32, 1, 140]
        att_pred = att_pred.sum(1) # [32, 140]
        att_pred = self.att_fully_connected(att_pred)
        att_pred = att_pred.view(-1)

        # Attention Mechanism 
        c_hs_prime = torch.mul(c_hs, att)
        c_hs_prime = c_hs_prime + c_hs

        c_hs_prime = c_hs_prime*c_valid.unsqueeze(-1).repeat(1, 1, c_hs_prime.size(-1)) #leave only ligand data
        c_hs_prime = c_hs_prime.sum(1) #[batch, n_out_features]

        #perceptron prediction
        pctr_pred = self.fully_connected(c_hs_prime)
        pctr_pred = pctr_pred.view(-1) 

        #note that if you don't use concrete dropout, regularization 1-2 is zero
        return pctr_pred, att_pred
    
