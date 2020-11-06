import torch.nn as nn 
import torch 
from torch.utils.data import DataLoader

import utils
from dataset import MolDataset, collate_fn, DTISampler
# from ABGGnet import ABGGnet 
from abggae import abggae
from optimizer import loss_function  

import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy
import pandas as pd 
import seaborn as sns
import os 
from sklearn.metrics import mean_squared_error
from rdkit import Chem

import time 
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print(s)

def input_file(path):
    """Check if input file exists.""" 

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File {} does not exists.'.format(path))
    return path 

def output_file(path):
    """Check if output file can be created."""
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError('File %s cannot be created (check your permissions).'
                      % path)
    return path

def mae(true, pred): 
    return (np.abs(true - pred)).mean()

def corr(true, pred):
    return scipy.stats.pearsonr(pred, true)

def sd(true, pred): 
    return (((true - pred) ** 2).sum() / (len(true) - 1)) ** 0.5

import argparse
parser = argparse.ArgumentParser()
# for model load
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)

parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)

parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default=300)
# for test
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
parser.add_argument("--num_workers", help="num_workers", type=int, default=1)

#parser.add_argument("--model", help="choose the model", type=str, default='abggae')
parser.add_argument('--data_path', help='enter path of keys', required=True, type=input_file, nargs='+')
parser.add_argument("--save_dir", help="save directory of model parameter", type=output_file, default = './save/')
parser.add_argument("--save_descript", help="A description with save dir", type=str, default=None)
parser.add_argument("--test_keys", help="test keys", type=input_file, default='data/test_keys.pkl')
parser.add_argument("--load_model", required=True, help="using lodaing model", type=input_file, default=False)

args = parser.parse_args()

data_path = args.test_keys 
data_name_ = os.path.splitext(os.path.split(data_path)[1])[0]
data_name = data_name_.split('_')[-2]+'_'+data_name_.split('_')[-1]

save_dir = args.save_dir
# save_dir check if it doesn't exist, make it 
if not os.path.isdir(save_dir):
    try:
        os.makedirs(save_dir)
        print('{} Directory Created'.format(save_dir))
    except OSError: 
        print('Error: Creating directory {}'.format(save_dir))

with open (args.test_keys, 'rb') as fp:
    test_keys = pickle.load(fp)

print (f'Number of test data: {len(test_keys)}')

if args.ngpu>0: 
    cmd = utils.set_cuda_visible_device(args.ngpu)
    os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
model = abggae(args)
print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, args.load_model)

#test dataset
test_dataset = MolDataset(test_keys, args.data_path)
test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

pred_list = []
true_list = []

model.eval()
for i_batch, sample in enumerate(test_dataloader): 
    model.zero_grad()
    H, A1, A2, Y, V, keys = sample 
    H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device), Y.to(device), V.to(device)

    pctr_pred, att_pred, recovered_feature = model.train_model((H, A1, A2, V))

    true_list.append(Y.data.cpu().numpy())
    pred_list.append(pctr_pred.data.cpu().numpy())

true_list = np.concatenate(np.array(true_list), 0)
pred_list = np.concatenate(np.array(pred_list), 0)

test_mse = mean_squared_error(true_list, pred_list)
test_rmse = np.sqrt(test_mse)
test_mae = mae(true_list, pred_list)
test_corr = corr(true_list, pred_list)
test_sd = sd(true_list, pred_list)
  
print(f'RMSE={test_rmse:.3f}   MAE={test_mae:.3f}   SD={test_sd:.3f}   R={test_corr[0]:.3f} (p={test_corr[1]:.2e})')

table = pd.DataFrame()
table['pred'] = pred_list
table['true'] = true_list

grid = sns.jointplot('true', 'pred', data=table, stat_func=None,
                         space=0.0, size=3, s=10, edgecolor='w', ylim=(0, 16), xlim=(0, 16))

grid.ax_joint.text(1, 14, data_name)
grid.ax_joint.set_xticks(range(0, 16, 5))
grid.ax_joint.set_yticks(range(0, 16, 5))
if args.save_descript is not None :
    grid.fig.savefig(save_dir+'/'+args.save_descript+'pred.png')
else : 
    grid.fig.savefig(save_dir+'/'+'pred.png')

#save plots -> linear regressor plot

"""
fig1 , ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
if pre_epoch is not None: 
    ax1.plot(range(pre_epoch+1, pre_epoch+1+len(num_epochs)), train_loss_list, 'b', label='train loss')
    ax1.plot(range(pre_epoch+1, pre_epoch+1+len(num_epochs)), test_loss_list, 'r', label='test loss')
    ax2.plot(range(pre_epoch+1, pre_epoch+1+len(num_epochs)), train_rmse_list, 'b', label='train rmse')
    ax2.plot(range(pre_epoch+1, pre_epoch+1+len(num_epochs)), test_rmse_list, 'b', label='test rmse')
else : 
    ax1.plot(range(len(num_epochs)), train_loss_list, 'b', label='train loss')
    ax1.plot(range(len(num_epochs)), test_loss_list, 'r', label='test loss')
    ax2.plot(range(len(num_epochs)), train_rmse_list, 'b', label='train rmse')
    ax2.plot(range(len(num_epochs)), test_rmse_list, 'r', label='test rmse')
ax1.legend()
ax2.legend()
if args.save_descript is not None : 
    fig1.savefig(save_dir+'/'+args.save_descript+'loss_plt.png')
    fig2.savefig(save_dir+'/'+args.save_descript+'rmse_plt.png')
else : 
    fig1.savefig(save_dir+'/'+'loss_plt.png')
    fig2.savefig(save_dir+'/'+'rmse_plt.png')
"""