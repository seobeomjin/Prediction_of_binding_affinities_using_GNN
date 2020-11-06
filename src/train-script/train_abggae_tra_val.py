import torch.nn as nn 
import torch 
from torch.utils.data import DataLoader

import utils
from dataset import MolDataset, collate_fn, DTISampler
from abggae import abggae
from optimizer import loss_function  

import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy
import os 
from sklearn.metrics import mean_squared_error
from rdkit import Chem

import time 
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print(s)

def input_file(path):
    """Check if input file exists.""" 
    if path is None : 
        return None
    else : 
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
parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default=300)
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
parser.add_argument("--num_workers", help="num_workers", type=int, default=1)

parser.add_argument("--att_mech", help="will you use the attention mechanism?", type=bool, default=True)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)

#parser.add_argument("--n_blocks", help="number of GCN blocks", type=int, default=2)
#parser.add_argument("--n_layer", help="number of layers in a block", type=int, default=3)
#parser.add_argument("--in_dim", help="dimension of input", type=int, default=68)
#parser.add_argument("--out_dim", help="dimmension of output", type=int, default=1)
#parser.add_argument("--hidden_dim", help="dimension of hidden node", type=int, default=128)
#parser.add_argument("--n_fc_layer", help="number of FC layers", type=int, default=4)
#parser.add_argument("--fc_dims", help="FC dims list, num of FC dims is n_fc_layers-1", type=int, nargs="*") 
# nargs - The number of command-line arguments that should be consumed.


#parser.add_argument("--act", help="activation func use of not", type=str, default='relu')
#parser.add_argument("--bn", help="batchnorm use or not", type=bool, default=True)
#parser.add_argument("--atn", help="attention use or not", type=bool, default=True)
#parser.add_argument("--num_head", help="num_head for attention", type=int, default=4)
#parser.add_argument("--sc", help="skip connection type", type=str, default="gsc")
#parser.add_argument("--dropout", help="deopout_rate", type=float, default=0)
#parser.add_argument("--gap", help="Global Average Pooling use or not", type=bool, default=True)

parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--recon_coef", help="coefficient of reconstruction loss", type=float, default=0.5)

parser.add_argument('--data_path', help='enter path of keys', required=True, type=input_file, nargs='+')
parser.add_argument("--save_dir", help="save directory of model parameter", type=output_file, default = './save/')
parser.add_argument("--save_descript", help="A description with save dir", type=str, default=None)
parser.add_argument("--train_keys", help="train keys", type=input_file, default='data/train_keys_except_coreset.pkl')
parser.add_argument("--test_keys", help="test keys", type=input_file, default='data/test_keys_coreset_v2016.pkl')
parser.add_argument("--load_model", help="using lodaing model", type=input_file, default=False)

args = parser.parse_args()


# hyper parameters 
num_epochs = args.epoch 
lr = args.lr 
ngpu = args.ngpu
batch_size = args.batch_size
save_dir = args.save_dir
recon_coef = args.recon_coef

# save_dir check if it doesn't exist, make it 
if not os.path.isdir(save_dir):
    try:
        os.makedirs(save_dir)
        print('{} Directory Created'.format(save_dir))
    except OSError: 
        print('Error occured during Creating directory {}'.format(save_dir))

model_save_dir = save_dir + '/saved_model'
if not os.path.isdir(model_save_dir):
    try:
        os.makedirs(model_save_dir)
        print('{} Directory Created'.format(model_save_dir))
    except OSError: 
        print('Error occured during Creating directory {}'.format(model_save_dir))

# read data. data is stored in format of dictionary. (In this case, not dictionary) Each key has information about protein-ligand complex.
with open (args.train_keys, 'rb') as fp:
    train_keys = pickle.load(fp)
if args.test_keys is not None : 
    with open (args.test_keys, 'rb') as fp:
        test_keys = pickle.load(fp)

# get max_natoms 
max_natoms = 1200  # max_natoms : 1177 -> padding in collate_fn as 1200 

# intialize model 
if args.ngpu>0: 
    cmd = utils.set_cuda_visible_device(args.ngpu)
    os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
model = abggae(args)
print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, args.load_model)

print (f'train dataset : {args.train_keys}')
print (f'test dataset : {args.test_keys}')

pre_epoch = None 
if args.load_model : 
    pre_epoch = os.path.splitext(os.path.split(args.load_model)[1])[0]
    pre_epoch = int(pre_epoch.split('_')[-1])
    print (f'training would be done with {args.epoch} epochs from continued {pre_epoch} epochs.')
else : 
    print (f'training would be done with {args.epoch} epochs')

# train and test dataset 

num_valid_pdb = 1000
num_train_pdb = len(train_keys) - num_valid_pdb
if args.test_keys is not None : 
    num_test_pdb = len(test_keys)
else : 
    num_test_pdb = 'test data is not used in train process.'

print(f'Number of train data: {num_train_pdb}')
print(f'Number of validation data: {num_valid_pdb}') 
print(f'Number of test data: {num_test_pdb}')

train_dataset = MolDataset(train_keys, args.data_path)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [num_train_pdb, num_valid_pdb])
if args.test_keys is not None : 
    test_dataset = MolDataset(test_keys, args.data_path)

train_weights = [1/num_train_pdb for k in range(num_train_pdb)]
valid_weights = [1/num_valid_pdb for k in range(num_valid_pdb)]
if args.test_keys is not None : 
    test_weights = [1/num_test_pdb for k in range(num_test_pdb)]

train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)
valid_sampler = DTISampler(valid_weights, len(valid_weights), replacement=True)
if args.test_keys is not None : 
    test_sampler = DTISampler(test_weights, len(test_weights), replacement=True)

train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, sampler=train_sampler)
valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, sampler=valid_sampler)
if args.test_keys is not None :     
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, sampler=test_sampler)

#optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loss_list = []
test_loss_list = []
valid_loss_list = []

train_rmse_list = []
test_rmse_list = []
valid_rmse_list = []

cur_rmse = None 
best_rmse = None 
best_model = None
best_epoch = None

print("Training ...")
for epoch in range(num_epochs):
    st = time.time()
    #collect losses of each iteration 
    train_losses = []
    train_att_losses = []
    train_pctr_losses = []
    train_recon_losses = []
    
    test_losses = []
    test_att_losses = []
    test_pctr_losses = []
    test_recon_losses = []

    valid_losses = []
    valid_att_losses = []
    valid_pctr_losses = []
    valid_recon_losses = []

    #collect true label of each iteration 
    train_true = []
    test_true = [] 
    valid_true = []

    #collect predicted label of each iteration 
    train_pred = [] 
    test_pred = []
    valid_pred = []

    model.train()
    for i_batch, sample in enumerate(train_dataloader):
        model.zero_grad() 
        H, A1, A2, Y, V, keys = sample 
        H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device), Y.to(device), V.to(device)
        
        # dimension matching 
        #Y = Y.unsqueeze(-1)

        # train 
        #att_pred, pctr_pred = model((H, A1, A2, V))
        pctr_pred, att_pred, recovered_feature = model.train_model((H, A1, A2, V))
        
        att_loss, pctr_loss, f_rec_loss = loss_function(att_pred, pctr_pred, recovered_feature, Y, H)
        loss = (1.0-recon_coef)*(att_loss+pctr_loss) + recon_coef*(f_rec_loss)
        loss.backward()
        optimizer.step()

        #collect loss, true label and predicted label 
        train_losses.append(loss.data.cpu().detach().numpy())     # loss.data.cpu().numpy()  -> error
        train_true.append(Y.data.cpu().detach().numpy())          # loss.data.cpu().numpy()  -> error
        train_pred.append(pctr_pred.data.cpu().detach().numpy())       # loss.data.cpu().numpy()  -> error

        train_att_losses.append(att_loss.data.cpu().detach().numpy())
        train_pctr_losses.append(pctr_loss.data.cpu().detach().numpy())
        train_recon_losses.append(f_rec_loss.data.cpu().detach().numpy())

        # RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
        #if i_batch == 0 : 
        #    print("no matter for model train part")

    model.eval()
    for i_batch, sample in enumerate(valid_dataloader): 
        model.zero_grad()
        H, A1, A2, Y, V, keys = sample 
        H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device), Y.to(device), V.to(device)

        pctr_pred, att_pred, recovered_feature = model.train_model((H, A1, A2, V))

        att_loss, pctr_loss, f_rec_loss = loss_function(pctr_pred, att_pred, recovered_feature, Y, H)
        loss = (1.0-recon_coef)*(att_loss+pctr_loss) + recon_coef*(f_rec_loss)

        valid_losses.append(loss.data.cpu().numpy())
        valid_true.append(Y.data.cpu().numpy())
        valid_pred.append(pctr_pred.data.cpu().numpy())

        valid_att_losses.append(att_loss.data.cpu().detach().numpy())
        valid_pctr_losses.append(pctr_loss.data.cpu().detach().numpy())
        valid_recon_losses.append(f_rec_loss.data.cpu().detach().numpy())

    # for i_batch, sample in enumerate(test_dataloader):
    #     model.zero_grad()
    #     H, A1, A2, Y, V, keys = sample 
    #     H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device), Y.to(device), V.to(device)

    #     #att_pred, pctr_pred = model((H, A1, A2, V))
    #     pctr_pred, att_pred, recovered_feature = model.train_model((H, A1, A2, V))
        
    #     att_loss, pctr_loss, f_rec_loss = loss_function(pctr_pred, att_pred, recovered_feature, Y, H)
    #     loss = (1.0-recon_coef)*(att_loss+pctr_loss) + recon_coef*(f_rec_loss)

    #     test_losses.append(loss.data.cpu().numpy())
    #     test_true.append(Y.data.cpu().numpy())
    #     test_pred.append(pctr_pred.data.cpu().numpy())

    #     test_att_losses.append(att_loss.data.cpu().detach().numpy())
    #     test_pctr_losses.append(pctr_loss.data.cpu().detach().numpy())
    #     test_recon_losses.append(f_rec_loss.data.cpu().detach().numpy())

    train_losses = np.mean(np.array(train_losses))
    train_att_losses = np.mean(np.array(train_att_losses))
    train_pctr_losses = np.mean(np.array(train_pctr_losses))
    train_recon_losses = np.mean(np.array(train_recon_losses))

    # test_losses = np.mean(np.array(test_losses))
    # test_att_losses = np.mean(np.array(test_att_losses))
    # test_pctr_losses = np.mean(np.array(test_pctr_losses))
    # test_recon_losses = np.mean(np.array(test_recon_losses))

    valid_losses = np.mean(np.array(valid_losses))
    valid_att_losses = np.mean(np.array(valid_att_losses))
    valid_pctr_losses = np.mean(np.array(valid_pctr_losses))
    valid_recon_losses = np.mean(np.array(valid_recon_losses))

    train_pred = np.concatenate(np.array(train_pred), 0)
    # test_pred = np.concatenate(np.array(test_pred), 0)
    valid_pred = np.concatenate(np.array(valid_pred), 0)

    train_true = np.concatenate(np.array(train_true), 0)
    # test_true = np.concatenate(np.array(test_true), 0) 
    valid_true = np.concatenate(np.array(valid_true), 0)

    train_mse = mean_squared_error(train_true, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mae(train_true, train_pred) 
    train_corr = corr(train_true, train_pred)
    train_sd = sd(train_true, train_pred)

    # test_mse = mean_squared_error(test_true, test_pred)
    # test_rmse = np.sqrt(test_mse)
    # test_mae = mae(test_true, test_pred)
    # test_corr = corr(test_true, test_pred)
    # test_sd = sd(test_true, test_pred)

    valid_mse = mean_squared_error(valid_true, valid_pred)
    valid_rmse = np.sqrt(valid_mse)
    valid_mae = mae(valid_true, valid_pred)
    valid_corr = corr(valid_true, valid_pred)
    valid_sd = sd(valid_true, valid_pred)

    train_loss_list.append(train_losses)
    # test_loss_list.append(test_losses)
    valid_loss_list.append(valid_losses)

    train_rmse_list.append(train_rmse)
    # test_rmse_list.append(test_rmse)
    valid_rmse_list.append(valid_rmse)

    end = time.time()
    m,s = divmod(end - st, 60)
    if pre_epoch is not None :
        print(f'epoch: {pre_epoch+1+epoch}\ttime: {m:.0f}m {s:.0f}s\ntrain loss={train_losses:.3f}\ttest_loss={test_losses:.3f}') 
        print(f'train att loss={train_att_losses:.3f}   train pctr loss={train_pctr_losses:.3f}   train recon loss={train_recon_losses:.3f}')
        # print(f'test att loss={test_att_losses:.3f}   test pctr loss={test_pctr_losses:.3f}   test recon loss={test_recon_losses:.3f}')
        print(f'valid att loss={valid_att_losses:.3f}   valid pctr loss={valid_pctr_losses:.3f}   valid recon loss={valid_recon_losses:.3f}')
    else : 
        print(f'epoch: {epoch}\ttime: {m:.0f}m {s:.0f}s\ntrain loss={train_losses:.3f}\ttest_loss={test_losses:.3f}')
        print(f'train att loss={train_att_losses:.3f}   train pctr loss={train_pctr_losses:.3f}   train recon loss={train_recon_losses:.3f}')
        # print(f'test att loss={test_att_losses:.3f}   test pctr loss={test_pctr_losses:.3f}   test recon loss={test_recon_losses:.3f}')
        print(f'valid att loss={valid_att_losses:.3f}   valid pctr loss={valid_pctr_losses:.3f}   valid recon loss={valid_recon_losses:.3f}')
    print(f'train_RMSE={train_rmse:.3f}   train_MAE={train_mae:.3f}   train_SD={train_sd:.3f}   train_R={train_corr[0]:.3f} (p={train_corr[1]:.2e})')
    # print(f'test_RMSE={test_rmse:.3f}   test_MAE={test_mae:.3f}   test_SD={test_sd:.3f}   test_R={test_corr[0]:.3f} (p={test_corr[1]:.2e})')
    print(f'valid_RMSE={valid_rmse:.3f}   valid_MAE={valid_mae:.3f}   valid_SD={valid_sd:.3f}   valid_R={valid_corr[0]:.3f} (p={valid_corr[1]:.2e})')
    print()

    if args.save_descript is not None : 
        if pre_epoch is not None: 
            name = model_save_dir + '/save_'+ args.save_descript + str(pre_epoch +1 + epoch)+'.pt'
        else : 
            name = model_save_dir + '/save_'+ args.save_descript + str(epoch)+'.pt'
    else : 
        if pre_epoch is not None: 
            name = model_save_dir + '/save_'+ str(pre_epoch +1 + epoch)+'.pt'
        else : 
            name = model_save_dir + '/save_'+ str(epoch)+'.pt'    
    torch.save(model.state_dict(), name)

    cur_rmse = valid_rmse
    if best_rmse is None :
        best_rmse = cur_rmse 
        best_model = model 
        if pre_epoch is not None:
            best_epoch = pre_epoch+1+epoch
        else : 
            best_epoch = epoch
    elif best_rmse > cur_rmse : 
        best_rmse = cur_rmse
        best_model = model
        if pre_epoch is not None:
            best_epoch = pre_epoch+1+epoch
        else : 
            best_epoch = epoch

#save best model 
if args.save_descript is not None : 
    best_model_name = model_save_dir + '/best_'+ args.save_descript + str(best_epoch) + '.pt'
else : 
    best_model_name = model_save_dir + '/best_'+ str(best_epoch) + '.pt'
torch.save(best_model.state_dict(),best_model_name) 

#save plots
fig1 , ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
if pre_epoch is not None: 
    ax1.plot(range(pre_epoch+1, pre_epoch+1+num_epochs), train_loss_list, 'b', label='train loss')
    # ax1.plot(range(pre_epoch+1, pre_epoch+1+num_epochs), test_loss_list, 'r', label='test loss')
    ax1.plot(range(pre_epoch+1, pre_epoch+1+num_epochs), valid_loss_list, 'g', label='valid loss')

    ax2.plot(range(pre_epoch+1, pre_epoch+1+num_epochs), train_rmse_list, 'b', label='train rmse')
    # ax2.plot(range(pre_epoch+1, pre_epoch+1+num_epochs), test_rmse_list, 'r', label='test rmse')
    ax2.plot(range(pre_epoch+1, pre_epoch+1+num_epochs), valid_rmse_list, 'g', label='valid rmse')
else : 
    ax1.plot(range(num_epochs), train_loss_list, 'b', label='train loss')
    # ax1.plot(range(num_epochs), test_loss_list, 'r', label='test loss')
    ax1.plot(range(num_epochs), valid_loss_list, 'g', label='valid loss')

    ax2.plot(range(num_epochs), train_rmse_list, 'b', label='train rmse')
    # ax2.plot(range(num_epochs), test_rmse_list, 'r', label='test rmse')
    ax2.plot(range(num_epochs), valid_rmse_list, 'g', label='valid rmse')
    
ax1.legend()
ax2.legend()
if args.save_descript is not None : 
    fig1.savefig(save_dir+'/'+args.save_descript+'loss_plt.png')
    fig2.savefig(save_dir+'/'+args.save_descript+'rmse_plt.png')
else : 
    fig1.savefig(save_dir+'/'+'loss_plt.png')
    fig2.savefig(save_dir+'/'+'rmse_plt.png')

print("Training is finished!")
    
    












