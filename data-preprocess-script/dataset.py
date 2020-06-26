from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import utils
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix #신기
import pickle

######
# pickle 모듈을 사용시, 원하는 데이터를 자료형의 변경없이 저장, 로드가 가능하다. 
# 모든 파이썬 객체를 저장하고 읽을 수 있다. 
# 저장하거나 불러올 때는 바이트 형식을 사용하여야 한다. 
# pickle.dump(data,file)
######

###################### Chem.AddHs()
# Note the calls to Chem.AddHs() in the examples above. 
# By default RDKit molecules do not have H atoms explicitly present in the graph, 
# but they are important for getting realistic geometries, so they generally should be added. 
# They can always be removed afterwards if necessary with a call to Chem.RemoveHs().
######################


random.seed(0)

def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms() # get number of whole atoms in a file
    H = []
    for i in range(n): #해당 원자의 길이만큼 
        H.append(utils.atom_feature(m, i, None, None))  #m이라는 mol file에서, i번째 index의 atom을 H에 넣음. # donor 나 acceptor는 None
        # return 값은 atom 하나에 대한 (28, )의 feature 들이 계속 append 됨 
        # [(28, ),
        #  (28, ), 
        #  ... 
        #  (28, )] --> mol 안에 들어있는 원자의 갯수만큼 
    H = np.array(H)        
    # 다시 정렬 이후, >>> H.shape = (n,28)
    if is_ligand:
        H = np.concatenate([H, np.zeros((n,34))], 1)
        # 리간드이면 뒤의 (n,28) 을 비워 놓음. ( 뒤가 프로틴 자리인가? )
    else:
        H = np.concatenate([np.zeros((n,34)), H], 1)
        # 리간드가 아니면 (프로틴 이니까) 뒤에 넣고, 앞의 자리(리간드 자리)를 비워둠. 
    return H  # H는 node feature matrix 였군       

class MolDataset(Dataset): # class 상속  torch.utils.data.Dataset 

    def __init__(self, keys, data_dir):
        self.keys = keys
        self.data_dir = data_dir
        # 'keys/train_keys.pkl'

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #idx = 0
        # I think sampler determine how to idx maded , for example SequentialSampler, RandomSampler, or CustomSampler stc
        # In this case, we used our custom sampler called DTISampler
        key = self.keys[idx] # in keys (a mount of tuple pairs) choose one pairs and then give them a index
        # 그러면 pickle 만들기 전에, binding affinity 정보를 같이 넣어서 pickle 로 만들자 
        #with open(self.data_dir+'/'+key, 'rb') as f: # (self.data_dir+'/'+key, 'rb')
        #    pair_list = pickle.load(f)
        #    m1, m2, aff = pair_list[0]
            # m1 이 ligand , m2가 protein 인가 봄 
            # pickle file 만드는 코드를 안 올려놨네 (내가 만들자)
            # test_dude_gene.pkl 이런 형식인거 보니, pkl 안에 tuple 쌍들이 list로 저장함. 
        m1, m2, aff = key

        #prepare ligand
        n1 = m1.GetNumAtoms()
        c1 = m1.GetConformers()[0]  # GetConformers( (Mol)arg1) -> object : Get all the conformers as a tuple
        d1 = np.array(c1.GetPositions())
        adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
        H1 = get_atom_feature(m1, True)

        #prepare protein
        n2 = m2.GetNumAtoms()
        c2 = m2.GetConformers()[0]  # GetConformers( (Mol)arg1) -> object : Get all the conformers as a tuple
        d2 = np.array(c2.GetPositions())
        adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
        H2 = get_atom_feature(m2, False)
        
        #aggregation
        H = np.concatenate([H1, H2], 0)
        agg_adj1 = np.zeros((n1+n2, n1+n2)) # only intra distance information is remained in A1
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1) # intra and inter distance information is remained in A2
        dm = distance_matrix(d1,d2)
        agg_adj2[:n1,n1:] = np.copy(dm)
        agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

        #node indice for aggregation
        valid = np.zeros((n1+n2,)) # shape (n1+n2,)
        valid[:n1] = 1 #from the bottom to n1 -> the value is 1 # if it is ligand part, valid value is 1 
        
        #binding affinities 
        Y = aff

        #if n1+n2 > 300 : return None
        sample = {
                  'H':H, 
                  'A1': agg_adj1,   
                  'A2': agg_adj2, 
                  'Y': Y, 
                  'V': valid, 
                  'key': key, 
                  }

        return sample

"""
>>> torch.utils.data.Sampler
are used to specify the sequence of indices/keys used in data loading. 
They represent iterable objects over the indices to datasets. 
E.g., in the common case with stochastic gradient decent (SGD), 
a Sampler could randomly permute a list of indices and yield each one at a time, 
or yield a small number of them for mini-batch SGD.

A sequential or shuffled sampler will be automatically constructed based on the shuffle argument to a DataLoader. 
Alternatively, users may use the sampler argument to specify a custom Sampler object 
that at each time yields the next index/key to fetch.
"""

class DTISampler(Sampler):

    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self):
        #return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        # p -> "Generate a non-uniform random sample"
        # np.arange(len(self.weights))의 값 사이에서, self.num_samples 만큼 만들고, replace 대체 가능, 
        # p는 
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples

"""
>>> collate_fn
You can use your own collate_fn to process the list of samples to form a batch.
The batch argument is a list with all your samples
"""
def collate_fn(batch): #collate means collect and combine 
    #max_natoms = max([len(item['H']) for item in batch if item is not None])
    # the longest num_atoms in a batch
    max_natoms = 1200

    H = np.zeros((len(batch), max_natoms, 68))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    Y = np.zeros((len(batch),))
    V = np.zeros((len(batch), max_natoms))
    keys = []
    
    for i in range(len(batch)):
        natom = len(batch[i]['H'])
        
        H[i,:natom] = batch[i]['H']
        A1[i,:natom,:natom] = batch[i]['A1']
        A2[i,:natom,:natom] = batch[i]['A2']
        Y[i] = batch[i]['Y']
        V[i,:natom] = batch[i]['V']
        keys.append(batch[i]['key'])

    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    Y = torch.from_numpy(Y).float()
    V = torch.from_numpy(V).float()
    
    return H, A1, A2, Y, V, keys

